import time
import requests
import pandas as pd
import numpy as np
import logging
import signal
import pyupbit
import os
import uuid
import threading
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

import os
from dotenv import load_dotenv

# .env 파일이 상위 폴더에 있는 경우 경로 지정
load_dotenv(dotenv_path="../.env")

ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
SERVER_URL = 'https://api.upbit.com'

# 로깅 설정
logging.basicConfig(filename="auto_trading.log", level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# 전역 변수 초기화 (initialize_trading_data 함수에서 실제 초기화)
buy_prices = {}
total_profit = 0.0
total_invested = 0.0
last_buy_time = {}
highest_prices = {}  # 만약 사용 중이라면

last_sell_signal_check = 0  # ← 이 줄 추가

# 설정값
INTERVAL = 60  # 1분마다 실행
BUY_RATIO_PER_ASSET = 0.2  # 종목당 3% (ATR 반영해 1.5%~3% 사이로 자동 조절)
MINIMUM_ORDER_KRW = 5000
MINIMUM_VOLUME_THRESHOLD = 0.0001
MINIMUM_EVALUATION_KRW = 5000
COOLDOWN_PERIOD_BUY = 180  # 3분으로 단축 (급등시 빠른 분할진입 가능)
COOLDOWN_PERIOD_SELL = 60
SELL_SIGNAL_CHECK_INTERVAL = 300  # 5분마다 매도 신호 확인 (급변 대응)
MAX_CONCURRENT_TRADES = 15  # 상승장 대비 포지션 확장 (시장 흐름에 따라 최대 5도 가능)

stop_trading = threading.Event()

# API 요청 제한 관리
REQUEST_LIMIT_PER_SECOND = 10 # Upbit API 요청 제한 (실제로는 더 보수적으로 10회/초)
REQUEST_COUNT = 0
LAST_REQUEST_TIME = time.time()

# Upbit 객체 초기화
upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

# --- 유틸리티 함수 ---
def rate_limit():
    global REQUEST_COUNT, LAST_REQUEST_TIME
    current_time = time.time()
    if current_time - LAST_REQUEST_TIME >= 1:
        REQUEST_COUNT = 0
        LAST_REQUEST_TIME = current_time
    REQUEST_COUNT += 1
    if REQUEST_COUNT > REQUEST_LIMIT_PER_SECOND:
        sleep_time = 1 - (current_time - LAST_REQUEST_TIME)
        time.sleep(max(sleep_time, 0))

def send_slack_message(message):
    if not SLACK_WEBHOOK_URL:
        logger.error("Slack Webhook URL is not configured.")
        return
    payload = {"text": message}
    try:
        rate_limit()
        response = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error(f"Slack message timed out: {message}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Slack message failed for '{message}': {e}")

def handle_stop_signal(signal_received, frame):
    global total_profit
    send_slack_message(f"Automated trading stopped. Total profit: {total_profit:.2f} KRW.")
    stop_trading.set()
    logger.info("Trading bot stopped by stop signal.")

def get_balance(ticker):
    """지정된 티커의 잔고를 조회합니다. (KRW-BTC -> BTC)"""
    rate_limit()
    balances = upbit.get_balances()
    for b in balances:
        if b["currency"] == ticker.replace("KRW-", ""):
            return float(b.get("balance", 0)), float(b.get("avg_buy_price", 0))
    return 0.0, 0.0 # 잔고, 평균 매수 단가

def get_owned_coins():
    """보유하고 있는 코인들의 목록을 반환합니다."""
    balances = upbit.get_balances()
    owned = {}
    for b in balances:
        if b['currency'] != 'KRW' and float(b['balance']) > MINIMUM_VOLUME_THRESHOLD:
            market_code = f"KRW-{b['currency']}"
            if float(b['balance']) * float(b['avg_buy_price']) >= MINIMUM_EVALUATION_KRW:
                owned[market_code] = float(b['balance'])
    return owned

def place_market_order(side, market, volume=None, price=None):
    """
    시장가 주문을 실행합니다.
    side: "bid" (매수) 또는 "ask" (매도)
    market: 마켓 코드 (예: "KRW-BTC")
    volume: 매도 시 주문 수량
    price: 매수 시 주문 금액 (원화)
    """
    try:
        rate_limit()
        identifier = str(uuid.uuid4())
        order_result = None
        if side == "bid":
            order_result = upbit.buy_market_order(market, price)
            if order_result:
                send_slack_message(f"[Buy Order] Market: {market}, Amount: {price:.2f} KRW, ID: {identifier}")
        elif side == "ask":
            order_result = upbit.sell_market_order(market, volume)
            if order_result:
                send_slack_message(f"[Sell Order] Market: {market}, Volume: {volume:.4f} units, ID: {identifier}")
        
        # 주문 결과 확인 및 UUID 반환 (업비트 API 응답 구조에 따라 유연하게 처리)
        if order_result and 'uuid' in order_result:
            return order_result
        elif order_result and 'error' not in order_result: # 성공했으나 uuid가 없는 경우 (체결 완료)
             return {'uuid': 'immediate_execution', 'state': 'done', **order_result} # 임시 UUID 및 상태 추가
        else:
            logger.error(f"Failed to place {side} order for {market}: {order_result}")
            send_slack_message(f"[Error] Failed to place {side} order for {market}: {order_result}")
            return None

    except Exception as e:
        logger.error(f"Failed to place {side} order for {market}: {e}")
        send_slack_message(f"[Error] Failed to place {side} order for {market}: {e}")
    return None

def get_order(uuid_or_market, state='done'):
    """현재 완료된 주문 정보를 가져오는 함수 (Upbit API Wrapper 사용)"""
    try:
        rate_limit()
        # pyupbit의 get_order는 uuid 또는 ticker_or_uuid를 받음
        result = upbit.get_order(ticker_or_uuid=uuid_or_market, state=state)
        return result
    except Exception as e:
        logger.error(f"Error fetching order {uuid_or_market}: {e}")
        return None

# --- 데이터 로드 함수 ---
def get_markets():
    url = f"{SERVER_URL}/v1/market/all"
    rate_limit()
    response = requests.get(url)
    if response.status_code == 200:
        return [market["market"] for market in response.json() if market["market"].startswith("KRW")]
    logger.error(f"Failed to fetch markets: {response.status_code}")
    return []

def get_candles_minutes(market, unit=1, count=200, retries=3, delay=10):
    url = f"{SERVER_URL}/v1/candles/minutes/{unit}"
    params = {'market': market, 'count': count}
    for attempt in range(retries):
        rate_limit()
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            if df.empty:
                return pd.DataFrame()
            # 컬럼 이름 통일: trade_price, high_price, low_price, open_price, candle_acc_trade_volume
            df = df.rename(columns={
                'trade_price': 'trade_price',
                'high_price': 'high_price',
                'low_price': 'low_price',
                'opening_price': 'open_price',
                'candle_acc_trade_volume': 'candle_acc_trade_volume'
            })
            df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
            return df[['trade_price', 'high_price', 'low_price', 'open_price', 'candle_acc_trade_volume']]
        elif response.status_code == 429:
            logger.warning(f"Rate limit exceeded for {market} (unit {unit}). Retrying after {delay} seconds...")
            time.sleep(delay)
            delay *= 2
        else:
            logger.error(f"Failed to fetch candles for {market} (unit {unit}): {response.status_code}")
            break
    return pd.DataFrame()

def get_candles_minutes_multiple(market, units=[1, 60, 240], count=200):
    all_dfs = {}
    for unit in units:
        df = get_candles_minutes(market, unit=unit, count=count)
        if not df.empty:
            all_dfs[unit] = calculate_indicators(df)
        else:
            all_dfs[unit] = pd.DataFrame()
    return all_dfs

# --- 지표 계산 함수 ---
def calculate_rsi(prices, window=14):
    if len(prices) < window: return pd.Series([np.nan] * len(prices))
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices):
    if len(prices) < 26: return pd.Series([np.nan] * len(prices)), pd.Series([np.nan] * len(prices))
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_adx(df, window=14):
    if len(df) < window + 1: return pd.Series([np.nan] * len(df))
    high = df["high_price"]
    low = df["low_price"]
    close = df["trade_price"]
    plus_dm = high.diff().where(high.diff() > low.diff(), 0.0)
    minus_dm = low.diff().where(low.diff() > high.diff(), 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(window=window).mean()

def calculate_obv(df):
    if df.empty: return df
    obv = [0]
    for i in range(1, len(df)):
        if df["trade_price"].iloc[i] > df["trade_price"].iloc[i - 1]:
            obv.append(obv[-1] + df["candle_acc_trade_volume"].iloc[i])
        elif df["trade_price"].iloc[i] < df["trade_price"].iloc[i - 1]:
            obv.append(obv[-1] - df["candle_acc_trade_volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv
    return df

def detect_fvg(df):
    if len(df) < 3:
        df["fvg"] = False
        return df
    fvg_up = (df["low_price"].shift(2) > df["high_price"])
    fvg_down = (df["high_price"].shift(2) < df["low_price"])
    df["fvg"] = fvg_up | fvg_down
    return df

def calculate_supertrend(df, atr_period=10, multiplier=3):
    if len(df) < atr_period:
        df["supertrend"] = False
        return df
    hl2 = (df["high_price"] + df["low_price"]) / 2
    tr = pd.concat([
        df["high_price"] - df["low_price"],
        (df["high_price"] - df["trade_price"].shift()).abs(),
        (df["low_price"] - df["trade_price"].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period).mean()
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    supertrend_values = [False] * len(df)
    for i in range(1, len(df)):
        if df["trade_price"][i] > upperband[i - 1]:
            supertrend_values[i] = True
        elif df["trade_price"][i] < lowerband[i - 1]:
            supertrend_values[i] = False
        else:
            supertrend_values[i] = supertrend_values[i - 1]
    df["supertrend"] = supertrend_values
    return df

def calculate_volume_momentum(df, window=5):
    if len(df) < window: return pd.Series([np.nan] * len(df))
    return df["candle_acc_trade_volume"].pct_change().rolling(window=window).mean()

def calculate_atr(df, window=14):
    if len(df) < window + 1: return pd.Series([np.nan] * len(df))
    tr = pd.concat([
        df['high_price'] - df['low_price'],
        (df['high_price'] - df['trade_price'].shift()).abs(),
        (df['low_price'] - df['trade_price'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

# --- ICT 기반 지표 함수 ---
def detect_order_block(df, lookback=2):
    """
    불리쉬 오더블록과 베리쉬 오더블록을 감지합니다.
    매우 단순화된 버전이며, 실제 ICT에서는 더 정교한 패턴 분석이 필요합니다.
    """
    df['bullish_ob'] = False
    df['bearish_ob'] = False

    if len(df) < lookback + 1: return df

    for i in range(lookback, len(df)):
        # 불리쉬 오더블록: 하락 후 마지막 하락 양봉 or 하락 음봉 (저점 형성)
        # 예: 큰 음봉 후 작은 양봉 (매수 압력 시작) 또는 매수 거래량 증가하는 구간
        # 현재는 이전 캔들 대비 현재 캔들의 종가가 높고 거래량이 높은 경우를 단순화
        if df['trade_price'].iloc[i] > df['trade_price'].iloc[i-1] and \
           df['candle_acc_trade_volume'].iloc[i] > df['candle_acc_trade_volume'].iloc[i-1] * 1.5:
            df.loc[i, 'bullish_ob'] = True

        # 베리쉬 오더블록: 상승 후 마지막 상승 음봉 or 상승 양봉 (고점 형성)
        # 예: 큰 양봉 후 작은 음봉 (매도 압력 시작) 또는 매도 거래량 증가하는 구간
        # 현재는 이전 캔들 대비 현재 캔들의 종가가 낮고 거래량이 높은 경우를 단순화
        if df['trade_price'].iloc[i] < df['trade_price'].iloc[i-1] and \
           df['candle_acc_trade_volume'].iloc[i] > df['candle_acc_trade_volume'].iloc[i-1] * 1.5:
            df.loc[i, 'bearish_ob'] = True
    return df

def detect_liquidity_sweep(df):
    """
    유동성 스윕 (Liquidity Sweep)을 감지합니다.
    이전 고점/저점을 잠깐 이탈 후 빠르게 되돌리는 움직임.
    """
    df['liquidity_sweep_up'] = False
    df['liquidity_sweep_down'] = False

    if len(df) < 3: return df

    for i in range(2, len(df)):
        # 상승 유동성 스윕 (고점 돌파 후 하락 전환)
        # 이전 고점보다 현재 고점이 높지만, 현재 종가가 이전 종가보다 낮고 음봉으로 마감
        if df['high_price'].iloc[i] > df['high_price'].iloc[i-1] and \
           df['trade_price'].iloc[i] < df['trade_price'].iloc[i-1] and \
           df['trade_price'].iloc[i] < df['open_price'].iloc[i]:
            df.loc[i, 'liquidity_sweep_up'] = True

        # 하락 유동성 스윕 (저점 이탈 후 상승 전환)
        # 이전 저점보다 현재 저점이 낮지만, 현재 종가가 이전 종가보다 높고 양봉으로 마감
        if df['low_price'].iloc[i] < df['low_price'].iloc[i-1] and \
           df['trade_price'].iloc[i] > df['trade_price'].iloc[i-1] and \
           df['trade_price'].iloc[i] > df['open_price'].iloc[i]:
            df.loc[i, 'liquidity_sweep_down'] = True
    return df

def detect_market_structure_shift(df):
    """
    시장 구조 변화 (Market Structure Shift / Change of Character)를 감지합니다.
    추세 전환의 초기 신호.
    """
    df['mss_bullish'] = False
    df['mss_bearish'] = False

    if len(df) < 5: return df # 최소 5개 캔들 필요

    for i in range(4, len(df)):
        # 불리쉬 MSS: 하락 추세 중 직전 고점 돌파 (Higher High) 후, 이전 저점보다 높은 저점 형성 (Higher Low)
        # 예시: LL, LH, LL 패턴에서 LL 이후 LH를 돌파하는 경우
        # 간략화된 조건: 현재 고점이 이전 고점보다 높고, 현재 저점이 2캔들 전 저점보다 높을 때
        if df['high_price'].iloc[i] > df['high_price'].iloc[i-1] and \
           df['low_price'].iloc[i] > df['low_price'].iloc[i-2]:
            df.loc[i, 'mss_bullish'] = True

        # 베리쉬 MSS: 상승 추세 중 직전 저점 이탈 (Lower Low) 후, 이전 고점보다 낮은 고점 형성 (Lower High)
        # 예시: HH, HL, HH 패턴에서 HL을 이탈하는 경우
        # 간략화된 조건: 현재 저점이 이전 저점보다 낮고, 현재 고점이 2캔들 전 고점보다 낮을 때
        if df['low_price'].iloc[i] < df['low_price'].iloc[i-1] and \
           df['high_price'].iloc[i] < df['high_price'].iloc[i-2]:
            df.loc[i, 'mss_bearish'] = True
    return df

def calculate_indicators(df):
    """모든 지표를 계산하여 DataFrame에 추가합니다."""
    if df.empty or 'trade_price' not in df.columns or df['trade_price'].isnull().all():
        logger.warning("Trade price column missing or all null in dataframe, skipping indicator calculation.")
        return df.copy() # 빈 df 또는 문제 있는 df는 복사본 반환

    df["rsi"] = calculate_rsi(df["trade_price"])
    df["macd"], df["macd_signal"] = calculate_macd(df["trade_price"])
    df["adx"] = calculate_adx(df)
    df = calculate_obv(df.copy()) # OBV는 DF 수정하므로 복사본 전달
    df = detect_fvg(df.copy()) # FVG도 DF 수정하므로 복사본 전달
    df = calculate_supertrend(df.copy()) # Supertrend도 DF 수정하므로 복사본 전달
    df["volume_momentum"] = calculate_volume_momentum(df)
    df["atr"] = calculate_atr(df)

    # ICT 지표
    df = detect_order_block(df.copy())
    df = detect_liquidity_sweep(df.copy())
    df = detect_market_structure_shift(df.copy())

    return df

# --- 트레이딩 로직 ---
def initialize_trading_data():
    """프로그램 시작 시 거래 관련 데이터를 초기화합니다."""
    global buy_prices, total_profit, total_invested, last_buy_time
    send_slack_message("[Info] Initializing trading data...")
    buy_prices = {}
    total_profit = 0.0
    total_invested = 0.0
    last_buy_time = {}

    try:
        balances = upbit.get_balances()
        for balance in balances:
            if balance['currency'] != 'KRW' and float(balance['balance']) > 0:
                market_code = f"KRW-{balance['currency']}"
                # Upbit API에서 제공하는 avg_buy_price 활용
                buy_prices[market_code] = float(balance.get('avg_buy_price', 0))
                send_slack_message(f"[Info] Initialized {market_code} with avg buy price: {buy_prices[market_code]:.2f} (from Upbit balances)")
    except Exception as e:
        logger.error(f"Failed to initialize buy prices from balances: {e}")
        send_slack_message(f"[Error] Failed to initialize buy prices: {e}")

def update_highest_price():
    balances = upbit.get_balances()
    for b in balances:
        if b["currency"] == "KRW": continue
        market = f"KRW-{b['currency']}"
        current_price = pyupbit.get_current_price(market)
        prev_high = float(b.get("highest_price", 0))
        if current_price > prev_high:
            b["highest_price"] = current_price  # dict로 유지 필요 (실제로는 DB 또는 파일 저장 권장)

def generate_daily_report():
    try:
        balances = upbit.get_balances()
        valid_markets = set(pyupbit.get_tickers(fiat="KRW"))
        report_lines = ["[Daily Summary]"]

        for b in balances:
            currency = b.get("currency")
            if currency == "KRW":
                continue

            market = f"KRW-{currency}"
            balance = float(b.get("balance", 0))
            avg_buy = float(b.get("avg_buy_price", 0))

            if balance == 0 or avg_buy == 0:
                continue

            if market not in valid_markets:
                report_lines.append(f"{market}: 상장폐지 또는 거래 중단")
                continue

            try:
                current_price = pyupbit.get_current_price(market)
                if current_price is None:
                    report_lines.append(f"{market}: 현재가 조회 실패")
                    continue

                pl = (current_price - avg_buy) * balance
                rate = ((current_price - avg_buy) / avg_buy) * 100
                report_lines.append(f"{market}: {pl:.2f} KRW ({rate:.2f}%)")

            except Exception as e:
                if "Code not found" in str(e):
                    report_lines.append(f"{market}: 마켓 코드 없음 (상장폐지 가능)")
                else:
                    report_lines.append(f"{market}: 가격 조회 오류 - {str(e)}")

        if len(report_lines) == 1:
            report_lines.append("보유 중인 자산이 없습니다.")

        send_slack_message("\n".join(report_lines))

    except Exception as e:
        send_slack_message(f"[Error] 리포트 생성 중 예외 발생: {str(e)}")

def start_report_thread():
    def report_loop():
        while not stop_trading.is_set():
            generate_daily_report()
            time.sleep(86400)  # 하루에 한 번
    threading.Thread(target=report_loop, daemon=True).start()

def track_buy_signals():
    """매수 시그널 추적 및 매수 실행"""
    global total_invested, last_buy_time

    balances = upbit.get_balances()
    balance_krw = 0.0
    total_asset_value = 0.0
    owned_currencies = set()

    for balance in balances:
        if balance['currency'] == 'KRW':
            balance_krw = float(balance['balance'])
        else:
            asset_balance = float(balance['balance'])
            avg_buy_price = float(balance.get('avg_buy_price', 0))
            if avg_buy_price > 0:
                total_asset_value += asset_balance * avg_buy_price
            owned_currencies.add(balance['currency'])

    total_asset_value += balance_krw

    if len(owned_currencies) >= MAX_CONCURRENT_TRADES:
        # logger.info(f"Max concurrent trades ({MAX_CONCURRENT_TRADES}) reached. Skipping buy signals.")
        return

    if balance_krw < MINIMUM_ORDER_KRW:
        logger.warning(f"잔액 부족: {balance_krw:.2f} KRW. 최소 주문 금액: {MINIMUM_ORDER_KRW:.2f} KRW")
        return

    for market in get_markets():
        if market in last_buy_time and (time.time() - last_buy_time[market] < COOLDOWN_PERIOD_BUY):
            continue

        multi_timeframe_dfs = get_candles_minutes_multiple(market, units=[1, 60, 240])
        df_1m = multi_timeframe_dfs.get(1)
        df_60m = multi_timeframe_dfs.get(60)
        df_240m = multi_timeframe_dfs.get(240)

        if df_1m.empty or df_60m.empty or df_240m.empty:
            # logger.warning(f"Failed to get sufficient candle data for {market}. Skipping buy.")
            continue

        current_price = df_1m["trade_price"].iloc[-1]

        existing_volume, _ = get_balance(market.replace("KRW-", ""))
        max_invested_per_coin = total_asset_value * 0.10

        if existing_volume * current_price >= max_invested_per_coin:
            continue

        # NaN 값이 있으면 분석 불가하므로 스킵
        if df_1m.isnull().values.any() or df_60m.isnull().values.any() or df_240m.isnull().values.any():
            logger.warning(f"NaN values detected in indicators for {market}. Skipping buy.")
            continue

        # --- 멀티 타임프레임 추세 확인 (상위 시간봉) ---
        # 240분봉과 60분봉이 상승 추세 (Supertrend True, MACD 골든크로스)
        is_bullish_trend_60m = df_60m["supertrend"].iloc[-1] and \
                                (df_60m["macd"].iloc[-1] > df_60m["macd_signal"].iloc[-1])
        is_bullish_trend_240m = df_240m["supertrend"].iloc[-1] and \
                                 (df_240m["macd"].iloc[-1] > df_240m["macd_signal"].iloc[-1])

        if not (is_bullish_trend_60m and is_bullish_trend_240m):
            continue # 상위 시간봉 추세가 상승이 아니면 매수하지 않음

        # --- ICT 기반 매수 조건 (하위 시간봉 - 1분봉) ---
        rsi = df_1m["rsi"].iloc[-1]
        macd = df_1m["macd"].iloc[-1]
        macd_signal = df_1m["macd_signal"].iloc[-1]
        adx = df_1m["adx"].iloc[-1]
        fvg = df_1m["fvg"].iloc[-1]
        volume_momentum = df_1m["volume_momentum"].iloc[-1]
        supertrend_1m = df_1m["supertrend"].iloc[-1]
        bullish_ob = df_1m['bullish_ob'].iloc[-1]
        liquidity_sweep_down = df_1m['liquidity_sweep_down'].iloc[-1]
        mss_bullish = df_1m['mss_bullish'].iloc[-1]

        # 복합적인 매수 조건: ICT 개념을 더 강하게 적용
        buy_condition_met = (
            supertrend_1m and
            (macd > macd_signal and df_1m["macd"].iloc[-2] <= df_1m["macd_signal"].iloc[-2]) and
            adx > 17 and
            volume_momentum > -0.01 and
            (bullish_ob or fvg or liquidity_sweep_down) and  # and → or
            rsi < 40
        )


        if buy_condition_met:
            send_slack_message(f"[Buy Signal Detected] Market: {market}, Price: {current_price:.2f}")

            # 동적 매수 금액 결정: 자산의 일정 비율 + 변동성 고려
            current_atr = df_1m['atr'].iloc[-1]
            if not np.isnan(current_atr) and current_atr > 0:
                # ATR이 높을수록 변동성이 크므로, 보수적으로 매수 비중 조절
                # (예: ATR이 높을수록 BUY_RATIO_PER_ASSET를 낮춤. 이 로직은 실제 투자 전략에 따라 커스텀 필요)
                # 여기서는 ATR이 높을수록 매수 비중을 살짝 줄이는 예시
                atr_normalized = min(current_atr / current_price, 0.05) # 최대 5%까지 정규화
                adjusted_buy_ratio = BUY_RATIO_PER_ASSET * (1 - atr_normalized * 2) # ATR에 비례하여 비율 감소
                if adjusted_buy_ratio < BUY_RATIO_PER_ASSET * 0.5: # 최소 비율 제한
                    adjusted_buy_ratio = BUY_RATIO_PER_ASSET * 0.5
            else:
                adjusted_buy_ratio = BUY_RATIO_PER_ASSET

            amount = total_asset_value * adjusted_buy_ratio
            if amount > balance_krw:
                amount = balance_krw

            if amount < MINIMUM_ORDER_KRW:
                send_slack_message(f"매수 금액 부족: {amount:.2f} KRW. 최소 주문 금액: {MINIMUM_ORDER_KRW:.2f} KRW for {market}")
                continue

            # 매수 주문 실행
            try:
                order = place_market_order("bid", market, price=amount)
                if order:
                    # 주문 UUID를 사용하여 체결 확인 (비동기 처리)
                    order_uuid = order.get('uuid')
                    if order_uuid == 'immediate_execution': # 즉시 체결된 경우 (가상의 UUID)
                        actual_executed_volume = float(order.get('executed_volume', 0))
                        actual_executed_price = float(order.get('price', current_price))
                        if actual_executed_volume > 0:
                            total_invested += amount # 매수 금액 가산
                            last_buy_time[market] = time.time()
                            send_slack_message(f"[Buy Completed] Market: {market}, Amount: {amount:.2f} KRW (Est.). Immediate confirmation.")
                            # Upbit API에서 가져온 평균 매수 단가로 buy_prices 갱신 (또는 get_balances로 갱신)
                            # initialize_trading_data() # 전체 초기화는 비효율적, 특정 코인만 갱신 필요
                            balance, avg_price = get_balance(market.replace("KRW-", ""))
                            if balance > 0 and avg_price > 0:
                                buy_prices[market] = avg_price
                            else:
                                buy_prices[market] = current_price # 임시 저장
                        else:
                             send_slack_message(f"[Buy Failed] {market} - Order executed but 0 volume. Review Upbit logs.")
                    else: # UUID가 있는 경우
                        # 별도 쓰레드나 루프에서 주문 상태를 지속적으로 확인하는 로직 필요
                        # 현재는 간단히 몇 초 대기 후 확인
                        time.sleep(5) # 주문 체결 대기
                        order_info = get_order(order_uuid)
                        if order_info and order_info['state'] == 'done' and float(order_info['executed_volume']) > 0:
                            executed_price = float(order_info.get('price', current_price)) # 체결 평균 단가
                            executed_volume = float(order_info['executed_volume'])
                            fee = float(order_info.get('paid_fee', 0))
                            
                            total_invested_for_this_order = executed_price * executed_volume + fee # 실제 투자 금액
                            total_invested += total_invested_for_this_order

                            # 평균 매수 단가 업데이트 (기존 buy_prices와 신규 매수 합산)
                            existing_volume, existing_avg_price = get_balance(market.replace("KRW-", ""))
                            if existing_volume > 0 and existing_avg_price > 0:
                                # (기존 매수 총액 + 신규 매수 총액) / (기존 수량 + 신규 수량)
                                new_avg_buy_price = ((existing_avg_price * (existing_volume - executed_volume)) + (executed_price * executed_volume)) / existing_volume
                            else:
                                new_avg_buy_price = executed_price
                            buy_prices[market] = new_avg_buy_price

                            last_buy_time[market] = time.time()
                            send_slack_message(f"[Buy Completed] Market: {market}, Amount: {amount:.2f} KRW, Avg Buy Price: {new_avg_buy_price:.2f}")
                        else:
                            send_slack_message(f"[Buy Order Status] {market} - Order {order_uuid} still processing or failed: {order_info}")
                            logger.warning(f"Buy order {order_uuid} for {market} not confirmed 'done'. Status: {order_info}")
                else:
                    send_slack_message(f"[Error] Failed to place buy order for {market}: Order object was None.")

            except Exception as e:
                logger.error(f"매수 실패: {e} for {market}")
                send_slack_message(f"[Error] 매수 실패: {e} for {market}")

# 전역 변수 선언 (코드 상단에 추가)
highest_prices = {}  # 각 코인별 최고가 저장용

def should_trail_stop(highest_price, current_price, trailing_percent=3.0):
    decline = ((highest_price - current_price) / highest_price) * 100
    return decline >= trailing_percent

def should_clear_inefficient_positions(df_1m, minutes_idle=180):
    if len(df_1m) < minutes_idle:
        return False
    recent_range = df_1m["high_price"].iloc[-minutes_idle:].max() - df_1m["low_price"].iloc[-minutes_idle:].min()
    avg_price = df_1m["trade_price"].iloc[-1]
    volatility = (recent_range / avg_price) * 100
    return volatility < 1.5

def track_sell_signals():
    global total_profit, last_sell_signal_check, highest_prices
    current_time = time.time()

    if current_time - last_sell_signal_check >= SELL_SIGNAL_CHECK_INTERVAL:
        owned_coins = get_owned_coins()
        if owned_coins:
            logger.info(f"[Info] Tracking sell signals for owned coins: {list(owned_coins.keys())}")
        else:
            logger.info("[Info] No coins owned. Sell signal tracking is active.")
        last_sell_signal_check = current_time

    balances = upbit.get_balances()
    for balance in balances:
        if balance['currency'] == 'KRW':
            continue

        market = f"KRW-{balance['currency']}"
        volume = float(balance['balance'])
        avg_buy_price = float(balance.get('avg_buy_price', 0))

        if volume * avg_buy_price < MINIMUM_EVALUATION_KRW:
            continue

        multi_timeframe_dfs = get_candles_minutes_multiple(market, units=[1, 60])
        df_1m = multi_timeframe_dfs.get(1)
        df_60m = multi_timeframe_dfs.get(60)

        if df_1m.empty or df_60m.empty:
            continue
        if df_1m.isnull().values.any() or df_60m.isnull().values.any():
            continue

        current_price = df_1m["trade_price"].iloc[-1]

        # 최고가 갱신
        if market not in highest_prices:
            highest_prices[market] = current_price
        else:
            highest_prices[market] = max(highest_prices[market], current_price)

        atr_1m = df_1m["atr"].iloc[-1]
        supertrend_1m = df_1m["supertrend"].iloc[-1]
        supertrend_prev_1m = df_1m["supertrend"].iloc[-2]
        rsi_1m = df_1m["rsi"].iloc[-1]
        macd_1m = df_1m["macd"].iloc[-1]
        macd_signal_1m = df_1m["macd_signal"].iloc[-1]
        bearish_ob = df_1m['bearish_ob'].iloc[-1]
        liquidity_sweep_up = df_1m['liquidity_sweep_up'].iloc[-1]
        mss_bearish = df_1m['mss_bearish'].iloc[-1]
        fvg = df_1m['fvg'].iloc[-1]

        profit_loss_ratio = ((current_price - avg_buy_price) / avg_buy_price) * 100
        sell_condition_met = False
        sell_volume_ratio = 1.0

        # 손절 조건
        if not np.isnan(atr_1m) and (avg_buy_price - current_price) > (atr_1m * 2.0):
            send_slack_message(f"[Sell SL (ATR)] {market} 손실: {profit_loss_ratio:.2f}%")
            sell_condition_met = True
        elif profit_loss_ratio <= -5.0:
            send_slack_message(f"[Sell SL (Fixed)] {market} 손실: {profit_loss_ratio:.2f}%")
            sell_condition_met = True

        # 익절 조건
        elif not np.isnan(atr_1m) and (current_price - avg_buy_price) > (atr_1m * 3.0):
            send_slack_message(f"[Sell TP (ATR)] {market} 이익: {profit_loss_ratio:.2f}%")
            sell_condition_met = True
            sell_volume_ratio = 0.5
        elif profit_loss_ratio >= 10.0:
            send_slack_message(f"[Sell TP (Fixed)] {market} 이익: {profit_loss_ratio:.2f}%")
            sell_condition_met = True
            sell_volume_ratio = 0.5

        # Supertrend 변곡점
        elif supertrend_prev_1m and not supertrend_1m:
            send_slack_message(f"[Supertrend Flip] {market}")
            sell_condition_met = True
            sell_volume_ratio = 0.7

        # MACD 데드크로스
        elif macd_1m < macd_signal_1m and df_1m["macd"].iloc[-2] > df_1m["macd_signal"].iloc[-2]:
            send_slack_message(f"[MACD Death Cross] {market}")
            sell_condition_met = True
            sell_volume_ratio = 0.5

        # RSI 하락 전환
        elif rsi_1m > 70 and df_1m["rsi"].iloc[-2] > rsi_1m:
            send_slack_message(f"[RSI Overbought Drop] {market}")
            sell_condition_met = True
            sell_volume_ratio = 0.3

        # ICT 조합
        elif bearish_ob and (liquidity_sweep_up or fvg) and mss_bearish:
            send_slack_message(f"[ICT Bearish Combo] {market}")
            sell_condition_met = True
            sell_volume_ratio = 1.0

        # ✅ 추가된 조건 ①: 트레일링 스탑
        elif should_trail_stop(highest_prices[market], current_price):
            send_slack_message(f"[Trailing Stop] {market} 고점대비 하락으로 매도")
            sell_condition_met = True
            sell_volume_ratio = 1.0

        # ✅ 추가된 조건 ②: 변동성 부족한 종목 청산
        elif should_clear_inefficient_positions(df_1m):
            send_slack_message(f"[Low Volatility] {market} 정체 → 매도")
            sell_condition_met = True
            sell_volume_ratio = 1.0

        # 매도 실행
        if sell_condition_met:
            trade_volume = volume * sell_volume_ratio
            if trade_volume * current_price < MINIMUM_ORDER_KRW:
                continue

            order = place_market_order("ask", market, volume=trade_volume)
            if order:
                order_uuid = order.get('uuid')
                if order_uuid == 'immediate_execution':
                    executed_volume = float(order.get('executed_volume', 0))
                    executed_price = float(order.get('price', current_price))
                    profit = (executed_price - avg_buy_price) * executed_volume
                    total_profit += profit
                    send_slack_message(f"[Sell Done] {market} 수익: {profit:.2f} KRW, 총 수익: {total_profit:.2f}")
                else:
                    time.sleep(5)
                    order_info = get_order(order_uuid)
                    if order_info and order_info['state'] == 'done':
                        executed_price = float(order_info.get('price', current_price))
                        executed_volume = float(order_info['executed_volume'])
                        profit = (executed_price - avg_buy_price) * executed_volume
                        total_profit += profit
                        send_slack_message(f"[Sell Done] {market} 수익: {profit:.2f} KRW, 총 수익: {total_profit:.2f}")

# ===== Slack 명령 처리 =====
# def handle_slack_commands():
#     """Slack 명령어 처리 (Poll 방식, 실제 Slack Bot으로 전환 권장)"""
#     # 이 부분은 실제 슬랙 봇 API (Slack Events API)를 사용하는 것이 더 효율적입니다.
#     # 현재는 단순하게 웹훅을 통해 메시지를 확인하는 방식이므로, 실제 사용에는 한계가 있습니다.
#     # 슬랙 봇을 사용하면 특정 채널에서 특정 명령어를 실시간으로 수신할 수 있습니다.
#     logger.info("Starting Slack command handler thread.")
#     while not stop_trading.is_set():
#         try:
#             # Slack 웹훅은 메시지를 보내는 용도이므로, 명령어를 받는 데는 적합하지 않습니다.
#             # 실제로는 Slack Events API를 사용하여 특정 채널의 메시지를 리슨해야 합니다.
#             # 이 예시에서는 "stop trading" 메시지가 어떻게든 수신된다고 가정하고 처리합니다.
#             # (이 부분은 사용자 환경에 맞게 직접 구현이 필요합니다.)
#             # 임시로 더미 로직 또는 수동 테스트를 위한 메시지 확인 로직을 유지
#             # send_slack_message("[Info] Slack command handler is active. Send 'stop trading' to stop.") # 너무 자주 보내지 않도록 주석 처리

#             # 임시: 로컬에서 파일로 명령을 읽는 방식 등 고려 가능
#             # if os.path.exists("stop_command.txt"):
#             #     with open("stop_command.txt", "r") as f:
#             #         command = f.read().strip().lower()
#             #         if "stop trading" in command:
#             #             send_slack_message("[Command Received] Stopping trading bot.")
#             #             os.remove("stop_command.txt")
#             #             os.kill(os.getpid(), signal.SIGTERM)

#         except Exception as e:
#             logger.error(f"Error in Slack command handler: {e}")
#         time.sleep(5) # 5초마다 확인 (폴링 방식)

def handle_slack_commands():
    """Slack 명령어 처리"""
    while not stop_trading.is_set():
        response = requests.get(SLACK_WEBHOOK_URL)
        if response.status_code == 200:
            messages = response.json()
            if "stop trading" in messages.get("text", "").lower():
                send_slack_message("[Command Received] Stopping trading bot.")
                os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)

def main():
    if not ACCESS_KEY or not SECRET_KEY:
        logger.error("ACCESS_KEY or SECRET_KEY is not set in environment variables.")
        send_slack_message("[Error] Upbit API keys are not configured. Exiting.")
        return

    send_slack_message("[Start] Automated trading bot initiated.")
    signal.signal(signal.SIGINT, handle_stop_signal)
    signal.signal(signal.SIGTERM, handle_stop_signal)

    initialize_trading_data()

    start_report_thread()  # ← 리포트 쓰레드 시작

    # Slack 명령 처리 쓰레드 시작 (데몬 쓰레드로 메인 프로그램 종료 시 함께 종료)
    slack_thread = threading.Thread(target=handle_slack_commands, daemon=True)
    slack_thread.start()

    logger.info("Starting main trading loop.")
    while not stop_trading.is_set():
        try:
            track_buy_signals()
            track_sell_signals()
        except Exception as e:
            logger.error(f"Error in main trading loop: {e}", exc_info=True)
            send_slack_message(f"[Critical Error] Main trading loop error: {e}. Check logs.")
        time.sleep(INTERVAL) # 설정된 INTERVAL마다 실행

    logger.info("Main trading loop finished.")

if __name__ == "__main__":
    main()