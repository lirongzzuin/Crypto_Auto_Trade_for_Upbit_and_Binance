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

# 환경 설정
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
SERVER_URL = 'https://api.upbit.com'

# 로깅 설정
logging.basicConfig(filename="auto_trading.log", level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# 변수
buy_prices = {}
total_profit = 0.0
total_invested = 0.0
INTERVAL = 5
buy_ratio = 0.3
MINIMUM_ORDER_KRW = 5000
MINIMUM_VOLUME_THRESHOLD = 0.0001
MINIMUM_EVALUATION_KRW = 5000  # 최소 평가 금액
last_buy_time = {}
COOLDOWN_PERIOD = 5
last_sell_signal_check = 0
SELL_SIGNAL_CHECK_INTERVAL = 1800
stop_trading = threading.Event()  # 프로그램 종료 플래그

# 요청 제한 변수
REQUEST_LIMIT_PER_SECOND = 10
REQUEST_COUNT = 0
LAST_REQUEST_TIME = time.time()

# Upbit 인스턴스
upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

# ===== Helper Functions =====
def rate_limit():
    """API 요청 속도를 제한합니다."""
    global REQUEST_COUNT, LAST_REQUEST_TIME
    current_time = time.time()
    if current_time - LAST_REQUEST_TIME >= 1:
        REQUEST_COUNT = 0
        LAST_REQUEST_TIME = current_time
    REQUEST_COUNT += 1
    if REQUEST_COUNT > REQUEST_LIMIT_PER_SECOND:
        sleep_time = 1 - (current_time - LAST_REQUEST_TIME)
        time.sleep(max(sleep_time, 0))


def calculate_rsi(prices, window=14):
    delta = prices.diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    return macd, macd_signal


def calculate_adx(df, window=14):
    high = df["high_price"]
    low = df["low_price"]
    close = df["trade_price"]
    plus_dm = np.where(high.diff() > low.diff(), high.diff(), 0)
    minus_dm = np.where(low.diff() > high.diff(), low.diff(), 0)
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=window).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=window).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.rolling(window=window).mean()


def calculate_supertrend(df, atr_period=10, multiplier=3):
    atr = calculate_atr(df, atr_period)
    hl2 = (df["high_price"] + df["low_price"]) / 2
    df["supertrend_upper"] = hl2 + (multiplier * atr)
    df["supertrend_lower"] = hl2 - (multiplier * atr)
    df["supertrend"] = np.where(
        df["trade_price"] > df["supertrend_upper"].shift(),
        df["supertrend_upper"],
        np.where(
            df["trade_price"] < df["supertrend_lower"].shift(),
            df["supertrend_lower"],
            np.nan
        )
    )
    df["supertrend"] = df["supertrend"].ffill()
    return df


def calculate_atr(df, window):
    high = df["high_price"]
    low = df["low_price"]
    close = df["trade_price"]
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


def calculate_volume_momentum(df):
    volume = df["candle_acc_trade_volume"]
    return volume.pct_change().rolling(window=5).mean()


def calculate_indicators(df):
    """데이터프레임에 필요한 지표를 추가"""
    df["rsi"] = calculate_rsi(df["trade_price"])
    df["macd"], df["macd_signal"] = calculate_macd(df["trade_price"])
    df["adx"] = calculate_adx(df)
    df = calculate_supertrend(df)
    df["volume_momentum"] = calculate_volume_momentum(df)
    return df


def calculate_dynamic_thresholds(df):
    atr = calculate_atr(df, 14).iloc[-1]
    return {
        "stop_loss": -0.05 - atr / 100,
        "take_profit": 0.07 + atr / 100
    }


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
            return response.json()
        elif response.status_code == 429:  # Too Many Requests
            logger.warning(f"Rate limit exceeded for {market}. Retrying after {delay} seconds...")
            time.sleep(delay)
            delay *= 2
        else:
            logger.error(f"Failed to fetch candles for {market}: {response.status_code}")
            break
    return []


def send_slack_message(message):
    payload = {"text": message}
    try:
        rate_limit()
        requests.post(SLACK_WEBHOOK_URL, json=payload)
    except Exception as e:
        logger.error(f"Slack message failed: {e}")


def get_order_details(market):
    """주문 완료 정보를 가져옵니다."""
    url = f"{SERVER_URL}/v1/orders/closed"
    headers = {"Authorization": f"Bearer {ACCESS_KEY}"}
    params = {"market": market, "state": "done", "order_by": "desc"}

    try:
        rate_limit()
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to fetch order details for {market}: {response.status_code}")
    except Exception as e:
        logger.error(f"Error fetching order details for {market}: {e}")
    return []


def get_balance(ticker):
    rate_limit()
    balances = upbit.get_balances()
    for b in balances:
        if b["currency"] == ticker:
            return float(b.get("balance", 0))
    return 0.0


def place_market_order(side, market, volume=None, price=None):
    try:
        rate_limit()
        identifier = str(uuid.uuid4())
        if side == "bid":
            order = upbit.buy_market_order(market, price)
            send_slack_message(f"[Buy Order] Market: {market}, Amount: {price:.2f} KRW, Identifier: {identifier}")
            return order
        elif side == "ask":
            order = upbit.sell_market_order(market, volume)
            send_slack_message(f"[Sell Order] Market: {market}, Volume: {volume:.4f} units, Identifier: {identifier}")
            return order
    except Exception as e:
        logger.error(f"Failed to place {side} order for {market}: {e}")
        send_slack_message(f"[Error] Failed to place {side} order for {market}: {e}")
    return None

def get_order(uuid_or_market, state='done'):
    """현재 완료된 주문 정보를 가져오는 함수"""
    try:
        if '-' in uuid_or_market:
            result = upbit.get_order(uuid_or_market)
        else:
            result = upbit.get_order(ticker_or_uuid=uuid_or_market, state=state)
        return result
    except Exception as e:
        logger.error(f"Error fetching order: {e}")
        return None

def calculate_profit_from_orders(market):
    """주문 완료 정보를 기준으로 누적 수익 계산"""
    orders = get_order_details(market)
    if not orders:
        return 0.0

    total_profit = 0.0
    for order in orders:
        if order["side"] == "ask":
            sell_price = float(order["price"])
            executed_volume = float(order["executed_volume"])
            buy_price = buy_prices.get(market, 0)
            if buy_price > 0:
                profit = (sell_price - buy_price) * executed_volume - float(order.get("paid_fee", 0))
                total_profit += profit
    return total_profit

def handle_stop_signal(signal_received, frame):
    global total_profit
    send_slack_message(f"Automated trading stopped. Total profit: {total_profit:.2f} KRW.")
    stop_trading.set()  # 프로그램 종료 플래그 설정
    logger.info("Trading bot stopped by stop signal.")

def get_owned_coins():
    balances = upbit.get_balances()
    return {
        f"KRW-{b['currency']}": float(b['balance'])
        for b in balances
        if float(b['balance']) > MINIMUM_VOLUME_THRESHOLD and float(b['balance']) * float(b['avg_buy_price']) >= MINIMUM_EVALUATION_KRW
    }

# def initialize_buy_prices():
#     """보유한 코인의 매수 가격을 초기화합니다."""
#     owned_coins = get_owned_coins()
#     for market in owned_coins.keys():
#         orders = get_order_details(market)
#         if orders:
#             for order in orders:
#                 if order["side"] == "bid" and float(order["executed_volume"]) > 0:
#                     # 가장 최근 매수 가격을 저장
#                     buy_prices[market] = float(order["price"])
#                     break

def track_buy_signals():
    """매수 시그널 추적 및 매수 실행"""
    global total_invested

    # 현재 원화 잔고 및 보유 자산 총 평가액 계산
    balances = upbit.get_balances()
    balance_krw = 0.0
    total_asset_value = 0.0

    for balance in balances:
        if balance['currency'] == 'KRW':
            balance_krw = float(balance['balance'])
        else:
            asset_balance = float(balance['balance'])
            avg_buy_price = float(balance['avg_buy_price'])
            total_asset_value += asset_balance * avg_buy_price

    total_asset_value += balance_krw

    if balance_krw < MINIMUM_ORDER_KRW:
        return

    for market in get_markets():
        candles = get_candles_minutes(market)
        if not candles:
            continue
        df = pd.DataFrame(candles)
        df = calculate_indicators(df)

        if (
                df["rsi"].iloc[-1] < 40 and  # RSI 완화
                df["macd"].iloc[-1] > 0 and  # MACD 양수
                df["adx"].iloc[-1] > 20 and  # ADX 완화
                df["volume_momentum"].iloc[-1] > 0 and  # 거래량 급증 확인
                df["supertrend"].iloc[-1] < df["trade_price"].iloc[-1]
        ):
            send_slack_message(f"[Buy Signal] Market: {market}")
            fee_rate = 0.0005  # 매수 수수료율 (예: 0.05%)
            amount = total_asset_value * buy_ratio  # 총 자산의 일정 비율로 매수
            amount_with_fee = amount * (1 - fee_rate)  # 수수료를 제외한 매수 가능 금액

            if amount_with_fee > balance_krw:
                amount_with_fee = balance_krw * (1 - fee_rate)  # 잔여 원화 잔고로 제한

            if amount_with_fee < MINIMUM_ORDER_KRW:
                continue

            order = place_market_order("bid", market, price=amount_with_fee)
            if order:
                buy_prices[market] = df["trade_price"].iloc[-1]
                total_invested += amount_with_fee
                last_buy_time[market] = time.time()
                send_slack_message(f"[Buy Completed] Market: {market}, Amount: {amount_with_fee:.2f} KRW")


def track_sell_signals():
    """매도 시그널 추적 및 매도 실행"""
    global total_profit, last_sell_signal_check
    current_time = time.time()

    if current_time - last_sell_signal_check >= SELL_SIGNAL_CHECK_INTERVAL:
        owned_coins = get_owned_coins()
        if owned_coins:
            send_slack_message(f"[Info] Tracking sell signals for owned coins: {list(owned_coins.keys())}")
        else:
            send_slack_message("[Info] No coins owned. Sell signal tracking is active.")
        last_sell_signal_check = current_time

    for balance in upbit.get_balances():
        market = f"KRW-{balance['currency']}"
        volume = float(balance['balance'])
        avg_buy_price = float(balance['avg_buy_price'])

        if volume * avg_buy_price < MINIMUM_EVALUATION_KRW:
            continue  # 최소 평가 금액 이하 자산은 매도하지 않음

        candles = get_candles_minutes(market)
        if not candles:
            continue
        df = pd.DataFrame(candles)
        df = calculate_indicators(df)

        current_price = df["trade_price"].iloc[-1]
        profit_ratio = ((current_price - avg_buy_price) / avg_buy_price) * 100  # 수익률 계산

        # 조건 1: 목표 수익률 도달 시 매도
        if profit_ratio >= 9.0:
            send_slack_message(f"[Sell Signal - Target Profit] Market: {market}, Profit Ratio: {profit_ratio:.2f}%")
            order = place_market_order("ask", market, volume=volume)
            if order:
                profit = (current_price - avg_buy_price) * volume
                total_profit += profit
                send_slack_message(
                    f"[Sell Completed] Market: {market}, Profit: {profit:.2f} KRW, Total Profit: {total_profit:.2f} KRW"
                )
                continue

        # 조건 2: Trailing Stop 적용
        if profit_ratio > 1.5 and df["supertrend"].iloc[-1] > df["trade_price"].iloc[-1]:
            send_slack_message(f"[Sell Signal - Trailing Stop] Market: {market}, Profit Ratio: {profit_ratio:.2f}%")
            order = place_market_order("ask", market, volume=volume)
            if order:
                profit = (current_price - avg_buy_price) * volume
                total_profit += profit
                send_slack_message(
                    f"[Sell Completed] Market: {market}, Profit: {profit:.2f} KRW, Total Profit: {total_profit:.2f} KRW"
                )
                continue

        # 조건 3: 손실 한계 도달 시 손절
        if profit_ratio <= -3.0:
            send_slack_message(f"[Sell Signal - Stop Loss] Market: {market}, Loss Ratio: {profit_ratio:.2f}%")
            order = place_market_order("ask", market, volume=volume)
            if order:
                loss = (current_price - avg_buy_price) * volume
                total_profit += loss
                send_slack_message(
                    f"[Sell Completed] Market: {market}, Loss: {loss:.2f} KRW, Total Profit: {total_profit:.2f} KRW"
                )


# ===== Slack 명령 처리 =====
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

# main 함수에서 초기화 호출
def main():
    send_slack_message("[Start] Automated trading started")
    signal.signal(signal.SIGINT, handle_stop_signal)
    signal.signal(signal.SIGTERM, handle_stop_signal)

    # initialize_buy_prices()

    threading.Thread(target=handle_slack_commands, daemon=True).start()

    while not stop_trading.is_set():
        track_buy_signals()
        track_sell_signals()
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
