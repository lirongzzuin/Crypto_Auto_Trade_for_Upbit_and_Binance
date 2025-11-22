import os
import time
import math
import logging
import threading
import signal
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import pyupbit
from dotenv import load_dotenv

pd.set_option('future.no_silent_downcasting', True)

# ================== 환경설정 ==================
load_dotenv(dotenv_path=os.getenv("ENV_PATH", "../.env"))
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# -------- 로깅 --------
logging.basicConfig(
    filename="auto_trading.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("autotrade")

# ================== 전역 상태 ==================
buy_prices: Dict[str, float] = {}
total_profit: float = 0.0
last_buy_time: Dict[str, float] = {}
highest_prices: Dict[str, float] = {}
stop_trading = threading.Event()

# 포지션 메타(초기 리스크/TP 단계/진입시각/스톱)
position_meta: Dict[str, Dict[str, float]] = {}
# 손실 직후 쿨다운 트래킹
last_loss_time: Dict[str, float] = {}

# 일간 드로우다운 컷을 위한 에쿼티 스냅샷
day_start_equity: float = 0.0
day_of_snapshot: str = ""

# 주봉 일목구름 상태 캐시: { market: (timestamp, state_dict) }
weekly_ich_cache: Dict[str, Tuple[float, Dict[str, bool]]] = {}

# ================== 파라미터 ==================
INTERVAL = 10
REQUEST_LIMIT_PER_SECOND = 5
MINIMUM_ORDER_KRW = 5000
MINIMUM_EVALUATION_KRW = 5000

COOLDOWN_PERIOD_BUY = 60
COOLDOWN_AFTER_LOSS_SEC = 900

MAX_CONCURRENT_TRADES = 10
TOP_VOLUME_POOL = 100  # 현재는 사용하지 않지만, 필요시 상위 N개로 제한할 때 활용 가능
PORTFOLIO_BUY_RATIO = 0.30
MIN_BUY_KRW = 6000
MAX_BUY_KRW = 3_000_000

CHASE_UP_PCT_BLOCK = 15.0
RSI_MAX_ENTRY = 90

FIXED_STOP_LOSS = -3.0          # -3% 손절
FIXED_TAKE_PROFIT = 7           # +7% 고정 익절 (현재는 R 베이스와 병행)
ATR_TP_MULT = 3
ATR_SL_MULT = 1.5
TRAILING_STOP_PCT = 2.5

LIMIT_OFFSET_BUY = -0.0005
LIMIT_OFFSET_SELL = +0.0010
LIMIT_TIMEOUT_SEC = 15
FAST_MOVE_PCT = 0.8
MAX_SLIPPAGE_PCT = 0.9

FEE_RATE = 0.0005
SAFETY_BUY_BUFFER = 0.003
SAFETY_SELL_BUFFER = 0.0
RETRY_ON_INSUFFICIENT = 2
RETRY_BACKOFF_SEC = 1.0

# ===== 신규 리스크/필터 파라미터 =====
RISK_PER_TRADE = 0.005           # 계좌 대비 0.5% 위험
TP1_R = 1.0                      # 1R 부분익절
TP2_R = 2.0                      # 2R 추가 익절
BREAK_EVEN_BUFFER_PCT = 0.15     # BE 이동 시 수수료+버퍼(%) 
DAILY_MAX_DRAWDOWN_PCT = 5.0     # 하루 -5% 도달 시 중지
SPREAD_MAX_PCT = 0.30            # 스프레드 한도
MIN_PRICE_KRW = 20               # 리스크 높은 초저가 코인 배제

REQUEST_COUNT = 0
LAST_REQUEST_TIME = time.time()
upbit = None

# ================== 레이트 리밋/Slack ==================
def rate_limit():
    global REQUEST_COUNT, LAST_REQUEST_TIME
    now = time.time()
    if now - LAST_REQUEST_TIME >= 1:
        REQUEST_COUNT = 0
        LAST_REQUEST_TIME = now
    REQUEST_COUNT += 1
    if REQUEST_COUNT > REQUEST_LIMIT_PER_SECOND:
        time.sleep(max(0, 1 - (now - LAST_REQUEST_TIME)))

def send_slack_message(message: str):
    if not SLACK_WEBHOOK_URL:
        return
    try:
        rate_limit()
        requests.post(SLACK_WEBHOOK_URL, json={"text": message}, timeout=5)
    except Exception as e:
        logger.warning(f"Slack send failed: {e}")

def handle_stop_signal(sig, frame):
    send_slack_message("자동매매 종료 신호 수신. 안전 종료합니다.")
    stop_trading.set()

# ================== API 유틸 ==================
def get_markets_krw() -> List[str]:
    try:
        return pyupbit.get_tickers(fiat="KRW")
    except Exception as e:
        logger.error(f"get_tickers failed: {e}")
        return []

def get_ohlcv(market: str, interval: str = "minute1", count: int = 200) -> pd.DataFrame:
    """
    pyupbit.get_ohlcv 래퍼
    interval 예시: "minute1", "minute15", "minute60", "minute240", "day", "week", "month"
    """
    try:
        rate_limit()
        df = pyupbit.get_ohlcv(market, interval=interval, count=count)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "open": "open_price",
            "high": "high_price",
            "low": "low_price",
            "close": "trade_price",
            "volume": "candle_acc_trade_volume",
            "value": "candle_acc_trade_value"
        })
        return df
    except Exception as e:
        logger.warning(f"get_ohlcv({market},{interval}) failed: {e}")
        return pd.DataFrame()

def get_balances_safe():
    try:
        rate_limit()
        return upbit.get_balances()
    except Exception as e:
        logger.warning(f"get_balances failed: {e}")
        return []

def get_current_price_safe(market: str) -> float:
    try:
        rate_limit()
        price = pyupbit.get_current_price(market)
        return float(price) if price is not None else np.nan
    except Exception:
        return np.nan

def get_orderbook_spread_pct(market: str) -> float:
    try:
        rate_limit()
        ob = pyupbit.get_orderbook(tickers=market)
        if not ob:
            return np.inf
        unit = ob[0]
        bids = unit.get("orderbook_units", [])
        if not bids:
            return np.inf
        best_ask = float(bids[0]["ask_price"])
        best_bid = float(bids[0]["bid_price"])
        mid = (best_ask + best_bid) / 2.0
        if mid <= 0:
            return np.inf
        return (best_ask - best_bid) / mid * 100.0
    except Exception:
        return np.inf

def get_balance(currency_or_krw_pair: str) -> Tuple[float, float]:
    cur = currency_or_krw_pair.replace("KRW-", "")
    bals = get_balances_safe()
    for b in bals:
        if b.get("currency") == cur:
            return float(b.get("balance", 0) or 0), float(b.get("avg_buy_price", 0) or 0)
    return 0.0, 0.0

# ================== 안전 접근 유틸 ==================
def safe_val(df: pd.DataFrame, col: str, idx: int, default=np.nan):
    try:
        return df[col].iloc[idx]
    except Exception:
        return default

def ema(series: pd.Series, span: int) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=float)
    return series.ewm(span=span, adjust=False).mean()

# ================== 지표 & ICT 유틸 ==================
def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    if series is None or series.empty or len(series) < window + 2:
        return pd.Series(index=(series.index if series is not None else None), dtype=float)
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).astype(float)

def calculate_macd(series: pd.Series):
    if series is None or series.empty or len(series) < 26:
        idx = series.index if isinstance(series, pd.Series) else None
        empty = pd.Series(index=idx, dtype=float)
        return empty, empty
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.astype(float), signal.astype(float)

def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    if df is None or df.empty or len(df) < window + 1:
        return pd.Series(index=(df.index if df is not None else None), dtype=float)
    high, low, close = df["high_price"], df["low_price"], df["trade_price"]
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean().astype(float)

def calculate_supertrend(df: pd.DataFrame, atr_period=10, multiplier=3.0) -> pd.Series:
    if df is None or df.empty or len(df) < atr_period + 2:
        return pd.Series([False]*len(df), index=(df.index if df is not None else None), dtype=bool)
    hl2 = (df["high_price"] + df["low_price"]) / 2
    tr = pd.concat([
        df["high_price"] - df["low_price"],
        (df["high_price"] - df["trade_price"].shift()).abs(),
        (df["low_price"] - df["trade_price"].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    st = [False]*len(df)
    for i in range(1, len(df)):
        if df["trade_price"].iloc[i] > upper.iloc[i-1]:
            st[i] = True
        elif df["trade_price"].iloc[i] < lower.iloc[i-1]:
            st[i] = False
        else:
            st[i] = st[i-1]
    return pd.Series(st, index=df.index, dtype=bool)

def detect_fvg(df: pd.DataFrame):
    n = len(df)
    bull = pd.Series([False]*n, index=df.index, dtype=bool)
    bear = pd.Series([False]*n, index=df.index, dtype=bool)
    bull_low = pd.Series([np.nan]*n, index=df.index, dtype=float)
    bull_high = pd.Series([np.nan]*n, index=df.index, dtype=float)
    if n < 3:
        return bull, bear, bull_low, bull_high
    cond_bull = df["low_price"] > df["high_price"].shift(2)
    cond_bear = df["high_price"] < df["low_price"].shift(2)
    bull[cond_bull.fillna(False)] = True
    bear[cond_bear.fillna(False)] = True
    bull_low[bull] = df["high_price"].shift(2)[bull]
    bull_high[bull] = df["low_price"][bull]
    return bull, bear, bull_low, bull_high

def detect_liquidity_sweep(df: pd.DataFrame, prev_swing_high: pd.Series, prev_swing_low: pd.Series):
    up = ((df["high_price"] > prev_swing_high) & (df["trade_price"] < prev_swing_high)).fillna(False)
    dn = ((df["low_price"] < prev_swing_low) & (df["trade_price"] > prev_swing_low)).fillna(False)
    return up.astype(bool), dn.astype(bool)

def detect_swings(df: pd.DataFrame, left: int = 2, right: int = 2) -> Tuple[pd.Series, pd.Series]:
    high = df["high_price"]
    low = df["low_price"]
    sh = (high > high.shift(1)) & (high > high.shift(2)) & (high > high.shift(-1)) & (high > high.shift(-2))
    sl = (low < low.shift(1)) & (low < low.shift(2)) & (low < low.shift(-1)) & (low < low.shift(-2))
    return sh.fillna(False).astype(bool), sl.fillna(False).astype(bool)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

    df["ema20"] = ema(df["trade_price"], 20)
    df["ema50"] = ema(df["trade_price"], 50)
    df["rsi"] = calculate_rsi(df["trade_price"])
    macd, macd_sig = calculate_macd(df["trade_price"])
    df["macd"] = macd
    df["macd_signal"] = macd_sig
    df["atr"] = calculate_atr(df)
    df["supertrend"] = calculate_supertrend(df)

    bull_fvg, bear_fvg, bull_low, bull_high = detect_fvg(df)
    df["fvg_bull"] = bull_fvg
    df["fvg_bear"] = bear_fvg
    df["fvg_bull_low_raw"] = bull_low
    df["fvg_bull_high_raw"] = bull_high
    df["fvg_bull_low"] = df["fvg_bull_low_raw"].ffill()
    df["fvg_bull_high"] = df["fvg_bull_high_raw"].ffill()
    df["in_bull_fvg"] = ((df["trade_price"] >= df["fvg_bull_low"]) & (df["trade_price"] <= df["fvg_bull_high"])).fillna(False)

    bear_low_raw = df["high_price"].where(df["fvg_bear"])
    bear_high_raw = df["low_price"].shift(2).where(df["fvg_bear"])
    df["fvg_bear_low"] = bear_low_raw.ffill()
    df["fvg_bear_high"] = bear_high_raw.ffill()
    df["in_bear_fvg"] = ((df["trade_price"] <= df["fvg_bear_low"]) & (df["trade_price"] >= df["fvg_bear_high"])).fillna(False)

    swing_high, swing_low = detect_swings(df)
    df["swing_high"] = swing_high
    df["swing_low"] = swing_low
    sh_price = df["high_price"].where(df["swing_high"])
    sl_price = df["low_price"].where(df["swing_low"])
    df["prev_swing_high"] = sh_price.ffill()
    df["prev_swing_low"] = sl_price.ffill()
    buf = 0.0002
    df["bos_up"] = (df["trade_price"] > (df["prev_swing_high"] * (1 + buf))).fillna(False)
    df["bos_dn"] = (df["trade_price"] < (df["prev_swing_low"] * (1 - buf))).fillna(False)

    df["dr_high"] = df["prev_swing_high"]
    df["dr_low"] = df["prev_swing_low"]
    df["dr_mid"] = (df["dr_high"] + df["dr_low"]) / 2.0
    rng = (df["dr_high"] - df["dr_low"]).replace(0, np.nan)
    df["ote_low"] = df["dr_low"] + 0.62 * rng
    df["ote_high"] = df["dr_low"] + 0.79 * rng
    df["discount_zone"] = (df["trade_price"] <= df["dr_mid"]).fillna(False)
    df["premium_zone"] = (df["trade_price"] >= df["dr_mid"]).fillna(False)
    df["in_ote_long"] = ((df["trade_price"] >= df["ote_low"]) & (df["trade_price"] <= df["ote_high"])).fillna(False)

    ls_up, ls_dn = detect_liquidity_sweep(df, df["prev_swing_high"], df["prev_swing_low"])
    df["liquidity_sweep_up"] = ls_up
    df["liquidity_sweep_down"] = ls_dn

    vol = df["candle_acc_trade_volume"].fillna(0)
    px = df["trade_price"]
    bull_ob = (px > px.shift()) & (vol > vol.shift()*1.5)
    bear_ob = (px < px.shift()) & (vol > vol.shift()*1.5)
    df["bullish_ob"] = bull_ob.fillna(False).astype(bool)
    df["bearish_ob"] = bear_ob.fillna(False).astype(bool)

    df.ffill(inplace=True)
    df.bfill(inplace=True)
    for col in ["supertrend","fvg_bull","fvg_bear","in_bull_fvg","in_bear_fvg","swing_high","swing_low",
                "bos_up","bos_dn","discount_zone","premium_zone","in_ote_long",
                "liquidity_sweep_up","liquidity_sweep_down","bullish_ob","bearish_ob"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


# ====== ICHIMOKU UTILS ======
def add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """일목구름 계산 유틸 (클라우드 및 돌파/이탈 플래그 생성)."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

    high = df["high_price"]
    low = df["low_price"]
    close = df["trade_price"]

    conversion = (high.rolling(9).max() + low.rolling(9).min()) / 2.0  # 전환선(9)
    base = (high.rolling(26).max() + low.rolling(26).min()) / 2.0      # 기준선(26)
    span_a = ((conversion + base) / 2.0).shift(26)                     # 선행스팬 A
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2.0).shift(26)  # 선행스팬 B

    df["ich_span_a"] = span_a
    df["ich_span_b"] = span_b

    cloud_high = np.maximum(span_a, span_b)
    cloud_low = np.minimum(span_a, span_b)

    above = (close > cloud_high) & span_a.notna() & span_b.notna()
    below = (close < cloud_low) & span_a.notna() & span_b.notna()
    inside = (~above & ~below) & span_a.notna() & span_b.notna()

    df["ich_above_cloud"] = above
    df["ich_below_cloud"] = below
    df["ich_in_cloud"] = inside

    prev_above = above.shift(1).fillna(False)
    df["ich_breakout_up"] = above & (~prev_above)

    now_not_above = ~above & (inside | below)
    df["ich_breakdown_from_above"] = now_not_above & prev_above

    for col in ["ich_above_cloud", "ich_below_cloud", "ich_in_cloud",
                "ich_breakout_up", "ich_breakdown_from_above"]:
        df[col] = df[col].astype(bool)

    return df


def get_weekly_ichimoku_state(market: str, max_age_sec: int = 1800) -> Dict[str, bool]:
    """
    주봉 일목구름 상태 요약 + 유효성 플래그:
      - valid: 주봉 캔들이 충분하고 일목구름(Span A/B)이 형성되어 있는지 여부
      - trend_ok: 종가가 클라우드 위에 위치 (롱 사이드만 허용)
      - breakout_up: 직전 구간 대비 클라우드 상방 돌파
      - breakdown: 클라우드 위에서 그려지다가 클라우드 안/아래로 재진입

    상장한지 얼마 안되었거나 클라우드가 형성되지 않은 경우 valid=False 로 반환하고,
    이 경우 해당 종목은 매수/매도 모두 진행하지 않는다.
    """
    default_state = {
        "valid": False,
        "trend_ok": False,
        "breakout_up": False,
        "breakdown": False,
    }

    now = time.time()
    cached = weekly_ich_cache.get(market)
    if cached is not None:
        ts, state = cached
        if now - ts < max_age_sec:
            return state

    dfw = get_ohlcv(market, interval="week", count=120)
    if dfw is None or dfw.empty:
        weekly_ich_cache[market] = (now, default_state)
        return default_state

    dfw = add_ichimoku(dfw)
    if dfw is None or dfw.empty or len(dfw) < 3:
        weekly_ich_cache[market] = (now, default_state)
        return default_state

    last = dfw.iloc[-1]
    prev = dfw.iloc[-2]

    span_a = last.get("ich_span_a")
    span_b = last.get("ich_span_b")

    # 일목구름이 아직 형성되지 않은 경우 (Span A/B 가 NaN 이면 유효하지 않은 상태로 본다)
    if pd.isna(span_a) or pd.isna(span_b):
        weekly_ich_cache[market] = (now, default_state)
        return default_state

    trend_ok = bool(last.get("ich_above_cloud", False))
    breakout_up = bool(last.get("ich_breakout_up", False)) or (
        trend_ok and not bool(prev.get("ich_above_cloud", False))
    )
    breakdown = bool(last.get("ich_breakdown_from_above", False))

    state = {
        "valid": True,
        "trend_ok": trend_ok,
        "breakout_up": breakout_up,
        "breakdown": breakdown,
    }
    weekly_ich_cache[market] = (now, state)
    return state

# ================== 스캐너/레짐 ==================
def pick_universe_by_krw_volume() -> List[str]:
    """
    KRW 마켓 전체를 대상으로:
      - 초저가 코인 배제(MIN_PRICE_KRW)
      - 일단 일봉 기준으로 거래대금(가격*거래량) 정렬
      - 스프레드 필터(SPREAD_MAX_PCT) 적용
    """
    markets = get_markets_krw()
    scores = []
    for m in markets:
        df_day = get_ohlcv(m, interval="day", count=2)
        if df_day.empty:
            continue
        last_price = float(safe_val(df_day, "trade_price", -1, 0))
        last_vol = float(safe_val(df_day, "candle_acc_trade_volume", -1, 0))
        if last_price < MIN_PRICE_KRW:
            continue
        v_krw = last_vol * last_price
        scores.append((m, v_krw))
    scores.sort(key=lambda x: x[1], reverse=True)
    # 볼륨 기준으로 정렬된 전체 KRW 마켓을 대상으로, 스프레드 필터만 적용
    top = [m for m, _ in scores]
    filtered = []
    for m in top:
        sp = get_orderbook_spread_pct(m)
        if sp <= SPREAD_MAX_PCT:
            filtered.append(m)
    return filtered

def btc_regime_factor() -> float:
    """BTC 15분봉 상향(EMA20>EMA50 or Supertrend True)이면 1.0, 아니면 0.5."""
    try:
        df = add_indicators(get_ohlcv("KRW-BTC", "minute15", 200))
        if df.empty:
            return 1.0
        ema_ok = bool(safe_val(df, "ema20", -1, np.nan) > safe_val(df, "ema50", -1, np.nan))
        st_ok = bool(safe_val(df, "supertrend", -1, False))
        return 1.0 if (ema_ok or st_ok) else 0.5
    except Exception:
        return 1.0

# ================== 초기화/리포트 ==================
def initialize_trading_data():
    global buy_prices, highest_prices, position_meta, day_start_equity, day_of_snapshot
    buy_prices.clear()
    highest_prices.clear()
    position_meta.clear()
    balances = get_balances_safe()
    valid = set(get_markets_krw())
    for b in balances:
        cur = b.get("currency")
        if cur == "KRW":
            continue
        m = f"KRW-{cur}"
        if m not in valid:
            continue
        avg = float(b.get("avg_buy_price", 0) or 0)
        if avg > 0:
            buy_prices[m] = avg
            cp = get_current_price_safe(m)
            if not np.isnan(cp):
                highest_prices[m] = cp
    eq = portfolio_value_krw()
    day_start_equity = eq
    day_of_snapshot = time.strftime("%Y-%m-%d")
    send_slack_message(f"[Start] 초기화 완료 / 시작 에쿼티: {eq:,.0f} KRW")

def generate_daily_report():
    try:
        bals = get_balances_safe()
        valid = set(get_markets_krw())
        lines = ["[Daily Summary]"]
        for b in bals:
            cur = b.get("currency")
            if cur == "KRW":
                continue
            m = f"KRW-{cur}"
            if m not in valid:
                continue
            bal = float(b.get("balance", 0) or 0)
            avg = float(b.get("avg_buy_price", 0) or 0)
            if bal == 0 or avg == 0:
                continue
            cp = get_current_price_safe(m)
            if np.isnan(cp):
                continue
            pnl = (cp - avg) * bal
            rate = (cp - avg) / avg * 100
            lines.append(f"{m}: {pnl:.0f} KRW ({rate:.2f}%)")
        if len(lines) == 1:
            lines.append("보유 자산 없음")
        send_slack_message("\n".join(lines))
    except Exception as e:
        logger.warning(f"daily report failed: {e}")

def start_report_thread():
    def loop():
        while not stop_trading.is_set():
            generate_daily_report()
            time.sleep(86400)
    threading.Thread(target=loop, daemon=True).start()

# ================== 포트폴리오/보유 ==================
def portfolio_value_krw() -> float:
    bals = get_balances_safe()
    total = 0.0
    for b in bals:
        cur = b.get("currency")
        bal = float(b.get("balance", 0) or 0)
        if cur == "KRW":
            total += bal
        else:
            m = f"KRW-{cur}"
            cp = get_current_price_safe(m)
            if np.isnan(cp):
                continue
            total += bal * cp
    return total

def owned_markets() -> Dict[str, float]:
    out = {}
    for b in get_balances_safe():
        cur = b.get("currency")
        if cur == "KRW":
            continue
        bal = float(b.get("balance", 0) or 0)
        m = f"KRW-{cur}"
        cp = get_current_price_safe(m)
        if not np.isnan(cp) and bal * cp >= MINIMUM_EVALUATION_KRW:
            out[m] = bal
    return out

# ================== 호가/체결 보정 ==================
def upbit_tick_size(krw_price: float) -> float:
    if krw_price >= 2_000_000: return 1000
    if krw_price >= 1_000_000: return 500
    if krw_price >= 500_000:   return 100
    if krw_price >= 100_000:   return 50
    if krw_price >= 10_000:    return 10
    if krw_price >= 1_000:     return 1
    if krw_price >= 100:       return 0.1
    if krw_price >= 10:        return 0.01
    if krw_price >= 1:         return 0.001
    return 0.0001

def round_to_tick(price: float) -> float:
    if price <= 0 or np.isnan(price): return price
    tick = upbit_tick_size(price)
    return math.floor(price / tick) * tick

def round_volume(v: float) -> float:
    if v <= 0 or np.isnan(v): return 0.0
    return float(f"{v:.8f}")

# ================== 가용/부족자금 ==================
def get_available_krw() -> float:
    try:
        for b in get_balances_safe():
            if b.get("currency") == "KRW":
                bal = float(b.get("balance", 0) or 0)
                locked = float(b.get("locked", 0) or 0)
                return max(0.0, bal - locked)
    except Exception:
        pass
    return 0.0

def get_available_volume(market: str) -> float:
    cur = market.replace("KRW-", "")
    try:
        for b in get_balances_safe():
            if b.get("currency") == cur:
                bal = float(b.get("balance", 0) or 0)
                locked = float(b.get("locked", 0) or 0)
                return max(0.0, bal - locked)
    except Exception:
        pass
    return 0.0

def _insufficient_funds(e_or_res) -> bool:
    s = ""
    try:
        s = str(e_or_res).lower()
    except Exception:
        pass
    return ("insufficient" in s) and ("fund" in s or "bid" in s)

# ================== 실제 체결가(VWAP) ==================
def get_order_fills(uuid: str) -> Tuple[float, float]:
    try:
        rate_limit()
        od = upbit.get_order(uuid)
        trades = od.get("trades", []) if isinstance(od, dict) else []
        if not trades:
            return np.nan, 0.0
        total_cost = 0.0
        total_vol = 0.0
        for t in trades:
            price = float(t.get("price", 0) or 0)
            vol = float(t.get("volume", 0) or 0)
            total_cost += price * vol
            total_vol += vol
        if total_vol <= 0:
            return np.nan, 0.0
        vwap = total_cost / total_vol
        return float(vwap), float(total_vol)
    except Exception as e:
        logger.warning(f"get_order_fills failed: {e}")
        return np.nan, 0.0

# ================== 주문 엔진(지정가→시장가) ==================
def place_limit_then_maybe_market(side: str, market: str, krw_amount: Optional[float]=None,
                                  limit_price: Optional[float]=None, volume: Optional[float]=None,
                                  timeout_sec: int=LIMIT_TIMEOUT_SEC, fast_move_pct: float=FAST_MOVE_PCT,
                                  max_slip_pct: float=MAX_SLIPPAGE_PCT):
    assert side in ("bid", "ask")
    cur0 = get_current_price_safe(market)
    if np.isnan(cur0):
        return None

    if limit_price is None:
        if side == "bid":
            limit_price = cur0 * (1 + LIMIT_OFFSET_BUY)
        else:
            limit_price = cur0 * (1 + LIMIT_OFFSET_SELL)
    limit_price = round_to_tick(limit_price)

    if side == "bid":
        avail = get_available_krw()
        if krw_amount is None:
            krw_amount = avail
        budget = min(krw_amount, avail)
        headroom = (1.0 - (FEE_RATE + SAFETY_BUY_BUFFER))
        headroom = max(headroom, 0.95)
        eff_budget = max(0.0, budget * headroom)
        if eff_budget < MINIMUM_ORDER_KRW:
            return None
        vol = round_volume(eff_budget / limit_price)
        if vol * limit_price < MINIMUM_ORDER_KRW:
            return None

        attempt = 0
        while attempt <= RETRY_ON_INSUFFICIENT:
            placed = None
            try:
                rate_limit()
                res = upbit.buy_limit_order(market, limit_price, vol)
                if res and "error" in res and _insufficient_funds(res["error"]):
                    raise RuntimeError("InsufficientFundsBid (payload)")
                placed = res if (res and "uuid" in res) else None
            except Exception as e:
                if _insufficient_funds(e):
                    placed = None
                else:
                    logger.warning(f"buy_limit_order error: {e}")
                    placed = None

            if placed:
                uuid = placed["uuid"]
                t0 = time.time()
                base = cur0
                while time.time() - t0 < timeout_sec and not stop_trading.is_set():
                    try:
                        rate_limit()
                        od = upbit.get_order(uuid)
                        remain_vol = float(od.get("remaining_volume", 0) or 0)
                        state = od.get("state", "")
                        cur2 = get_current_price_safe(market)
                        if not np.isnan(cur2) and base > 0:
                            move = abs((cur2 - base) / base) * 100
                            if move >= fast_move_pct:
                                try:
                                    upbit.cancel_order(uuid)
                                except Exception:
                                    pass
                                slip = ((cur2 - limit_price) / limit_price) * 100
                                if slip > max_slip_pct:
                                    send_slack_message(f"[경고] {market} 매수 슬리피지 {slip:.2f}% > {max_slip_pct:.2f}% → 주문 취소")
                                    return None
                                avail2 = get_available_krw()
                                mrk_budget = min(avail2, remain_vol * limit_price) * headroom
                                mrk_budget = math.floor(mrk_budget)
                                if mrk_budget >= MINIMUM_ORDER_KRW:
                                    return upbit.buy_market_order(market, mrk_budget)
                                return od
                        if state == "done" or remain_vol <= 1e-12:
                            return od
                    except Exception as e:
                        logger.warning(f"monitor order {uuid} failed: {e}")
                    time.sleep(2)

                try:
                    upbit.cancel_order(uuid)
                except Exception:
                    pass
                cur3 = get_current_price_safe(market)
                slip = ((cur3 - limit_price) / limit_price) * 100 if (not np.isnan(cur3) and limit_price > 0) else 0
                if slip > max_slip_pct:
                    send_slack_message(f"[경고] {market} 매수 슬리피지 {slip:.2f}% > {max_slip_pct:.2f}% → 주문 취소")
                    return None
                try:
                    rate_limit()
                    od = upbit.get_order(uuid)
                    remain_vol = float(od.get("remaining_volume", 0) or 0)
                except Exception:
                    remain_vol = 0.0
                avail3 = get_available_krw()
                mrk_budget = min(avail3, remain_vol * limit_price) * headroom
                mrk_budget = math.floor(mrk_budget)
                if mrk_budget >= MINIMUM_ORDER_KRW:
                    return upbit.buy_market_order(market, mrk_budget)
                return None

            attempt += 1
            eff_budget *= 0.97
            vol = round_volume(eff_budget / limit_price)
            if eff_budget < MINIMUM_ORDER_KRW or vol * limit_price < MINIMUM_ORDER_KRW:
                time.sleep(RETRY_BACKOFF_SEC)
                continue
            time.sleep(RETRY_BACKOFF_SEC)
        return None

    else:
        if volume is None or volume <= 0:
            return None
        avail_vol = get_available_volume(market)
        vol = min(volume, avail_vol)
        vol = round_volume(vol)
        if vol <= 0:
            return None
        if limit_price * vol < MINIMUM_ORDER_KRW:
            return None
        try:
            rate_limit()
            res = upbit.sell_limit_order(market, limit_price, vol)
        except Exception as e:
            logger.warning(f"sell_limit_order failed: {e}")
            return None
        if not res or "uuid" not in res:
            return None

        uuid = res["uuid"]
        t0 = time.time()
        base = cur0
        while time.time() - t0 < timeout_sec and not stop_trading.is_set():
            try:
                rate_limit()
                od = upbit.get_order(uuid)
                remain_vol = float(od.get("remaining_volume", 0) or 0)
                state = od.get("state", "")
                cur2 = get_current_price_safe(market)
                if not np.isnan(cur2) and base > 0:
                    move = abs((cur2 - base) / base) * 100
                    if move >= fast_move_pct:
                        try:
                            upbit.cancel_order(uuid)
                        except Exception:
                            pass
                        slip = ((limit_price - cur2) / limit_price) * 100
                        if slip > max_slip_pct:
                            send_slack_message(f"[경고] {market} 매도 슬리피지 {slip:.2f}% > {max_slip_pct:.2f}% → 주문 취소")
                            return None
                        return upbit.sell_market_order(market, remain_vol)
                if state == "done" or remain_vol <= 1e-12:
                    return od
            except Exception as e:
                logger.warning(f"monitor order {uuid} failed: {e}")
            time.sleep(2)

        try:
            upbit.cancel_order(uuid)
        except Exception:
            pass
        cur3 = get_current_price_safe(market)
        slip = ((limit_price - cur3) / limit_price) * 100 if (not np.isnan(cur3) and limit_price > 0) else 0
        if slip > max_slip_pct:
            send_slack_message(f"[경고] {market} 매도 슬리피지 {slip:.2f}% > {max_slip_pct:.2f}% → 주문 취소")
            return None
        return upbit.sell_market_order(market, vol)

# ================== 트레이딩 로직 ==================
def _risk_budget_krw(entry_price: float, atr_val: float) -> float:
    equity = portfolio_value_krw()
    risk_budget = equity * RISK_PER_TRADE
    stop_pct = abs(FIXED_STOP_LOSS) / 100.0
    dist = max(entry_price * stop_pct, atr_val * ATR_SL_MULT)
    if dist <= 0:
        return MIN_BUY_KRW
    amount = risk_budget * (entry_price / dist)
    return max(MIN_BUY_KRW, min(amount, MAX_BUY_KRW))

def _update_daily_dd_and_maybe_stop():
    global day_start_equity, day_of_snapshot
    today = time.strftime("%Y-%m-%d")
    if day_of_snapshot != today:
        day_of_snapshot = today
        day_start_equity = portfolio_value_krw()
        send_slack_message(f"[Daily Reset] 기준 에쿼티: {day_start_equity:,.0f} KRW")
        return
    eq = portfolio_value_krw()
    if day_start_equity > 0:
        dd = (eq - day_start_equity) / day_start_equity * 100.0
        if dd <= -DAILY_MAX_DRAWDOWN_PCT and not stop_trading.is_set():
            send_slack_message(f"[중지] 일간 드로우다운 {dd:.2f}% ≤ -{DAILY_MAX_DRAWDOWN_PCT:.2f}%")
            stop_trading.set()

def track_buy_signals(universe: List[str]):
    global last_buy_time, buy_prices, highest_prices, position_meta
    owned = owned_markets()
    if len(owned) >= MAX_CONCURRENT_TRADES:
        return
    krw = get_available_krw()
    if krw < MINIMUM_ORDER_KRW:
        return
    pv = portfolio_value_krw()
    regime = btc_regime_factor()
    base_target = pv * PORTFOLIO_BUY_RATIO

    for m in universe:
        if m in owned and owned[m] * get_current_price_safe(m) > pv * 0.25:
            continue
        if m in last_buy_time and time.time() - last_buy_time[m] < COOLDOWN_PERIOD_BUY:
            continue
        if m in last_loss_time and time.time() - last_loss_time[m] < COOLDOWN_AFTER_LOSS_SEC:
            continue

        sp = get_orderbook_spread_pct(m)
        if sp > SPREAD_MAX_PCT:
            continue

        # 1순위 필터: 주봉 일목구름
        #  - valid=False (상장 초기/클라우드 미형성) 인 종목은 매수 자체를 진행하지 않음
        #  - trend_ok=True 인, 즉 주봉 종가가 클라우드 위에 있는 종목만 롱 진입
        ich_state = get_weekly_ichimoku_state(m)
        if not ich_state.get("valid", False):
            continue
        if not ich_state["trend_ok"]:
            continue

        # 거시 타임프레임 기반 보조 지표 (4시간/1시간)
        df4h = add_indicators(get_ohlcv(m, "minute240", 200))
        df1h = add_indicators(get_ohlcv(m, "minute60", 200))
        if df4h.empty or df1h.empty or len(df4h) < 3 or len(df1h) < 3:
            continue

        # 1시간봉 기준 단기 모멘텀 확인
        cur = float(safe_val(df1h, "trade_price", -1, np.nan))
        prev = float(safe_val(df1h, "trade_price", -2, np.nan))
        if np.isnan(cur) or np.isnan(prev) or prev <= 0:
            continue
        chg1 = (cur - prev) / prev * 100

        # 4시간/1시간 RSI, MACD, 추세 확인
        rsi_4h = float(safe_val(df4h, "rsi", -1, np.nan))
        rsi_1h = float(safe_val(df1h, "rsi", -1, np.nan))

        macd_now_4h = float(safe_val(df4h, "macd", -1, np.nan))
        macd_sig_now_4h = float(safe_val(df4h, "macd_signal", -1, np.nan))
        macd_prev_4h = float(safe_val(df4h, "macd", -2, np.nan))
        sig_prev_4h = float(safe_val(df4h, "macd_signal", -2, np.nan))
        golden_4h = (macd_now_4h > macd_sig_now_4h) and (macd_prev_4h <= sig_prev_4h)

        ema20_4h = float(safe_val(df4h, "ema20", -1, np.nan))
        ema50_4h = float(safe_val(df4h, "ema50", -1, np.nan))
        ema_trend_4h = (ema20_4h > ema50_4h)

        st4h = bool(safe_val(df4h, "supertrend", -1, False))

        in_bull_fvg_4h = bool(safe_val(df4h, "in_bull_fvg", -1, False))
        ls_down_4h = bool(safe_val(df4h, "liquidity_sweep_down", -1, False))
        discount_zone_4h = bool(safe_val(df4h, "discount_zone", -1, False))
        in_ote_4h = bool(safe_val(df4h, "in_ote_long", -1, False))

        # 과한 단기 급등 & 과열 구간 필터 (거시 기준으로 완화)
        if (not np.isnan(rsi_1h) and rsi_1h > RSI_MAX_ENTRY) or (not np.isnan(rsi_4h) and rsi_4h > RSI_MAX_ENTRY - 5):
            continue
        if chg1 > (CHASE_UP_PCT_BLOCK * 0.7):
            continue

        # 보조 지표 기반 매수 로직 (거시 타임프레임 점수제)
        score = 0
        if golden_4h or ema_trend_4h:
            score += 1
        if st4h:
            score += 1
        if in_bull_fvg_4h or ls_down_4h or discount_zone_4h or in_ote_4h:
            score += 1
        if not np.isnan(rsi_4h) and 35 <= rsi_4h <= (RSI_MAX_ENTRY - 10):
            score += 1

        buy_ok = score >= 2
        if not buy_ok:
            continue

        # 리스크 계산은 4시간 ATR 기준
        atr_val = float(safe_val(df4h, "atr", -1, np.nan))
        if np.isnan(atr_val) or atr_val <= 0:
            atr_val = cur * 0.02  # fallback 2%

        risk_amount = _risk_budget_krw(cur, atr_val)
        target_amt = max(MIN_BUY_KRW, min(MAX_BUY_KRW, min(base_target, risk_amount) * regime))
        target_amt = min(target_amt, krw)
        if target_amt < MINIMUM_ORDER_KRW:
            continue

        limit_px = round_to_tick(cur * (1 + LIMIT_OFFSET_BUY))
        res = place_limit_then_maybe_market("bid", m, krw_amount=target_amt, limit_price=limit_px)
        if not res or "uuid" not in res:
            continue

        vwap, filled = get_order_fills(res["uuid"])
        if filled <= 0 or np.isnan(vwap):
            _, new_avg = get_balance(m)
            entry = new_avg if new_avg > 0 else cur
        else:
            entry = float(vwap)

        last_buy_time[m] = time.time()
        buy_prices[m] = entry
        highest_prices[m] = cur

        stop_pct = abs(FIXED_STOP_LOSS) / 100.0
        init_dist = max(entry * stop_pct, atr_val * ATR_SL_MULT)
        position_meta[m] = {
            "entry": entry,
            "init_dist": init_dist,
            "tp1": 0.0,
            "tp2": 0.0,
            "tp1_done": 0.0,
            "tp2_done": 0.0,
            "entered_at": time.time(),
            "stop": max(0.0, entry - init_dist),
        }

        slip_report = ""
        if limit_px > 0:
            slip = ((entry - limit_px) / limit_px) * 100
            slip_report = f" / 슬리피지:{slip:.2f}%"

        ichi_info = f"IchiTrend:{'UP' if ich_state['trend_ok'] else 'N/A'} BreakOut:{'Y' if ich_state.get('breakout_up') else 'N'}"
        ict_info = f"4h_FVG:{in_bull_fvg_4h} OTE:{in_ote_4h} Dsc:{discount_zone_4h}"
        rsi_4h_str = f"{rsi_4h:.2f}" if not np.isnan(rsi_4h) else "nan"
        send_slack_message(
            f"[매수] {m} / 금액: {target_amt:.0f} KRW / 체결가(VWAP): {entry:.2f}{slip_report} "
            f"/ {ict_info} / 4h_RSI:{rsi_4h_str} / 리스크:{init_dist:.4f} / {ichi_info}"
        )

def should_trail_stop(high: float, cur: float, pct: float) -> bool:
    if high <= 0 or np.isnan(high) or np.isnan(cur):
        return False
    drop = (high - cur) / high * 100
    return drop >= pct

def _maybe_be_move(market: str, cur: float):
    meta = position_meta.get(market)
    if not meta:
        return
    if meta.get("tp1_done", 0.0) and cur > 0:
        be_price = meta["entry"] * (1.0 + BREAK_EVEN_BUFFER_PCT/100.0 + FEE_RATE*2)
        meta["stop"] = max(meta.get("stop", 0.0), be_price)

def _r_multiples(meta: Dict[str, float], cur: float) -> float:
    if not meta:
        return 0.0
    R = meta["init_dist"]
    if R <= 0:
        return 0.0
    return (cur - meta["entry"]) / R

def track_sell_signals():
    global total_profit, highest_prices, position_meta, last_loss_time
    owned = owned_markets()
    if not owned:
        return

    for m, vol in owned.items():
        df4h = add_indicators(get_ohlcv(m, "minute240", 200))
        df1h = add_indicators(get_ohlcv(m, "minute60", 200))
        if df4h.empty or len(df4h) < 3:
            continue

        # 주봉 일목구름 상태 (1순위 방향성 필터)
        #  - valid=False (상장 초기/클라우드 미형성) 인 종목은 매수/매도 모두 건드리지 않음
        ich_state = get_weekly_ichimoku_state(m)
        if not ich_state.get("valid", False):
            continue
        ich_breakdown = ich_state.get("breakdown", False)

        # 가격은 가급적 더 민감한 1시간 봉에서 확인, 없으면 4시간 기준
        cur = float(safe_val(df1h, "trade_price", -1, np.nan)) if not df1h.empty else float(safe_val(df4h, "trade_price", -1, np.nan))
        if np.isnan(cur):
            continue

        avg = float(buy_prices.get(m, 0) or 0)
        if avg <= 0:
            _, avg2 = get_balance(m)
            avg = avg2 if avg2 > 0 else cur

        if m not in highest_prices or np.isnan(highest_prices[m]):
            highest_prices[m] = cur
        else:
            highest_prices[m] = max(highest_prices[m], cur)

        pnl_pct = (cur - avg) / avg * 100 if avg > 0 else 0.0

        atr = float(safe_val(df4h, "atr", -1, np.nan))
        st_now_4h = bool(safe_val(df4h, "supertrend", -1, False))
        st_prev_4h = bool(safe_val(df4h, "supertrend", -2, False))
        macd_4h = float(safe_val(df4h, "macd", -1, np.nan))
        macd_sig_4h = float(safe_val(df4h, "macd_signal", -1, np.nan))
        macd_prev_4h = float(safe_val(df4h, "macd", -2, np.nan))
        sig_prev_4h = float(safe_val(df4h, "macd_signal", -2, np.nan))
        death_4h = (macd_4h < macd_sig_4h) and (macd_prev_4h >= sig_prev_4h)

        in_bear_fvg_4h = bool(safe_val(df4h, "in_bear_fvg", -1, False))
        ls_up_4h = bool(safe_val(df4h, "liquidity_sweep_up", -1, False))
        premium_zone_4h = bool(safe_val(df4h, "premium_zone", -1, False))
        ict_bearish_4h = (in_bear_fvg_4h and premium_zone_4h) or ls_up_4h

        meta = position_meta.get(m, None)
        if meta:
            _maybe_be_move(m, cur)
            stop_line = meta.get("stop", 0.0)
            if stop_line > 0 and cur <= stop_line:
                qty = vol
                if qty * cur >= MINIMUM_ORDER_KRW:
                    limit_px = round_to_tick(cur * (1 + LIMIT_OFFSET_SELL))
                    res = place_limit_then_maybe_market("ask", m, limit_price=limit_px, volume=qty)
                    if res and "uuid" in res:
                        vwap, filled = get_order_fills(res["uuid"])
                        exec_qty = filled if filled > 0 else qty
                        exec_price = float(vwap) if (filled > 0 and not np.isnan(vwap)) else cur
                        profit = (exec_price - avg) * exec_qty
                        total_profit += profit
                        if profit < 0:
                            last_loss_time[m] = time.time()
                        send_slack_message(f"[스톱] {m} / {exec_qty:.6f}@{exec_price:.2f} / PnL:{profit:.0f} / 누적:{total_profit:.0f}")
                        position_meta.pop(m, None)
                        continue

        sell = False
        ratio = 1.0
        reason = ""

        # 1차 방어: ATR/고정 손절
        if not np.isnan(atr) and (avg - cur) > atr * ATR_SL_MULT:
            sell, ratio, reason = True, 1.0, f"4h ATR 손절({pnl_pct:.2f}%)"
        elif pnl_pct <= FIXED_STOP_LOSS:
            sell, ratio, reason = True, 1.0, f"고정 손절({pnl_pct:.2f}%)"

        # 2차: 주봉 일목구름 하락 시그널 → 전량 청산
        elif ich_breakdown:
            sell, ratio, reason = True, 1.0, "주봉 일목구름 하락(클라우드 재진입/하회)"

        else:
            if meta:
                r_mult = _r_multiples(meta, cur)
                if (not meta.get("tp1_done")) and r_mult >= TP1_R:
                    sell, ratio, reason = True, 0.5, f"1R 부분익절({pnl_pct:.2f}%)"
                    meta["tp1_done"] = 1.0
                    _maybe_be_move(m, cur)
                elif (not meta.get("tp2_done")) and r_mult >= TP2_R:
                    sell, ratio, reason = True, 0.5, f"2R 추가익절({pnl_pct:.2f}%)"
                    meta["tp2_done"] = 1.0

        if not sell:
            strong_combo = ict_bearish_4h and (death_4h or (st_prev_4h and not st_now_4h))
            if strong_combo:
                sell, ratio, reason = True, 1.0, "강한 복합 매도 시그널(ICT+추세, 4h)"
            elif st_prev_4h and not st_now_4h:
                sell, ratio, reason = True, 0.7, "4h Supertrend 반전"
            elif death_4h:
                sell, ratio, reason = True, 0.5, "4h MACD 데드크로스"
            elif ict_bearish_4h:
                sell, ratio, reason = True, 1.0, "4h ICT 베어리시"
            elif should_trail_stop(highest_prices[m], cur, TRAILING_STOP_PCT):
                sell, ratio, reason = True, 1.0, "트레일링 스탑"

        if not sell:
            continue

        qty = vol * ratio
        if qty * cur < MINIMUM_ORDER_KRW:
            continue

        limit_px = round_to_tick(cur * (1 + LIMIT_OFFSET_SELL))
        res = place_limit_then_maybe_market("ask", m, limit_price=limit_px, volume=qty)
        if not res or "uuid" not in res:
            continue

        vwap, filled = get_order_fills(res["uuid"])
        exec_qty = filled if filled > 0 else qty
        exec_price = float(vwap) if (filled > 0 and not np.isnan(vwap)) else cur

        profit = (exec_price - avg) * exec_qty
        total_profit += profit
        if profit < 0:
            last_loss_time[m] = time.time()

        slip_report = ""
        if limit_px > 0:
            slip = ((limit_px - exec_price) / limit_px) * 100
            slip_report = f" / 슬리피지:{slip:.2f}%"
        send_slack_message(
            f"[매도] {m} / 수량:{exec_qty:.6f} / 체결가(VWAP):{exec_price:.2f} "
            f"/ PnL:{profit:.0f} KRW / 누적:{total_profit:.0f} KRW / 사유:{reason}{slip_report}"
        )

        bal_after, _ = get_balance(m)
        if bal_after * exec_price < MINIMUM_EVALUATION_KRW:
            position_meta.pop(m, None)

# ================== Slack 간이 명령 ==================
def handle_slack_commands():
    while not stop_trading.is_set():
        try:
            if os.path.exists("stop_command.txt"):
                with open("stop_command.txt", "r") as f:
                    if "stop trading" in f.read().lower():
                        send_slack_message("[명령] 중지 요청 수신")
                        os.kill(os.getpid(), signal.SIGTERM)
        except Exception:
            pass
        time.sleep(5)

# ================== 메인 ==================
def main():
    global upbit
    if not ACCESS_KEY or not SECRET_KEY:
        logger.error("API 키가 설정되지 않았습니다.")
        print("API 키가 설정되지 않았습니다.")
        return

    upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
    signal.signal(signal.SIGINT, handle_stop_signal)
    signal.signal(signal.SIGTERM, handle_stop_signal)

    send_slack_message("[Start] Upbit 자동매매 시작")
    initialize_trading_data()
    start_report_thread()

    threading.Thread(target=handle_slack_commands, daemon=True).start()

    last_universe_refresh = 0
    universe: List[str] = []

    while not stop_trading.is_set():
        try:
            _update_daily_dd_and_maybe_stop()
            if stop_trading.is_set():
                break

            if time.time() - last_universe_refresh > 600 or not universe:
                universe = pick_universe_by_krw_volume()
                last_universe_refresh = time.time()
                logger.info(f"Universe refreshed: {universe[:10]} ...")

            track_buy_signals(universe)
            track_sell_signals()

        except Exception as e:
            logger.error(f"main loop error: {e}", exc_info=True)
            send_slack_message(f"[오류] 메인 루프 예외: {e}")

        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()