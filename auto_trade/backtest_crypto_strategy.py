import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyupbit

# ===== Helper Functions =====
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
    high = df["high"]
    low = df["low"]
    close = df["close"]
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
    hl2 = (df["high"] + df["low"]) / 2
    df["supertrend_upper"] = hl2 + (multiplier * atr)
    df["supertrend_lower"] = hl2 - (multiplier * atr)
    df["supertrend"] = np.where(
        df["close"] > df["supertrend_upper"].shift(),
        df["supertrend_upper"],
        np.where(
            df["close"] < df["supertrend_lower"].shift(),
            df["supertrend_lower"],
            np.nan
        )
    )
    df["supertrend"] = df["supertrend"].ffill()
    return df

def calculate_atr(df, window):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def calculate_volume_momentum(df):
    volume = df["volume"]
    return volume.pct_change().rolling(window=5).mean()

def calculate_indicators(df):
    """데이터프레임에 필요한 지표를 추가"""
    df["rsi"] = calculate_rsi(df["close"])
    df["macd"], df["macd_signal"] = calculate_macd(df["close"])
    df["adx"] = calculate_adx(df)
    df = calculate_supertrend(df)
    df["volume_momentum"] = calculate_volume_momentum(df)
    return df

def fetch_historical_data(ticker, interval="minute240", count=2000):
    """PyUpbit를 사용하여 과거 데이터를 가져옵니다."""
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
    if df is None or df.empty:
        raise ValueError("Failed to fetch data. Please check the ticker or interval.")
    df.reset_index(inplace=True)
    df.rename(columns={"index": "timestamp"}, inplace=True)
    return df

# ===== Backtest Function =====
def backtest(data, initial_balance=1_000_000, buy_ratio=0.3):
    balance = initial_balance
    holdings = 0
    total_value = []
    trades = []

    for i in range(len(data)):
        row = data.iloc[i]

        # 매수 조건
        if (
            row["rsi"] < 25 and  # RSI가 더 낮은 수준에서만 매수
            row["macd"] > row["macd_signal"] and  # MACD 상향 교차
            row["adx"] > 25 and  # ADX 강한 추세 확인
            row["volume_momentum"] > 0.02 and  # 거래량 모멘텀이 더 강한 경우
            row["supertrend"] < row["close"] and  # Supertrend가 현재 가격보다 낮음
            (row["close"] - row["low"]) / row["low"] < 0.01  # 저점 대비 매우 적은 상승폭
        ):
            if balance > 0:
                buy_amount = balance * buy_ratio
                holdings += buy_amount / row["close"]
                balance -= buy_amount
                trades.append((row["timestamp"], "BUY", row["close"], buy_amount))

        # 매도 조건
        profit_ratio = ((row["close"] * holdings - balance) / balance) * 100 if holdings > 0 else 0

        # 조건 1: 목표 수익률 도달 시 매도
        if profit_ratio >= 10.0:
            sell_amount = holdings * row["close"]
            balance += sell_amount
            holdings = 0
            trades.append((row["timestamp"], "SELL", row["close"], sell_amount))
            continue

        # 조건 2: Trailing Stop 적용
        if profit_ratio > 2.0 and row["supertrend"] > row["close"]:
            sell_amount = holdings * row["close"]
            balance += sell_amount
            holdings = 0
            trades.append((row["timestamp"], "SELL", row["close"], sell_amount))
            continue

        # 조건 3: 손실 한계 도달 시 손절
        if profit_ratio <= -2.0:
            sell_amount = holdings * row["close"]
            balance += sell_amount
            holdings = 0
            trades.append((row["timestamp"], "SELL", row["close"], sell_amount))

        total_value.append(balance + holdings * row["close"])

    # 결과 반환
    result = {
        "final_balance": balance,
        "final_holdings_value": holdings * data.iloc[-1]["close"],
        "total_value": total_value,
        "trades": pd.DataFrame(trades, columns=["Timestamp", "Action", "Price", "Amount"])
    }
    return result

# ===== Main Execution =====
if __name__ == "__main__":
    try:
        # 과거 데이터 가져오기
        ticker = "KRW-ETH"  # 예: 비트코인
        print("Fetching historical data...")
        data = fetch_historical_data(ticker, interval="minute240", count=2000)

        # 데이터 처리
        print("Calculating Indicators...")
        data = calculate_indicators(data)
        data.dropna(inplace=True)

        # 백테스트 실행
        print("Running Backtest...")
        result = backtest(data)

        # 결과 출력
        print(f"Final Balance: {result['final_balance']:.2f}")
        print(f"Final Holdings Value: {result['final_holdings_value']:.2f}")
        print(f"Total Trades: {len(result['trades'])}")
        print(result['trades'])

        # 자산 변화 그래프
        plt.plot(result['total_value'])
        plt.title("Total Asset Value Over Time")
        plt.xlabel("Time Steps")
        plt.ylabel("Asset Value (KRW)")
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
