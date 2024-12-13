import time
import pandas as pd
import numpy as np
from binance.client import Client
from binance.enums import *
from dotenv import load_dotenv
import os
import requests
import signal
import threading

# Load environment variables
load_dotenv()

# Binance API keys
ACCESS_KEY_BINANCE = os.getenv("ACCESS_KEY_BINANCE")
SECRET_KEY_BINANCE = os.getenv("SECRET_KEY_BINANCE")

# Slack webhook URL
SLACK_WEBHOOK_URL_BINANCE = os.getenv("SLACK_WEBHOOK_URL_BINANCE")

# Initialize Binance Client
client = Client(ACCESS_KEY_BINANCE, SECRET_KEY_BINANCE, testnet=False)  # Set testnet=True for testing

# Constants
LEVERAGE = 75
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "UNIUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "LINKUSDT", "1000PEPEUSDT", "AVAXUSDT", "1MBABYDOGEUSDT"]
POSITION_SIZE_RATIO = 0.8
PROFIT_TARGET = 15.0  # %
STOP_LOSS = -5.0  # %
TRAILING_STOP_TRIGGER = 5.0  # %
SHORT_INTERVAL = "5m"
LONG_INTERVAL = "1h"
API_CALL_DELAY = 2  # Delay in seconds between API calls to avoid hitting rate limits

# Variables to track cumulative profit and returns
cumulative_profit = 0.0
initial_balance = 0.0
status_thread = None
running = True

# Helper functions
def send_slack_message(message):
    """Send a message to Slack using the webhook URL."""
    if not SLACK_WEBHOOK_URL_BINANCE:
        print("Slack Webhook URL not configured.")
        return

    payload = {"text": message}
    try:
        response = requests.post(SLACK_WEBHOOK_URL_BINANCE, json=payload)
        if response.status_code != 200:
            print(f"Slack notification failed: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error sending Slack message: {e}")

def fetch_historical_data(symbol, interval, limit=500):
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    return df

def calculate_indicators(df):
    df["rsi"] = calculate_rsi(df)
    df["macd"], df["macd_signal"] = calculate_macd(df)
    df["atr"] = calculate_atr(df)
    df["bollinger_upper"], df["bollinger_lower"] = calculate_bollinger_bands(df)
    df["stoch_rsi"] = calculate_stochastic_rsi(df)
    df["supertrend"] = calculate_supertrend(df)
    return df

def calculate_rsi(data, window=14):
    delta = data["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data):
    ema12 = data["close"].ewm(span=12, adjust=False).mean()
    ema26 = data["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_atr(data, window=14):
    high = data["high"]
    low = data["low"]
    close = data["close"]
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def calculate_bollinger_bands(data, window=20):
    sma = data["close"].rolling(window=window).mean()
    std = data["close"].rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

def calculate_stochastic_rsi(data, window=14):
    min_val = data["rsi"].rolling(window=window).min()
    max_val = data["rsi"].rolling(window=window).max()
    return (data["rsi"] - min_val) / (max_val - min_val) * 100

def calculate_supertrend(data, atr_period=10, multiplier=3):
    atr = calculate_atr(data, atr_period)
    hl2 = (data["high"] + data["low"]) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    supertrend = np.where(data["close"] > upper_band.shift(), upper_band, lower_band)
    return supertrend

def get_available_balance():
    global initial_balance
    balance_info = client.futures_account_balance()
    for asset in balance_info:
        if asset["asset"] == "USDT":
            balance = float(asset["balance"])
            if initial_balance == 0.0:
                initial_balance = balance
            return balance
    return 0.0

def transfer_to_spot(excess_amount):
    try:
        result = client.universal_transfer(
            asset="USDT",
            amount=str(excess_amount),
            type="FUTURE_MAIN"  # Transfer from Futures to Spot
        )
        send_slack_message(f"Transferred {excess_amount} USDT from Futures to Spot.")
        print(f"Transferred {excess_amount} USDT from Futures to Spot.")
    except Exception as e:
        send_slack_message(f"Error transferring to Spot: {e}")
        print(f"Error transferring to Spot: {e}")

def place_order(symbol, side, quantity):
    try:
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )
        return order
    except Exception as e:
        print(f"Error placing order: {e}")
        send_slack_message(f"Error placing order for {symbol}: {e}")
        return None

def send_periodic_status():
    global running
    while running:
        balance = get_available_balance()
        send_slack_message(f"[Status] Bot running. Current Futures balance: {balance:.2f} USDT\nCumulative Profit: {cumulative_profit:.2f} USDT\nCumulative Return: {(cumulative_profit / initial_balance) * 100:.2f}%")
        time.sleep(1200)  # Send status every 20 minutes

def handle_exit_signal(signal_received, frame):
    global running
    running = False
    send_slack_message("[Stop] Binance Futures trading bot stopped (Signal Received).")
    print("Bot stopped via signal.")
    exit(0)

def main():
    global cumulative_profit
    global status_thread
    global running

    send_slack_message("[Start] Binance Futures trading bot started.")

    # Start status thread
    status_thread = threading.Thread(target=send_periodic_status, daemon=True)
    status_thread.start()

    positions = {symbol: None for symbol in SYMBOLS}

    try:
        while running:
            # Check available balance at the start of each loop
            balance = get_available_balance()
            if balance > 200:
                excess_amount = balance - 100  # Leave 20 USDT in Futures
                transfer_to_spot(excess_amount)

            if balance <= 0:
                print("Insufficient balance.")
                send_slack_message("Insufficient balance for trading.")
                break

            signals = []

            for symbol in SYMBOLS:
                print(f"Fetching data for {symbol}...")
                short_data = fetch_historical_data(symbol, SHORT_INTERVAL)
                long_data = fetch_historical_data(symbol, LONG_INTERVAL)

                short_data = calculate_indicators(short_data)
                long_data = calculate_indicators(long_data)

                last_short = short_data.iloc[-1]
                last_long = long_data.iloc[-1]

                position_size = balance * POSITION_SIZE_RATIO / LEVERAGE

                # Generate signals
                if positions[symbol] is None:
                    if last_short["rsi"] < 30 and last_short["stoch_rsi"] < 20 and last_short["macd"] > last_short["macd_signal"] and last_long["rsi"] < 50:
                        signals.append({"symbol": symbol, "type": "LONG", "priority": last_short["rsi"]})
                    elif last_short["rsi"] > 70 and last_short["stoch_rsi"] > 80 and last_short["macd"] < last_short["macd_signal"] and last_long["rsi"] > 50:
                        signals.append({"symbol": symbol, "type": "SHORT", "priority": 100 - last_short["rsi"]})

            # Sort signals by priority (lowest RSI for LONG, highest for SHORT)
            signals = sorted(signals, key=lambda x: x["priority"])

            # Execute the highest-priority signal
            if signals:
                best_signal = signals[0]
                symbol = best_signal["symbol"]
                trade_type = best_signal["type"]
                quantity = round(position_size / short_data.iloc[-1]["close"], 3)

                print(f"Placing {trade_type} order for {quantity} {symbol}...")
                order_side = SIDE_BUY if trade_type == "LONG" else SIDE_SELL
                order = place_order(symbol, order_side, quantity)

                if order:
                    positions[symbol] = {"type": trade_type, "entry_price": short_data.iloc[-1]["close"], "quantity": quantity}
                    send_slack_message(f"[{trade_type}] Entered {symbol} at {short_data.iloc[-1]['close']} with quantity {quantity}")

            # Manage existing positions
            for symbol, position in positions.items():
                if position:
                    last_short = fetch_historical_data(symbol, SHORT_INTERVAL).iloc[-1]
                    profit_loss = ((last_short["close"] - position["entry_price"]) / position["entry_price"]) * LEVERAGE * 100 if position["type"] == "LONG" else ((position["entry_price"] - last_short["close"]) / position["entry_price"]) * LEVERAGE * 100

                    if profit_loss >= PROFIT_TARGET or profit_loss <= STOP_LOSS:
                        exit_side = SIDE_SELL if position["type"] == "LONG" else SIDE_BUY
                        print(f"Closing {position['type']} position for {symbol}...")
                        order = place_order(symbol, exit_side, position["quantity"])
                        if order:
                            profit_amount = (profit_loss / 100) * balance
                            cumulative_profit += profit_amount
                            positions[symbol] = None

                            send_slack_message(f"[{position['type']} EXIT] Closed {symbol} at {last_short['close']} with P/L {profit_loss:.2f}%\nCumulative Profit: {cumulative_profit:.2f} USDT\nCumulative Return: {(cumulative_profit / initial_balance) * 100:.2f}%")

            time.sleep(API_CALL_DELAY)

    except Exception as e:
        print(f"Error: {e}")
        send_slack_message(f"Error in trading loop: {e}")

    finally:
        running = False
        send_slack_message("[Stop] Binance Futures trading bot stopped.")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)
    main()
