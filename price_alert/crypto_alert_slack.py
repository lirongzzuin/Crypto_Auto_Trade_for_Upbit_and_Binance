import requests
from time import sleep
import os

# Binance API endpoint
BINANCE_API_URL = "https://api.binance.com/api/v3/ticker/price"

# Slack Webhook URL
SLACK_WEBHOOK_URL_FOR_ALERT = os.getenv("SLACK_WEBHOOK_URL_FOR_ALERT")

# List of crypto alerts
crypto_alerts = [
    {"symbol": "BTCUSDT", "above": 98600, "below": 92580},
    {"symbol": "ETHUSDT", "above": 3750, "below": 3400},
    {"symbol": "ADAUSDT", "above": 1.15, "below": 0.85},
    {"symbol": "SOLUSDT", "above": 240, "below": 199},
    # Add more as needed
]

# Function to get the current price of a cryptocurrency
def get_crypto_price(symbol):
    try:
        response = requests.get(BINANCE_API_URL, params={"symbol": symbol})
        response.raise_for_status()
        data = response.json()
        return float(data["price"])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None

# Function to send a Slack message
def send_slack_message(message):
    try:
        response = requests.post(SLACK_WEBHOOK_URL_FOR_ALERT, json={"text": message})
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error sending Slack message: {e}")

# Main monitoring loop
def monitor_prices():
    alerted = {}
    for alert in crypto_alerts:
        alerted[alert["symbol"]] = {"above": False, "below": False}

    while True:
        for alert in crypto_alerts:
            symbol = alert["symbol"]
            current_price = get_crypto_price(symbol)

            if current_price is None:
                continue

            if not alerted[symbol]["above"] and current_price > alert["above"]:
                send_slack_message(f"{symbol} 의 가격이 {alert['above']} 위로 올랐어요! : 현재 가격 {current_price}")
                send_slack_message("와ㅏㅏㅏㅏ!! 모두들 불장을 맞으라!! 달려들어!!")
                alerted[symbol]["above"] = True

            if not alerted[symbol]["below"] and current_price < alert["below"]:
                send_slack_message(f"{symbol} 의 가격이 {alert['below']} 밑으로 떨어졌어요! : 현재 가격 {current_price}")
                send_slack_message("으아아아악!!! 추미애로 대응해!!!")
                alerted[symbol]["below"] = True

            # Reset alerts if price goes back to normal range
            if alerted[symbol]["above"] and current_price <= alert["above"]:
                alerted[symbol]["above"] = False

            if alerted[symbol]["below"] and current_price >= alert["below"]:
                alerted[symbol]["below"] = False

        sleep(30)  # Check prices every 30 seconds

if __name__ == "__main__":
    monitor_prices()
