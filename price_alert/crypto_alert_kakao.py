import requests
import json
import os
from time import sleep
from dotenv import load_dotenv

# Binance API endpoint
BINANCE_API_URL = "https://api.binance.com/api/v3/ticker/price"

# 환경 변수 로드
load_dotenv()

# 카카오 REST API 키 및 Access Token
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")
KAKAO_ACCESS_TOKEN = os.getenv("KAKAO_ACCESS_TOKEN")

# List of crypto alerts
crypto_alerts = [
    {"symbol": "BTCUSDT", "above": 98700, "below": 97530},
    {"symbol": "ETHUSDT", "above": 4000, "below": 3400},
    {"symbol": "ADAUSDT", "above": 1.15, "below": 0.85},
    {"symbol": "SOLUSDT", "above": 250, "below": 199},
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

# Function to send a KakaoTalk message
def send_kakao_message(text):
    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    headers = {
        "Authorization": f"Bearer {KAKAO_ACCESS_TOKEN}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    payload = {
        "template_object": json.dumps({
            "object_type": "text",
            "text": text,
            "link": {
                "web_url": "https://www.binance.com",
                "mobile_web_url": "https://www.binance.com"
            },
        })
    }

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        print("카카오톡 메시지가 전송되었습니다.")
    except requests.exceptions.RequestException as e:
        print(f"카카오톡 메시지 전송 실패: {e}")

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
                message = f"{symbol} 의 가격이 {alert['above']} 위로 올랐어요! 현재 가격: {current_price}"
                send_kakao_message(message)
                alerted[symbol]["above"] = True

            if not alerted[symbol]["below"] and current_price < alert["below"]:
                message = f"{symbol} 의 가격이 {alert['below']} 아래로 떨어졌어요! 현재 가격: {current_price}"
                send_kakao_message(message)
                alerted[symbol]["below"] = True

            # Reset alerts if price goes back to normal range
            if alerted[symbol]["above"] and current_price <= alert["above"]:
                alerted[symbol]["above"] = False

            if alerted[symbol]["below"] and current_price >= alert["below"]:
                alerted[symbol]["below"] = False

        sleep(30)  # Check prices every 30 seconds

if __name__ == "__main__":
    monitor_prices()
