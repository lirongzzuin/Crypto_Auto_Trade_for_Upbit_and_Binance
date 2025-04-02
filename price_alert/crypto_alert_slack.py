import requests
from time import sleep, time
import os
import signal
from dotenv import load_dotenv
from datetime import datetime
import pytz

# .env 파일 로드
load_dotenv()

# Binance API endpoint
BINANCE_API_URL = "https://api.binance.com/api/v3/ticker/price"

# Slack Webhook URL
SLACK_WEBHOOK_URL_FOR_ALERT = os.getenv("SLACK_WEBHOOK_URL_FOR_ALERT")

# 감시할 코인 목록 및 가격 조건
crypto_alerts = [
    {"symbol": "BTCUSDT", "above": 92000, "below": 74000},
    {"symbol": "ETHUSDT", "above": 2230, "below": 1700},
    {"symbol": "SOLUSDT", "above": 150, "below": 100},
    {"symbol": "ADAUSDT", "above": 0.8, "below": 0.6},
    {"symbol": "HBARUSDT", "above": 0.285, "below": 0.18},
    {"symbol": "TRUMPUSDT", "above": 12.5, "below": 9.7},
]

running = True

def get_crypto_price(symbol):
    try:
        response = requests.get(BINANCE_API_URL, params={"symbol": symbol})
        response.raise_for_status()
        data = response.json()
        return float(data["price"])
    except requests.exceptions.RequestException as e:
        print(f"⚠️ {symbol} 가격을 가져오는 중 오류 발생: {e}")
        return None

def send_slack_message(message):
    try:
        response = requests.post(SLACK_WEBHOOK_URL_FOR_ALERT, json={"text": message})
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Slack 메시지 전송 실패: {e}")

def signal_handler(sig, frame):
    global running
    send_slack_message("⚠️ 코인 가격 모니터링이 종료되었습니다.")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def is_korean_weekday_business_hours():
    now = datetime.now(pytz.timezone("Asia/Seoul"))
    return now.weekday() < 5 and 8 <= now.hour < 18  # 평일 월~금, 08:00~17:59

def monitor_prices():
    global running
    alerted = {alert["symbol"]: {"above": False, "below": False} for alert in crypto_alerts}
    last_summary_time = time()

    send_slack_message("📢 코인 가격 모니터링이 시작되었습니다. (알림 상시 + 요약은 평일 8~18시)")

    while running:
        now = datetime.now(pytz.timezone("Asia/Seoul"))
        summary_message = "📊 **현재 코인 가격 요약** 📊\n\nhttps://coinmarketcap.com/ko/\n\n"

        for alert in crypto_alerts:
            symbol = alert["symbol"]
            current_price = get_crypto_price(symbol)

            if current_price is None:
                continue

            # 가격 도달 알림 (상시)
            if not alerted[symbol]["above"] and current_price > alert["above"]:
                send_slack_message(f"🚀 {symbol} 가격이 {alert['above']}을 돌파했습니다! 현재 가격: {current_price}")
                send_slack_message("🔥 불장 시작?!! 🚀🚀🚀")
                alerted[symbol]["above"] = True

            if not alerted[symbol]["below"] and current_price < alert["below"]:
                send_slack_message(f"📉 {symbol} 가격이 {alert['below']} 아래로 떨어졌습니다! 현재 가격: {current_price}")
                send_slack_message("😭 저점인거죠...? 지금인거죠...? 📉")
                alerted[symbol]["below"] = True

            if alerted[symbol]["above"] and current_price <= alert["above"]:
                alerted[symbol]["above"] = False

            if alerted[symbol]["below"] and current_price >= alert["below"]:
                alerted[symbol]["below"] = False

            # 요약 메시지 구성 (조건만족 시 전송 예정)
            summary_message += f"- {symbol}: {current_price} USDT\n"

        # 1시간마다 요약 메시지 전송 (단, 조건 만족 시에만)
        if time() - last_summary_time >= 3600:
            if is_korean_weekday_business_hours():
                send_slack_message(summary_message)
            last_summary_time = time()

        sleep(10)

if __name__ == "__main__":
    monitor_prices()