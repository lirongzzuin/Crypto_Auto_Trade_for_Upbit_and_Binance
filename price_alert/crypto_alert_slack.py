import requests
from time import sleep
import os
import signal
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz

# .env 파일 로드
load_dotenv()

# Binance API endpoint
BINANCE_API_URL = "https://api.binance.com/api/v3/ticker/price"

# Slack Webhook URL
SLACK_WEBHOOK_URL_FOR_ALERT = os.getenv("SLACK_WEBHOOK_URL_FOR_ALERT")

# 요약 알림 시간 설정 (기본값 8~18시)
SUMMARY_START_HOUR = int(os.getenv("SUMMARY_START_HOUR", 8))
SUMMARY_END_HOUR = int(os.getenv("SUMMARY_END_HOUR", 18))

# 감시할 코인 목록 및 가격 조건
crypto_alerts = [
    {"symbol": "BTCUSDT", "above": 83000, "below": 71000},
    {"symbol": "ETHUSDT", "above": 1700, "below": 1400},
    {"symbol": "XRPUSDT", "above": 2.5, "below": 1.6},
    {"symbol": "SOLUSDT", "above": 120, "below": 98},
    {"symbol": "ADAUSDT", "above": 0.7, "below": 0.5},
    {"symbol": "HBARUSDT", "above": 0.2, "below": 0.132},
    {"symbol": "TRUMPUSDT", "above": 10, "below": 7},
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

def is_in_summary_time_range():
    now = datetime.now(pytz.timezone("Asia/Seoul"))
    return now.weekday() < 5 and SUMMARY_START_HOUR <= now.hour <= SUMMARY_END_HOUR

def get_summary_message():
    message = "📊 **현재 코인 가격 요약** 📊\n\nhttps://coinmarketcap.com/ko/\n\n"
    for alert in crypto_alerts:
        symbol = alert["symbol"]
        price = get_crypto_price(symbol)
        if price is not None:
            message += f"- {symbol}: {price} USDT\n"
    return message

def monitor_prices():
    global running
    alerted = {alert["symbol"]: {"above": False, "below": False} for alert in crypto_alerts}
    
    # 최초 실행 시 현재 가격 요약 전송
    send_slack_message("📢 코인 가격 모니터링이 시작되었습니다. (알림 상시 + 요약은 설정된 시간대 정시마다)")
    summary_message = get_summary_message()
    send_slack_message(summary_message)

    # 다음 정시 계산
    tz = pytz.timezone("Asia/Seoul")
    now = datetime.now(tz)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    while running:
        now = datetime.now(tz)

        for alert in crypto_alerts:
            symbol = alert["symbol"]
            current_price = get_crypto_price(symbol)

            if current_price is None:
                continue

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

        if now >= next_hour:
            if is_in_summary_time_range():
                summary_message = get_summary_message()
                send_slack_message(summary_message)
            next_hour = next_hour + timedelta(hours=1)

        sleep(10)

if __name__ == "__main__":
    monitor_prices()
