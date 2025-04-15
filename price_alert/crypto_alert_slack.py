import requests
from time import sleep
import os
import signal
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz

# 로깅 설정
logging.basicConfig(
    filename='crypto_alert.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)

# .env 파일 로드
load_dotenv()

# Binance API endpoint
BINANCE_API_URL = "https://api.binance.com/api/v3/ticker/price"

# Slack Webhook URL
SLACK_WEBHOOK_URL_FOR_ALERT = os.getenv("SLACK_WEBHOOK_URL_FOR_ALERT")

# 요약 알림 시간 설정 (기본값: 8시 ~ 18시)
SUMMARY_START_HOUR = int(os.getenv("SUMMARY_START_HOUR", 8))
SUMMARY_END_HOUR = int(os.getenv("SUMMARY_END_HOUR", 18))

# 가격 알림 쿨다운 시간 (초)
ALERT_COOLDOWN_SECONDS = 1800

# 감시할 코인 목록 및 가격 조건
crypto_alerts = [
    {"symbol": "BTCUSDT", "above": 87000, "below": 71000},
    {"symbol": "ETHUSDT", "above": 1750, "below": 1400},
    {"symbol": "XRPUSDT", "above": 3, "below": 1.6},
    {"symbol": "SOLUSDT", "above": 150, "below": 98},
    {"symbol": "ADAUSDT", "above": 0.8, "below": 0.5},
    {"symbol": "HBARUSDT", "above": 0.25, "below": 0.132},
    {"symbol": "TRUMPUSDT", "above": 12, "below": 7},
]

running = True

def get_crypto_price(symbol):
    try:
        response = requests.get(BINANCE_API_URL, params={"symbol": symbol})
        response.raise_for_status()
        data = response.json()
        return float(data["price"])
    except requests.exceptions.RequestException as e:
        logging.error(f"{symbol} 가격을 가져오는 중 오류 발생: {e}")
        return None

def send_slack_message(message):
    try:
        response = requests.post(SLACK_WEBHOOK_URL_FOR_ALERT, json={"text": message})
        response.raise_for_status()
        logging.info(f"Slack 메시지 전송: {message}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Slack 메시지 전송 실패: {e}")

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
    message = "📊 *현재 코인 가격 요약* 📊\n\n🔗 https://coinmarketcap.com/ko/\n\n"
    for alert in crypto_alerts:
        symbol = alert["symbol"]
        price = get_crypto_price(symbol)
        if price is not None:
            message += f"- *{symbol}*: {price} USDT\n"
    return message

def monitor_prices():
    global running

    tz = pytz.timezone("Asia/Seoul")
    alerted = {
        alert["symbol"]: {
            "above": {"last_time": None, "last_price": None},
            "below": {"last_time": None, "last_price": None}
        } for alert in crypto_alerts
    }

    # 시작 알림
    send_slack_message("✅ 코인 가격 모니터링을 시작합니다. (요약: 정시마다 / 알림: 상시)")
    summary_message = get_summary_message()
    send_slack_message(summary_message)

    now = datetime.now(tz)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    while running:
        now = datetime.now(tz)

        for alert in crypto_alerts:
            symbol = alert["symbol"]
            current_price = get_crypto_price(symbol)
            if current_price is None:
                continue

            # 상단 돌파
            if current_price > alert["above"]:
                last_time = alerted[symbol]["above"]["last_time"]
                last_price = alerted[symbol]["above"]["last_price"]
                if last_time is None or (now - last_time).total_seconds() >= ALERT_COOLDOWN_SECONDS:
                    send_slack_message(
                        f"🚀 *{symbol} 상단 돌파!* 현재 가격: {current_price} USDT\n"
                        f"> 설정 상단: {alert['above']}\n"
                        f"🔗 https://www.tradingview.com/symbols/{symbol.replace('USDT', '')}USDT/"
                    )
                    alerted[symbol]["above"]["last_time"] = now
                    alerted[symbol]["above"]["last_price"] = current_price

            # 하단 이탈
            if current_price < alert["below"]:
                last_time = alerted[symbol]["below"]["last_time"]
                last_price = alerted[symbol]["below"]["last_price"]
                if last_time is None or (now - last_time).total_seconds() >= ALERT_COOLDOWN_SECONDS:
                    send_slack_message(
                        f"📉 *{symbol} 하단 이탈!* 현재 가격: {current_price} USDT\n"
                        f"> 설정 하단: {alert['below']}\n"
                        f"🔗 https://www.tradingview.com/symbols/{symbol.replace('USDT', '')}USDT/"
                    )
                    alerted[symbol]["below"]["last_time"] = now
                    alerted[symbol]["below"]["last_price"] = current_price

        # 정시 요약
        if now >= next_hour:
            if is_in_summary_time_range():
                summary_message = get_summary_message()
                send_slack_message(summary_message)
            next_hour = next_hour + timedelta(hours=1)

        sleep(10)

if __name__ == "__main__":
    monitor_prices()
