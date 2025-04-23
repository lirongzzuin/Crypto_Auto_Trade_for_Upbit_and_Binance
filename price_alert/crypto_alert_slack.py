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
UPBIT_API_URL = "https://api.upbit.com/v1/ticker"

# Slack Webhook URL
SLACK_WEBHOOK_URL_FOR_ALERT = os.getenv("SLACK_WEBHOOK_URL_FOR_ALERT")

# 요약 알림 시간 설정 (기본값: 8시 ~ 18시)
SUMMARY_START_HOUR = int(os.getenv("SUMMARY_START_HOUR", 8))
SUMMARY_END_HOUR = int(os.getenv("SUMMARY_END_HOUR", 18))

# 가격 알림 쿨다운 시간 (초)
ALERT_COOLDOWN_SECONDS = 3600

# 감시할 코인 목록 및 가격 조건
crypto_alerts = [
    {"symbol": "BTCUSDT", "above": 97000, "below": 87000},
    {"symbol": "ETHUSDT", "above": 1850, "below": 1600},
    {"symbol": "XRPUSDT", "above": 3, "below": 2},
    {"symbol": "SOLUSDT", "above": 155, "below": 130},
    {"symbol": "ADAUSDT", "above": 0.9, "below": 0.6},
    {"symbol": "HBARUSDT", "above": 0.3, "below": 0.15},
    {"symbol": "TRUMPUSDT", "above": 12, "below": 8},
]

running = True

def get_crypto_price(symbol):
    try:
        response = requests.get(BINANCE_API_URL, params={"symbol": symbol})
        response.raise_for_status()
        return float(response.json()["price"])
    except Exception as e:
        logging.error(f"{symbol} 가격 조회 실패: {e}")
        return None

def get_upbit_price(symbol):
    try:
        symbol_krw = "KRW-" + symbol.replace("USDT", "")
        response = requests.get(UPBIT_API_URL, params={"markets": symbol_krw})
        response.raise_for_status()
        return float(response.json()[0]["trade_price"])
    except Exception as e:
        logging.warning(f"{symbol_krw} 원화 가격 조회 실패: {e}")
        return None

def send_slack_message(message):
    try:
        response = requests.post(SLACK_WEBHOOK_URL_FOR_ALERT, json={"text": message})
        response.raise_for_status()
        logging.info(f"Slack 메시지 전송: {message}")
    except Exception as e:
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

def get_all_prices():
    prices = {}
    for alert in crypto_alerts:
        symbol = alert["symbol"]
        binance_price = get_crypto_price(symbol)
        upbit_price = get_upbit_price(symbol)
        prices[symbol] = {
            "usdt": binance_price,
            "krw": upbit_price
        }
    return prices

def get_summary_message(all_prices):
    message = "📊 *현재 코인 가격 요약 (Binance / Upbit)* 📊\n\n🔗 https://coinmarketcap.com/ko/\n\n"
    for symbol, data in all_prices.items():
        usdt = data["usdt"]
        krw = data["krw"]
        if usdt is not None:
            message += f"- *{symbol}*: {usdt} USDT"
            if krw is not None:
                message += f" / {int(krw):,} KRW"
            message += "\n"
    return message

def monitor_prices():
    global running

    tz = pytz.timezone("Asia/Seoul")
    alerted = {
        alert["symbol"]: {
            "above": {"last_time": None},
            "below": {"last_time": None}
        } for alert in crypto_alerts
    }

    send_slack_message("✅ 코인 가격 모니터링을 시작합니다. (요약: 정시마다 / 알림: 상시)")
    all_prices = get_all_prices()
    send_slack_message(get_summary_message(all_prices))

    now = datetime.now(tz)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    while running:
        now = datetime.now(tz)

        for alert in crypto_alerts:
            symbol = alert["symbol"]
            current_price = get_crypto_price(symbol)
            current_krw = get_upbit_price(symbol)

            if current_price is None:
                continue

            # 상단 돌파
            if current_price > alert["above"]:
                last_time = alerted[symbol]["above"]["last_time"]
                if last_time is None or (now - last_time).total_seconds() >= ALERT_COOLDOWN_SECONDS:
                    msg = (
                        f"🚀 *{symbol} 상단 돌파!*"
                        f"> 현재: {current_price} USDT"
                    )
                    if current_krw is not None:
                        msg += f" / {int(current_krw):,} KRW"
                    msg += f" (설정 상단: {alert['above']})\n"
                    msg += f"🔗 https://www.tradingview.com/symbols/{symbol.replace('USDT', '')}USDT/"
                    send_slack_message(msg)
                    alerted[symbol]["above"]["last_time"] = now

            # 하단 이탈
            if current_price < alert["below"]:
                last_time = alerted[symbol]["below"]["last_time"]
                if last_time is None or (now - last_time).total_seconds() >= ALERT_COOLDOWN_SECONDS:
                    msg = (
                        f"📉 *{symbol} 하단 이탈!*"
                        f"> 현재: {current_price} USDT"
                    )
                    if current_krw is not None:
                        msg += f" / {int(current_krw):,} KRW"
                    msg += f" (설정 하단: {alert['below']})\n"
                    msg += f"🔗 https://www.tradingview.com/symbols/{symbol.replace('USDT', '')}USDT/"
                    send_slack_message(msg)
                    alerted[symbol]["below"]["last_time"] = now

        if now >= next_hour:
            if is_in_summary_time_range():
                all_prices = get_all_prices()
                send_slack_message(get_summary_message(all_prices))
            next_hour += timedelta(hours=1)

        sleep(10)

if __name__ == "__main__":
    monitor_prices()
