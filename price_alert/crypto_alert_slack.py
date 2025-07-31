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
ALERT_COOLDOWN_SECONDS = 7200

# 감시할 코인 목록 및 가격 조건
crypto_alerts = [
    {"symbol": "BTCUSDT", "above": 125000, "below": 95000},
    {"symbol": "ETHUSDT", "above": 5000, "below": 3000},
    {"symbol": "XRPUSDT", "above": 5, "below": 2},
    {"symbol": "SOLUSDT", "above": 230, "below": 150},
    {"symbol": "ADAUSDT", "above": 1, "below": 0.5}
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

def get_upbit_price(symbol, retries=3, delay=1):
    symbol_krw = "KRW-" + symbol.replace("USDT", "")
    for attempt in range(retries):
        try:
            response = requests.get(UPBIT_API_URL, params={"markets": symbol_krw}, timeout=2)
            response.raise_for_status()
            return float(response.json()[0]["trade_price"])
        except Exception as e:
            logging.warning(f"[{symbol_krw}] 원화 가격 조회 실패 (시도 {attempt+1}/{retries}): {e}")
            sleep(delay)
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
        if upbit_price is None:
            logging.warning(f"{symbol}의 업비트 원화 가격이 None입니다 (슬랙 메시지에서 누락될 수 있음)")
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

def get_next_summary_time(now):
    # 9시부터 시작해서 4시간 간격의 시간대
    base_hour = 9
    hour_list = [(base_hour + 4 * i) % 24 for i in range(6)]  # [9, 13, 17, 21, 1, 5]

    today = now.replace(minute=0, second=0, microsecond=0)
    candidate_times = []

    for h in hour_list:
        candidate_time = today.replace(hour=h)
        if h < now.hour:
            # 시각이 지났으면 다음 날로 이월
            candidate_time += timedelta(days=1 if h <= 5 else 0)
        candidate_times.append(candidate_time)

    # 가장 가까운 미래의 시간 반환
    next_time = min([t for t in candidate_times if t > now])
    return next_time

def monitor_prices():
    global running

    tz = pytz.timezone("Asia/Seoul")
    alerted = {
        alert["symbol"]: {
            "above": {"last_time": None},
            "below": {"last_time": None}
        } for alert in crypto_alerts
    }

    send_slack_message("✅ 코인 가격 모니터링을 시작합니다. (요약: 4시간 간격 / 알림: 상시)")
    all_prices = get_all_prices()
    send_slack_message(get_summary_message(all_prices))

    now = datetime.now(tz)
    next_summary_time = get_next_summary_time(now)

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

        if now >= next_summary_time:
            if is_in_summary_time_range():
                all_prices = get_all_prices()
                send_slack_message(get_summary_message(all_prices))
            next_summary_time += timedelta(hours=4)

        sleep(10)

if __name__ == "__main__":
    monitor_prices()
