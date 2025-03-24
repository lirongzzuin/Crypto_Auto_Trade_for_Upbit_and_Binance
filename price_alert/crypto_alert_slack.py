import requests
from time import sleep, time
import os
import signal
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Binance API endpoint
BINANCE_API_URL = "https://api.binance.com/api/v3/ticker/price"

# Slack Webhook URL (환경 변수 사용 추천)
# SLACK_WEBHOOK_URL_FOR_ALERT = os.getenv("SLACK_WEBHOOK_URL_FOR_ALERT")
SLACK_WEBHOOK_URL_FOR_ALERT = os.getenv("SLACK_WEBHOOK_URL_FOR_ALERT")

# 감시할 코인 목록 및 가격 조건
crypto_alerts = [
    {"symbol": "BTCUSDT", "above": 92000, "below": 74000},
    {"symbol": "ETHUSDT", "above": 2230, "below": 1680},
    {"symbol": "SOLUSDT", "above": 150, "below": 100},
    {"symbol": "ADAUSDT", "above": 0.77, "below": 0.6},
    {"symbol": "HBARUSDT", "above": 0.285, "below": 0.18},
    {"symbol": "TRUMPUSDT", "above": 12.5, "below": 9.7},
]

# 종료 감지를 위한 플래그
running = True

# 현재 코인 가격 가져오기
def get_crypto_price(symbol):
    try:
        response = requests.get(BINANCE_API_URL, params={"symbol": symbol})
        response.raise_for_status()
        data = response.json()
        return float(data["price"])
    except requests.exceptions.RequestException as e:
        print(f"⚠️ {symbol} 가격을 가져오는 중 오류 발생: {e}")
        return None

# Slack 메시지 전송
def send_slack_message(message):
    try:
        response = requests.post(SLACK_WEBHOOK_URL_FOR_ALERT, json={"text": message})
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Slack 메시지 전송 실패: {e}")

# 종료 신호 처리 함수
def signal_handler(sig, frame):
    global running
    send_slack_message("⚠️ 코인 가격 모니터링이 종료되었습니다.")
    running = False

# 종료 신호 등록
signal.signal(signal.SIGINT, signal_handler)  # Ctrl + C
signal.signal(signal.SIGTERM, signal_handler) # 시스템 종료

# 가격 감시 루프
def monitor_prices():
    global running

    # 알람 상태 추적 (초기값: False)
    alerted = {alert["symbol"]: {"above": False, "below": False} for alert in crypto_alerts}

    # 1시간마다 가격 요약을 위한 타이머 (초 단위)
    last_summary_time = time()

    # 스크립트 시작 알림
    send_slack_message("📢 코인 가격 모니터링이 시작되었습니다. (실시간 감지 + 2시간 간격 요약)")

    while running:
        summary_message = "📊 **현재 코인 가격 요약** 📊\n\nhttps://coinmarketcap.com/ko/\n\n"

        for alert in crypto_alerts:
            symbol = alert["symbol"]
            current_price = get_crypto_price(symbol)

            if current_price is None:
                continue

            # 가격 상단 돌파 시 알림
            if not alerted[symbol]["above"] and current_price > alert["above"]:
                send_slack_message(f"🚀 {symbol} 가격이 {alert['above']}을 돌파했습니다! 현재 가격: {current_price}")
                send_slack_message("🔥 불장 시작?!! 🚀🚀🚀")
                alerted[symbol]["above"] = True

            # 가격 하단 이탈 시 알림
            if not alerted[symbol]["below"] and current_price < alert["below"]:
                send_slack_message(f"📉 {symbol} 가격이 {alert['below']} 아래로 떨어졌습니다! 현재 가격: {current_price}")
                send_slack_message("😭 저점인거죠...? 지금인거죠...? 📉")
                alerted[symbol]["below"] = True

            # 가격이 정상 범위로 돌아오면 알람 상태 초기화
            if alerted[symbol]["above"] and current_price <= alert["above"]:
                alerted[symbol]["above"] = False

            if alerted[symbol]["below"] and current_price >= alert["below"]:
                alerted[symbol]["below"] = False

            # 2시간 요약 메시지 생성
            summary_message += f"- {symbol}: {current_price} USDT\n"

        # 1시간마다 현재 가격 요약 전송
        if time() - last_summary_time >= 7200:  # 1시간(3600초) 경과 시
            send_slack_message(summary_message)
            last_summary_time = time()

        sleep(10)  # 30초마다 가격 확인

if __name__ == "__main__":
    monitor_prices()