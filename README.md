# Cryptocurrency Trading Bot and Alert System

## 프로젝트 개요

이 리포지토리는 다음 두 가지 주요 기능을 제공합니다:

1. **자동화된 암호화폐 매매 봇 (`auto_trade_upbit.py`)**  
   - Upbit API를 사용하여 자동으로 암호화폐를 매수/매도합니다.
   - RSI, MACD, ADX, Supertrend 등 다양한 기술 지표를 활용하여 매수/매도 시그널을 포착합니다.
   - Slack 알림을 통해 거래 상태를 실시간으로 확인할 수 있습니다.

2. **암호화폐 가격 알림 시스템 (`crypto_alert_slack.py`)**  
   - Binance API를 사용하여 실시간으로 암호화폐 가격을 모니터링합니다.
   - 설정한 가격 이상 상승하거나 하락할 경우 Slack으로 알림을 전송합니다.
   - kakao 알림 기능은 아직 미완성

---

## 주요 기능

### `auto_trade_upbit.py`
- **기능**:
  - RSI, MACD 등 기술 지표를 기반으로 매수 및 매도 시점 결정.
  - 누적 수익 계산 및 Slack을 통한 거래 보고.
  - Cooldown Period를 통해 매수 직후 바로 매도 방지.
  - 30분 간격으로 현재 보유 자산에 대한 매도 시그널 알림.
  
- **환경 변수 설정**:
  - `ACCESS_KEY`: Upbit API 접근 키.
  - `SECRET_KEY`: Upbit API 시크릿 키.
  - `SLACK_WEBHOOK_URL`: Slack Webhook URL.

- **실행 방법**:
  1. `.env` 파일에 필요한 환경 변수를 설정합니다:
     ```
     ACCESS_KEY=your_upbit_access_key
     SECRET_KEY=your_upbit_secret_key
     SLACK_WEBHOOK_URL=your_slack_webhook_url
     ```
  2. 스크립트를 실행합니다:
     ```bash
     python auto_trade_upbit.py
     ```

---

### `crypto_alert_slack.py`
- **기능**:
  - 설정된 가격 조건(상승/하락)을 기준으로 암호화폐 가격을 모니터링.
  - 조건이 충족되면 Slack 알림 전송.
  - 가격이 정상 범위로 돌아오면 알림 상태를 초기화.

- **환경 변수 설정**:
  - `SLACK_WEBHOOK_URL_FOR_ALERT`: Slack Webhook URL.

- **알림 설정**:
  - `crypto_alerts` 리스트에서 감시할 코인과 조건을 설정합니다:
    ```python
    crypto_alerts = [
        {"symbol": "BTCUSDT", "above": 98600, "below": 92580},
        {"symbol": "ETHUSDT", "above": 3750, "below": 3400},
        ...
    ]
    ```

- **실행 방법**:
  1. `.env` 파일에 Slack Webhook URL을 설정합니다:
     ```
     SLACK_WEBHOOK_URL_FOR_ALERT=your_slack_webhook_url
     ```
  2. 스크립트를 실행합니다:
     ```bash
     python crypto_alert_slack.py
     ```

---

## 설치 및 의존성

### 의존성 설치
필요한 라이브러리는 `requirements.txt`를 통해 설치할 수 있습니다:
```bash
pip install -r requirements.txt
