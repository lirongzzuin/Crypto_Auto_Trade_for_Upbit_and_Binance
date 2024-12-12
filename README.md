# Cryptocurrency Trading Bot and Alert System

## 프로젝트 개요

이 프로젝트는 암호화폐 시장에서 다음 주요 기능을 제공합니다:

1. **자동 매매 시스템 (`auto_trade_upbit.py`)**
   - Upbit API를 사용하여 자동으로 암호화폐를 매수/매도합니다.
   - 다양한 기술 지표를 활용하여 최적의 매수/매도 시점을 포착합니다.
   - Slack 알림을 통해 거래 상황을 실시간으로 모니터링합니다.

2. **가격 알림 시스템 (`crypto_alert_slack.py`)**
   - Binance API를 사용하여 실시간 암호화폐 가격을 모니터링합니다.
   - 설정된 조건에 따라 Slack으로 가격 상승/하락 알림을 전송합니다.

---

## 주요 기능

### `auto_trade_upbit.py`
- **기능**:
  - 기술 지표 기반 자동 매매:
    - RSI, MACD, ADX, Supertrend 등의 지표를 활용.
  - 누적 수익 계산 및 Slack으로 실시간 거래 보고.
  - Cooldown Period 설정으로 매수 직후의 매도를 방지.
  - 30분마다 매도 시그널 추적 상태를 Slack으로 알림.

- **환경 변수 설정**:
  `.env` 파일에 다음 환경 변수를 추가해야 합니다:
  ```plaintext
  ACCESS_KEY=your_upbit_access_key
  SECRET_KEY=your_upbit_secret_key
  SLACK_WEBHOOK_URL=your_slack_webhook_url
