# Cryptocurrency Trading Bot and Alert System

## 프로젝트 개요

이 리포지토리는 다음 두 가지 주요 기능을 제공합니다:

1. **자동 암호화폐 매매 봇 (`auto_trade_upbit.py`)**  
   - Upbit API를 사용하여 암호화폐를 자동으로 매수/매도합니다.
   - RSI, MACD, ADX, Supertrend 등 다양한 기술 지표를 활용하여 진입 및 청산 시점을 판단합니다.
   - Slack 알림을 통해 거래 현황 및 결과를 실시간으로 확인할 수 있습니다.

2. **암호화폐 가격 알림 시스템 (`crypto_alert_slack.py`)**  
   - Binance API를 이용해 실시간으로 암호화폐 가격을 모니터링합니다.
   - 설정한 가격 범위를 벗어날 경우 Slack으로 즉시 알림을 전송합니다.
   - 가격이 정상 범위로 복귀하면 알림 상태가 초기화됩니다.
   - 가격 요약 메시지는 평일 오전 8시부터 오후 6시(KST)까지 **정각 기준**으로 전송됩니다. (`.env` 설정 가능)
   - 프로그램 시작 시점에는 즉시 1회 요약 메시지를 전송합니다.
   - `.env`에 다음 항목이 없으면 기본값은 `SUMMARY_START_HOUR=8`, `SUMMARY_END_HOUR=18`로 동작합니다.
   - 주말 및 그 외 시간에는 요약 메시지가 전송되지 않으며, 가격 도달 알림은 언제든 실시간 전송됩니다.
   - 단, **동일한 코인에 대해 동일 조건(상단 돌파/하단 이탈)에 대한 알림은 설정한 쿨다운 시간(기본 30분) 내 중복 전송되지 않습니다.**
   - Slack 메시지에 실시간 확인 가능한 TradingView 링크 포함
   - 모든 알림은 로그 파일(`crypto_alert.log`)에 자동 기록됩니다

---

## 주요 기능 및 거래 전략

### `auto_trade_upbit.py`

#### 주요 기능
- **거래 전략**:
  - **매수(Entry) 조건**:
    - **RSI**: 최근 RSI 값이 40 미만일 경우 (과매도 상태)
    - **MACD**: MACD 값이 0 이상일 경우 (상승 모멘텀 확인)
    - **ADX**: ADX 값이 20 이상일 경우 (추세 강도 확인)
    - **Volume Momentum**: 거래량이 증가 추세일 경우
    - **Supertrend**: 현재 가격이 Supertrend 값보다 높을 경우

    위 조건이 모두 충족되면 매수 신호가 발생하며, Slack 알림과 함께 자산의 일정 비율(예: 30%)에 해당하는 금액으로 시장가 매수 주문이 실행됩니다.

  - **매도(Exit) 조건**:
    - **목표 수익 실현**: 수익률이 9% 이상일 경우 매도
    - **Trailing Stop**: 수익률이 1.5% 이상이고 가격이 Supertrend 아래로 하락하면 매도
    - **손절매(Stop Loss)**: 수익률이 -3% 이하일 경우 손실 최소화를 위해 매도

- **거래 관리**:
  - 실시간 자산 평가 및 잔고 확인
  - 매수 후 일정 시간 동안 재매도 방지 (쿨다운 적용)
  - API 호출 빈도 제한 고려
  - 주문 완료 및 실패 내역 Slack 전송

#### 환경 변수 설정
```env
ACCESS_KEY=your_upbit_access_key
SECRET_KEY=your_upbit_secret_key
SLACK_WEBHOOK_URL=your_slack_webhook_url
```

#### 실행 방법
```bash
python auto_trade_upbit.py
```

---

### `crypto_alert_slack.py`

#### 주요 기능
- 실시간 가격 모니터링 (Binance API 활용)
- 설정한 코인의 가격이 상단/하단 경계를 넘을 경우 Slack 알림 전송
- 가격이 정상 범위로 돌아오면 알림 상태 초기화
- 요약 메시지는 평일 오전 ~ 오후 정시마다 전송됨 (기본 08~18시, `.env`에서 설정 가능)
- 시작 시점에 즉시 1회 요약 메시지 전송
- 종료 시 Slack에 종료 메시지 전송
- 가격 도달 알림은 시간과 무관하게 상시 전송됨
- **같은 코인의 같은 조건에 대해서는 30분(1800초) 동안 중복 알림 방지 처리됨**
- Slack 메시지에 TradingView 링크 포함
- 로그 파일(`crypto_alert.log`)에 모든 동작 자동 기록됨

#### 환경 변수 설정 예시
```env
SLACK_WEBHOOK_URL_FOR_ALERT=slack_webhook_url
SUMMARY_START_HOUR=9
SUMMARY_END_HOUR=17
```

- 위 항목이 없으면 기본값 `SUMMARY_START_HOUR=8`, `SUMMARY_END_HOUR=18`이 사용됩니다.

#### 감시 코인 및 조건 설정 예시
```python
crypto_alerts = [
    {"symbol": "BTCUSDT", "above": 98600, "below": 92580},
    {"symbol": "ETHUSDT", "above": 3750, "below": 3400},
    # 추가 코인 설정 가능
]
```

#### 실행 방법
```bash
python crypto_alert_slack.py
```

---

## 설치 및 의존성

### 의존성 설치
```bash
pip install -r requirements.txt
```

---

## 거래 전략 상세 설명

### 매수(Entry) 기준

#### 지표 분석
- **RSI (Relative Strength Index)**: 과매도 상태를 나타내며, 일반적으로 40 미만이면 진입 고려
- **MACD (Moving Average Convergence Divergence)**: 상승 전환을 감지
- **ADX (Average Directional Index)**: 추세 강도를 나타냄, 20 이상이면 유의미한 추세로 간주
- **Supertrend**: 상승 추세 여부 판단에 활용
- **Volume Momentum**: 거래량의 증가 여부 판단

#### 매수 조건 요약
- RSI < 40
- MACD > 0
- ADX > 20
- Volume Momentum > 0
- 현재 가격 > Supertrend

### 매도(Exit) 기준
- 수익률 >= 9%: 목표 수익 실현을 위해 매도
- 수익률 >= 1.5% & 가격이 Supertrend 아래: 모멘텀 약화로 판단하여 매도
- 수익률 <= -3%: 손실 최소화를 위해 즉시 매도

---

## 결론

이 프로젝트는 자동화된 암호화폐 매매 및 실시간 가격 감시 시스템을 통해 체계적인 거래를 구현하는 것을 목표로 합니다. 다양한 기술 지표를 바탕으로 매매 시점을 정교하게 판단하며, Slack 알림을 통해 거래 상태를 실시간으로 모니터링할 수 있도록 설계되었습니다. 향후 전략 고도화와 기능 확장을 통해 더욱 안정적이고 효과적인 시스템으로 발전시킬 예정입니다.
