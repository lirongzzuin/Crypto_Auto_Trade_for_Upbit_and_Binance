# Cryptocurrency Trading Bot and Alert System

## 프로젝트 개요

이 리포지토리는 두 가지 주요 기능을 제공합니다:

1. **자동화된 암호화폐 매매 봇 (`auto_trade_upbit.py`)**  
   - Upbit API를 사용하여 암호화폐를 자동으로 매수/매도합니다.
   - RSI, MACD, ADX, Supertrend 등 다양한 기술 지표를 활용하여 거래 진입 및 청산 시점을 결정합니다.
   - Slack 알림을 통해 거래 진행 상황과 결과를 실시간으로 모니터링할 수 있습니다.

2. **암호화폐 가격 알림 시스템 (`crypto_alert_slack.py`)**  
   - Binance API를 활용하여 실시간 암호화폐 가격을 모니터링합니다.
   - 설정한 가격 범위를 벗어날 경우 Slack으로 알림을 전송합니다.
   - Kakao 알림 기능은 개발 중입니다.

---

## 주요 기능 및 거래 전략

### `auto_trade_upbit.py`

#### 주요 기능
- **거래 전략**:
  - **매수(Entry) 조건**:
    - **RSI 지표**: 최근 RSI 값이 40 미만일 경우 (과매도 상태로 진입 가능성이 있음)
    - **MACD 지표**: MACD 값이 0 이상일 경우 (상승 모멘텀 확인)
    - **ADX 지표**: ADX 값이 20 이상일 경우 (추세 강도 확인)
    - **Volume Momentum**: 거래량 모멘텀이 양수일 경우 (거래량 증가 추세)
    - **Supertrend 조건**: 현재 가격이 Supertrend 값보다 높을 경우 (추세 반전 확인)
    
    위 조건이 모두 충족되면 매수 신호가 발생하며, Slack 알림과 함께 총 자산의 일정 비율(예: 30%)에 해당하는 금액으로 시장가 매수 주문을 실행합니다.

  - **매도(Exit) 조건**:
    - **목표 수익 실현**: 수익률이 9% 이상일 경우 목표 수익 실현을 위해 매도합니다.
    - **Trailing Stop (후행 스탑)**: 수익률이 1.5% 이상이면서 현재 가격이 Supertrend 아래로 하락하면, 상승 모멘텀이 약화되었다고 판단하여 매도합니다.
    - **손절매 (Stop Loss)**: 수익률이 -3% 이하로 하락하면 추가 손실을 방지하기 위해 즉시 매도 주문을 실행합니다.
    
    매도 주문은 Slack 알림을 통해 주문 내역과 누적 수익/손실이 보고됩니다.

- **거래 관리**:
  - **잔액 및 자산 평가**: 보유 자산과 원화 잔액을 실시간으로 평가하여 주문 금액 및 최소 주문 금액(예: 5,000 KRW)을 확인합니다.
  - **쿨다운 기간**: 매수 후 바로 매도되지 않도록 일정 시간(쿨다운 기간)을 적용합니다.
  - **주문 제한**: API 요청 빈도를 조절하여 업비트의 요청 제한을 준수합니다.
  - **주문 및 수익 관리**: 주문 완료 정보를 기반으로 누적 수익을 계산하며, 주문 실패 시 Slack을 통해 에러 메시지를 전송합니다.

#### 환경 변수 설정
- `ACCESS_KEY`: Upbit API 접근 키
- `SECRET_KEY`: Upbit API 시크릿 키
- `SLACK_WEBHOOK_URL`: Slack Webhook URL

#### 실행 방법
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

#### 주요 기능
- **가격 모니터링**:  
  Binance API를 사용하여 설정된 암호화폐의 실시간 가격을 모니터링합니다.
- **알림 조건**:
  - 설정된 가격 조건(상승/하락)에 도달하면 Slack으로 즉시 알림을 전송합니다.
  - 가격이 정상 범위로 돌아오면 알림 상태를 초기화합니다.

#### 환경 변수 설정
- `SLACK_WEBHOOK_URL_FOR_ALERT`: Slack Webhook URL

#### 알림 설정
- 감시할 코인 및 조건은 `crypto_alerts` 리스트에서 설정합니다:
    ```python
    crypto_alerts = [
        {"symbol": "BTCUSDT", "above": 98600, "below": 92580},
        {"symbol": "ETHUSDT", "above": 3750, "below": 3400},
        ...
    ]
    ```

#### 실행 방법
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
```

## 거래 전략 상세 설명

### 매수(Entry) 시점

#### 지표 기반 분석
- **RSI (Relative Strength Index)**: 14일 기준으로 계산되며, 값이 40 미만일 경우 과매도 상태로 판단합니다.
- **MACD (Moving Average Convergence Divergence)**: 12일과 26일 지수이동평균의 차이를 통해 상승 모멘텀을 확인합니다.
- **ADX (Average Directional Index)**: 추세 강도를 나타내며, 20 이상이면 의미 있는 추세로 판단합니다.
- **Supertrend**: ATR(평균 실제 범위)을 기반으로 계산되며, 현재 가격이 Supertrend 값 위에 있으면 상승 추세로 해석합니다.
- **Volume Momentum**: 최근 거래량의 증감률을 확인하여 거래 활성 증가를 파악합니다.

#### 매수 조건
위 지표들이 모두 다음 조건을 만족하면 매수 신호가 발생합니다:
- RSI < 40
- MACD > 0
- ADX > 20
- Volume Momentum > 0
- 현재 가격 > Supertrend 값

조건 충족 시, 시스템은 총 자산의 일정 비율(예: 30%)에 해당하는 금액으로 시장가 매수 주문을 실행합니다.

### 매도(Exit) 시점
- **목표 수익 실현**: 보유 코인의 현재 가격이 매수 가격 대비 9% 이상의 수익률을 기록하면 목표 수익 실현을 위해 매도합니다.
- **Trailing Stop (후행 스탑)**: 수익률이 1.5% 이상일 때, 가격이 Supertrend 값 아래로 하락하면 상승 모멘텀이 약화되었다고 판단하여 매도합니다.
- **손절매 (Stop Loss)**: 보유 코인의 가격이 매수 가격 대비 3% 이상 하락하면 추가 손실을 방지하기 위해 즉시 매도 주문을 실행합니다.

이와 같이 다양한 기술 지표와 조건을 종합하여 거래 진입과 청산 시점을 결정함으로써 리스크 관리와 수익 실현의 균형을 맞추고 있습니다.

---

## 결론

이 프로젝트는 자동화된 암호화폐 거래와 실시간 가격 알림 시스템을 통해 체계적인 거래 전략을 구현하는 것을 목표로 합니다.  
다양한 기술 지표를 활용하여 매매 시점을 결정하고, Slack 알림을 통해 거래 상태를 실시간 모니터링할 수 있도록 구성되어 있습니다.  
향후 전략 최적화 및 기능 확장을 통해 더욱 발전시킬 예정입니다.
