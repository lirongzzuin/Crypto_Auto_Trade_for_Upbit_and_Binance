# Cryptocurrency Trading Bot & Alert System

본 프로젝트는 **Upbit 자동매매 봇**과 **Binance 시세 알림 도구**로 구성된 실전형 트레이딩 시스템입니다.  
**기술 PM(Technical Program Manager)** 전환과 **백엔드 개발자** 포지션 지원 모두를 염두에 두고, 문제정의–설계–구현–운영까지의 전체 흐름과 엔지니어링 품질을 보여줍니다.

---

## 1. Executive Summary

- 목표: **일관된 매매 규율**과 **리스크 통제**, **운영 가시성**을 갖춘 자동화 트레이딩 파이프라인 구축
- 산출물:
  - `auto_trade_upbit.py`: Upbit API 기반 자동매매(지표·ICT·리스크·주문엔진·관측성 포함)
  - `crypto_alert_slack.py`: Binance 시세 경계 돌파 알림(정시 요약, 중복 억제, 운영 로그)
- 핵심 지표(KPI):
  - 체결 성공률, 슬리피지 상한 준수율, 일간 드로우다운 초과 빈도, 장애 복구 시간, 알림 중복률

---

## 2. 역할별 하이라이트

### A) 기술 PM 관점
- **문제정의 → 성공지표 → 단계적 고도화**를 담은 구조  
  - 요구사항: 자동매매 일관성, 손실 제한, 실시간 가시성, 무중단 운영  
  - 제약조건: 거래소 레이트리밋, 최소주문금액, 슬리피지·스프레드, API 특성
- **리스크 거버넌스**  
  - 거래당 위험(R), 일간 드로우다운 컷, 추격매수 차단, 스프레드·슬리피지 상한
- **운영/모니터링 체계**  
  - Slack 실시간 이벤트(시작/체결/경고/오류), 파일 로깅, 일일 리포트, 안전 종료 프로토콜
- **확장성**  
  - 전략/리스크/주문엔진/알림 모듈화, 파라미터화를 통한 A/B 튜닝과 리스크 조정 용이

### B) 백엔드 개발 관점
- **아키텍처**: 시장데이터 수집 → 지표계산 → 유니버스 필터 → 시그널 → 리스크/주문 → 체결 추적 → 리포팅
- **신뢰성**:  
  - 지정가→시장가 폴백, 슬리피지 상한 가드, 부족자금 재시도, 레이트리밋, 예외·재시도 정책
- **품질/유지보수성**:  
  - 타입 힌트, 단일 책임 함수, 보수적 기본값, .env 구성, 로깅/알림 일원화
- **관측 가능성**:  
  - VWAP 기반 체결 가격 산출, 포지션 메타 관리(Entry/Stop/TP 진행상태), 일일 요약

---

## 3. 시스템 구성

```
.
├─ auto_trade_upbit.py        # Upbit 자동매매
├─ crypto_alert_slack.py      # Binance 시세 경계 알림
├─ requirements.txt
├─ .env.example               # 환경 변수 샘플
└─ README.md
```

---

## 4. 자동매매 봇 (`auto_trade_upbit.py`)

### 4.1 전략 개요
- **유니버스 선정**: KRW 마켓 상위 거래대금 TOP N, 초저가 및 고스프레드 종목 제외
- **지표/ICT**: EMA20/50, RSI, MACD, Supertrend, ATR, FVG/OTE/유동성 스윕, BOS 등
- **진입(매수)**  
  - 모멘텀: MACD 골든 or EMA20 상향 재돌파  
  - 레짐: 상위 타임프레임(15m) 상승(EMA20>EMA50 or Supertrend)  
  - ICT 컨텍스트: Bullish FVG/OTE/Discount/유동성 스윕 조합  
  - 추격매수 차단: 단기 급등 차단, 스프레드 상한
- **사이징**:  
  - 거래당 위험(R) = max(고정 손절 %, ATR×배수)  
  - 금액 = 계좌×RISK_PER_TRADE×(Entry/초기R), 최소/최대 주문 한도 적용
- **청산(매도)**  
  - 손절: 고정 손절%, ATR 손절  
  - 익절: 1R/2R 분할 → BE 이동(수수료 버퍼 포함) → 트레일링 스톱  
  - 추세/모멘텀: Supertrend 반전, MACD 데드, ICT 베어리시

### 4.2 주문/체결 신뢰성
- 지정가 우선 → 일정 시간 또는 급격 변동 시 **시장가 폴백**
- **슬리피지 상한** 초과 시 주문 취소
- 체결가 산정은 VWAP(부분 체결 고려)

### 4.3 리스크/운영 가드
- 일간 드로우다운 컷(예: -5%) → 자동 중지
- 손실 직후 쿨다운(재진입 제한), 매수 쿨다운
- 레이트리밋(초당 호출수), 예외·재시도 정책, 안전 종료(`stop_command.txt`)

### 4.4 관측성
- Slack: 시작/체결/TP/스톱/슬리피지 경고/오류/일일 요약
- 파일 로그: `auto_trading.log`
- 포지션 메타: entry/init_dist/stop/TP 단계/고점 추적

---

## 5. 시세 알림 (`crypto_alert_slack.py`)

- Binance 가격이 상단/하단 경계를 돌파/이탈 시 Slack 알림
- 평일 특정 시간대 정시 요약(기본 08–18시, .env로 변경 가능)
- 동일 조건 중복 알림 쿨다운, TradingView 링크 포함, 파일 로그 기록

---

## 6. 설치

### 6.1 요구사항
- Python 3.10+
- macOS/Linux 기준 개발 및 운영(Windows도 동작 가능)
- Upbit API 키(자동매매), Slack Webhook URL(알림), Binance 공개 엔드포인트(시세 알림)

### 6.2 의존성
```bash
pip install -r requirements.txt
```

`requirements.txt`
```
numpy
pandas
requests
pyupbit
python-dotenv
```

---

## 7. 환경 변수(.env)

`.env.example`를 참고해 `.env` 생성:

```env
# 공통
ENV_PATH=./.env

# Upbit 자동매매
ACCESS_KEY=your_upbit_access_key
SECRET_KEY=your_upbit_secret_key
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# (옵션) 시세 알림
SLACK_WEBHOOK_URL_FOR_ALERT=https://hooks.slack.com/services/...
SUMMARY_START_HOUR=8
SUMMARY_END_HOUR=18
```

---

## 8. 실행

### 8.1 자동매매
```bash
python auto_trade_upbit.py
```

- 시작 시 포트폴리오 스냅샷과 일일 기준 에쿼티를 Slack으로 전송
- 안전 종료:
  ```bash
  echo "stop trading" > stop_command.txt
  ```

### 8.2 시세 알림
```bash
python crypto_alert_slack.py
```

- 정시 요약 및 경계 돌파/이탈 알림 전송

---

## 9. 주요 파라미터 요약(자동매매)

- 실행/거래 한도  
  - `INTERVAL=10`(루프 주기), `REQUEST_LIMIT_PER_SECOND=5`, `MINIMUM_ORDER_KRW=5000`
- 유니버스/포지션  
  - `TOP_VOLUME_POOL=100`, `SPREAD_MAX_PCT=0.30`, `MAX_CONCURRENT_TRADES=10`
  - `PORTFOLIO_BUY_RATIO=0.30`(포트폴리오 내 신규 진입 규모 상한)
- 리스크/손익  
  - `RISK_PER_TRADE=0.005`(계좌 0.5%)  
  - `FIXED_STOP_LOSS=-3.0`, `ATR_SL_MULT=1.5`, `TRAILING_STOP_PCT=2.5`  
  - `TP1_R=1.0`, `TP2_R=2.0`, `BREAK_EVEN_BUFFER_PCT=0.15`, `DAILY_MAX_DRAWDOWN_PCT=5.0`
- 체결/슬리피지  
  - `LIMIT_OFFSET_BUY=-0.0005`, `LIMIT_OFFSET_SELL=+0.0010`  
  - `LIMIT_TIMEOUT_SEC=15`, `FAST_MOVE_PCT=0.8`, `MAX_SLIPPAGE_PCT=0.9`
- 행동 억제  
  - `COOLDOWN_PERIOD_BUY=60`, `COOLDOWN_AFTER_LOSS_SEC=900`  
  - `CHASE_UP_PCT_BLOCK=15.0`, `RSI_MAX_ENTRY=90`, `MIN_PRICE_KRW=20`

> 운영 환경과 리스크 성향에 따라 조정 가능합니다. 기본값은 보수적으로 설정했습니다.

---

## 10. 품질/테스트 전략

- 지표/시그널 유닛 테스트(EMA/RSI/MACD/ATR/Supertrend/FVG 등 알고리즘 검증)
- 주문엔진 시뮬레이션(슬리피지·시간초과·폴백 경로)
- KPI 기반 회귀 체크(슬리피지 초과율, 중복 알림률, DD 컷 동작)
- 백테스트/리플레이는 별도 프레임워크 연동(로드맵 항목) 전제로 설계

---

## 11. 보안 및 책임 한계

- API 키는 `.env`로 관리, 리포지토리에 커밋하지 않음
- 실제 거래는 금전적 손실 가능성이 존재하며 모든 책임은 사용자에게 있습니다
- 본 코드는 연구·교육 목적의 참고 구현입니다

---

## 12. 로드맵

- 백테스트/리플레이 모듈 연동(PyFolio/벡테스트 프레임워크)
- 전략 플러그인 구조(팩토리/인터페이스) 도입
- Slack 명령형 컨트롤(`/pause`, `/resume`, `/risk 0.3` 등)
- 메트릭 수집(Prometheus/Grafana) 및 알림 정책 고도화
- 체결/슬리피지 분석 리포트 자동화

