# capstone

파이프라인 흐름:

```
데이터 → 입력(ta 지표) → [폴드마다: 라벨 → 스케일링 → ESN 학습 → threshold 선택 → 테스트 신호] → 백테스트 → 평가
```


## 1. 데이터

저장된 가격 파일 `data/processed/<종목>/1d/<종목>_with_tp.csv`(일봉, git 제외, capstone2에서 복사)를 씁니다. `--tp-path`로 지정합니다.

## 2. 입력

| 선택지 | 내용 | 위치 |
|---|---|---|
| **raw** | ta 지표 20개 (모멘텀 7, 추세 6, 변동성 4, 거래량 3) 값 그대로 | [ta_features.py](legacy_v1/features/ta_features.py) |
| z | 같은 지표를 종목 자신의 과거 252일로 z-score 변환, ±5에서 자름 (여러 종목 학습에서만) | [pooled.py](legacy_v1/experiments/pooled.py) |
| window_scale | 모든 지표 기간을 한꺼번에 늘리거나 줄임. **1.0** (현재 어디서도 다른 값을 쓰지 않음. CLI 옵션 없음) | `build_ta_features(window_scale=)` |

[ohlcv_features.py](legacy_v1/features/ohlcv_features.py)의 지표 8개는 모델 입력으로 쓰지 않고, 그중 20일 변동성만 리스크 오버레이에 씁니다.

### ta 지표 20개

`ta` 라이브러리로 계산한 값입니다. 기간은 `window_scale=1.0` 기준입니다. 모든 지표는 가격 수준이 아니라 비율이나 정해진 범위의 값입니다(ATR도 종가로 나눔). ESN에 넣기 전에 학습 행의 min/max로 [−1, 1]에 맞춥니다.

**모멘텀 (7)**  
평활 = 계산한 값을 며칠 단순평균해 잔떨림을 줄이는 것.

| 열 | 지표 | 기간 | 범위 | 의미 |
|---|---|---|---|---|
| `ta_rsi` | RSI | 14 | 0~100 | 최근 상승폭과 하락폭의 비율 |
| `ta_stoch_k` | 스토캐스틱 %K | 14 | 0~100 | 최근 14일 고저 범위 안에서 종가의 위치 |
| `ta_stoch_d` | 스토캐스틱 %D | 14, 평활 3 | 0~100 | %K의 3일 단순평균 |
| `ta_stochrsi_k` | 스토캐스틱 RSI %K | 14, 평활 3 | 0~1 | RSI에 스토캐스틱을 적용한 값의 3일 단순평균 |
| `ta_williams_r` | Williams %R | 14 | −100~0 | %K − 100 (%K와 같은 정보) |
| `ta_roc` | 변화율 | 12 | % | 12일 전 대비 종가 변화율 |
| `ta_ultimate_osc` | Ultimate Oscillator | 7·14·28 | 0~100 | 세 기간 매수 압력의 가중 평균 |

**추세 (6)**

| 열 | 지표 | 기간 | 범위 | 의미 |
|---|---|---|---|---|
| `ta_adx` | ADX | 14 | 0~100 | 추세의 강도 (방향 없음) |
| `ta_adx_pos` | +DI | 14 | 0~100 | 상승 방향 움직임의 강도 |
| `ta_adx_neg` | −DI | 14 | 0~100 | 하락 방향 움직임의 강도 |
| `ta_cci` | CCI | 20 | 상한 없음 | 평균 가격에서 벗어난 정도 |
| `ta_trix` | TRIX | 15 | % | 3중 지수이동평균의 변화율 |
| `ta_aroon_osc` | Aroon 오실레이터 | 25 | −100~100 | Aroon Up − Aroon Down |

**변동성 (4)**

| 열 | 지표 | 기간 | 범위 | 의미 |
|---|---|---|---|---|
| `ta_bb_pband` | 볼린저 %B | 20, 2σ | 보통 0~1 | 볼린저 밴드 안에서 종가의 위치 |
| `ta_bb_wband` | 볼린저 밴드 폭 | 20, 2σ | % | 밴드 폭 ÷ 중심선 |
| `ta_atr_pct` | ATR ÷ 종가 | 14 | % | 하루 평균 변동폭의 가격 대비 비율 |
| `ta_keltner_pband` | 켈트너 %B | 20, ATR 10 | 보통 0~1 | 켈트너 채널 안에서 종가의 위치 |

**거래량 (3)**

| 열 | 지표 | 기간 | 범위 | 의미 |
|---|---|---|---|---|
| `ta_mfi` | MFI | 14 | 0~100 | 거래량을 반영한 RSI |
| `ta_cmf` | Chaikin Money Flow | 20 | −1~1 | 종가가 하루 범위 위쪽에서 마감한 날의 거래량 비중 |
| `ta_pvo` | PVO | 12·26, 신호선 9 | % | 거래량 단기·장기 지수이동평균의 차이 비율 |

참고:
- 스토캐스틱 %K, %D, Williams %R, 볼린저 %B, 켈트너 %B는 모두 "최근 범위 안에서 종가의 위치"를 잽니다. 특히 Williams %R은 정확히 %K − 100이라, 스케일링 뒤에는 %K와 똑같은 값이 됩니다(입력 20개 중 1개는 중복).
- `window_scale`을 바꾸면 볼린저 밴드의 σ 배수도 함께 반올림되어 바뀝니다(0.5배 → 1σ, 1.5배 → 3σ).

## 3. 라벨

`--target-method`로 고릅니다.

| 선택지 | 내용 | 미래 정보 |
|---|---|---|
| **triple_barrier** | 종가 ± k×ATR(14) 배리어 중 max_holding일 안에 먼저 닿는 쪽: 위 −1(매수), 아래 +1(매도), 둘 다 아니면 0. **k = 2.0, 10일** (`--tb-atr-mult`, `--tb-max-holding`) | 없음. 학습·검증 끝 10일은 라벨을 버림 |
| cpm_causal | 폴드마다 학습 구간 가격만으로 CPM의 P·T를 파레토 무릎으로 고르고, 전환점 사이를 −1~+1 삼각파로 이은 연속값. 확정되지 않은 마지막 전환점 이후는 버림 | 없음 |
| cpm | 가격 파일의 `cpm_point_type` 열 (전체 기간으로 만든 전환점) | **있음.** `--allow-lookahead` 없이는 실행 거부 |
| cpm_global | P·T를 전체 기간으로 고른 CPM | **있음.** 위와 같음 |

[triple_barrier.py](legacy_v1/labels/triple_barrier.py)의 `label_vol_adjusted_horizon`, `label_trend_regime`, `cusum_event_mask`는 파이프라인에 연결되어 있지 않습니다.

두 라벨이 SPY 2020년에 찍히는 위치입니다. 위는 CPM 전환점(cpm_causal 방식으로 P·T 선택), 아래는 트리플배리어 라벨입니다. CPM은 1년에 전환점 10여 개만 찍지만, 트리플배리어는 매일 라벨이 하나씩 붙고 며칠 간격으로 바뀝니다.

![SPY 2020년 CPM 전환점과 트리플배리어 라벨](docs/figures/SPY_cpm_vs_tb_2020_marks.png)

`python -m legacy_v1.experiments.plot_labels --ticker SPY --years 2020 --no-targets --out-dir docs/figures`로 다시 그릴 수 있습니다. `--no-targets`를 빼면 ESN 목표값 패널이 붙은 그림이 `results/legacy_v1/figures/`에 저장됩니다.

## 4. 학습

**학습 단위**

| 선택지 | 내용 | 위치 |
|---|---|---|
| **단일 종목** | 한 종목의 시계열 전체를 ESN에 끊김 없이 넣음. 종목마다 모델 하나 | [run_walkforward.py](legacy_v1/pipeline/run_walkforward.py) |
| 여러 종목 | 종목별 "직전 20일" 창을 0 상태에서 읽어 행으로 쌓음. 학습 행은 5일 간격. 시장(US/KOSPI/KOSDAQ)마다 모델 하나 | [pooled.py](legacy_v1/experiments/pooled.py) |

**학습 창**

| 선택지 | 내용 | 지정 방법 |
|---|---|---|
| **고정 길이 창 (rolling)** | 검증 직전 `--train-years`년(5년)만. 폴드마다 창 전체가 1년씩 뒤로 이동 | `--train-window rolling` (기본값) |
| 누적 창 (expanding) | 데이터 시작부터 검증 직전까지. 시작점은 고정되고 끝만 1년씩 늘어남 (첫 폴드 5년). v1 원래 값 | `--train-window expanding` |

**구간 나누기:** 검증 1년(`--val-years`), 테스트 1년씩 이동(`--step-years`). 단일 종목은 그 종목의 데이터 시작일을, 여러 종목은 달력 연도를 기준으로 경계를 잡습니다. 검증 연도는 그 폴드의 학습에 넣지 않습니다.

**모델**

| 항목 | 선택지 |
|---|---|
| 스케일링 | **학습 행의 min/max로 [−1, 1]** ([scaling.py](legacy_v1/model/scaling.py)) |
| reservoir | **100개, spectral radius 0.9, sparsity 0.15, input scale 0.3, leak 0.3, washout 30** ([run_walkforward.py](legacy_v1/pipeline/run_walkforward.py)의 `ESN_CONFIG`) |
| readout | **ridge 회귀, alpha 200** (`[1, 입력, 상태]` → 점수) / 로지스틱 분류 ([pooled_classifier.py](legacy_v1/experiments/pooled_classifier.py)에서만) |
| 앙상블 | **1개** (실험) / 10개 (서비스 설정, [production_config.py](legacy_v1/pipeline/production_config.py)). `--ensemble-size`, `--random-state` |

## 5. threshold와 신호

| 선택지 | 내용 | 위치 |
|---|---|---|
| **검증 정확도 최대** | {0.1, …, 0.9} 중 선택. 점수 < −thr 매수, > thr 매도, 그 사이 보유 | [thresholds.py](legacy_v1/model/thresholds.py) |
| 학습 점수 상위 20% | 학습 기간의 약 20%에서 신호가 켜지도록 고정 | [pooled_classifier.py](legacy_v1/experiments/pooled_classifier.py) |

| 신호 해석 | 내용 |
|---|---|
| **v1** | 매수 신호에 사서 매도 신호까지 보유 |
| 켜진 동안 보유 | 신호가 켜진 날만 보유, 꺼지면 현금 (pooled_classifier) |

## 6. 백테스트

모든 폴드의 테스트 구간 신호를 이어 붙여 한 번에 실행합니다 ([simple_backtest.py](legacy_v1/backtest/simple_backtest.py)).

| 항목 | 선택지 | 지정 방법 |
|---|---|---|
| 체결 | **다음 날 시가** (기본값) / 신호 당일 종가 (v1 원래 값. 신호를 만든 종가에 바로 체결하므로 실제로는 불가능. v1 숫자를 재현할 때만 사용) | `--execution next_open / same_close` |
| 비중 | **전부 사거나 전부 팔기** / 리스크 오버레이: 비중 = min(1, 목표 변동성 ÷ 20일 변동성), 차이가 밴드를 넘을 때만 조정. **목표 10%, 밴드 10%p** | `--risk-overlay`, `--target-vol`, `--rebalance-band` |
| 비용 | **수수료 0.05%, 슬리피지 0.05%** (CLI 기본값은 0이라 직접 지정) | `--fee-rate`, `--slippage-rate` |

## 7. 평가

| 선택지 | 내용 | 위치 |
|---|---|---|
| 단일 종목 요약 | Sharpe, 연수익, 최대 낙폭, 같은 기간 보유 성과 | 출력 폴더의 `summary.json` |
| 스크리닝 | Sharpe ≥ 0.40 active, 0.20~0.40 weak, 그 아래 excluded | [screen_universe.py](legacy_v1/pipeline/screen_universe.py) |
| 여러 종목 비교 | 종목별 Sharpe 차이의 중앙값 + 월 단위 블록 부트스트랩 95% 구간 → 이김 / 짐 / 구분 안 됨 | [judge.py](legacy_v1/experiments/judge.py) |

## 실험별로 바꾼 단계

| 실험 | 바꾼 단계 | 나머지 |
|---|---|---|
| [compare_execution.py](legacy_v1/experiments/compare_execution.py) | 6. 체결 | v1 |
| [compare_labels.py](legacy_v1/experiments/compare_labels.py) | 3. 라벨 (triple_barrier / cpm_causal) | v1, 다음 날 시가 체결 |
| [compare_train_window.py](legacy_v1/experiments/compare_train_window.py) | 4. 학습 창 (누적 / 고정 길이) | v1, 다음 날 시가 체결 |
| [pooled.py](legacy_v1/experiments/pooled.py) | 4. 학습 단위, 2. 입력 (raw / z), 3. 라벨 | ridge, 검증 정확도 |
| [pooled_classifier.py](legacy_v1/experiments/pooled_classifier.py) | 4. readout, 5. threshold와 신호 해석 | pooled 설정 |
