# legacy_v1: 예전 확정 파이프라인 (Phase 12 레시피) 독립 사본

예전에 결과가 좋았던 방식(ta 지표 → 트리플배리어 → ESN → 리스크 오버레이)을 다시 분석하고 살리기 위해, 그 코드만 `src/`에서 떼어 따로 관리하는 패키지입니다.

- `src/`에는 이후 실험(pipeline_v2, 라벨링, 통합 백테스트, 대시보드)이 계속 쌓이므로, 이 버전은 거기에 영향받지 않게 분리했습니다.
- `pipeline_final/`은 열람용 사본이라 실행이 안 됐지만, 이 폴더는 **그대로 실행됩니다.**
- 2026-11-03부터 capstone2와 별도인 저장소 `capstone2-legacy-v1`에서 관리합니다. pooled 실험의 종목 목록은 `pipeline_v2` 대신 `universe/pooled_<시장>.txt`에서 읽습니다.
- 만든 날: 2026-10-15. 원본은 그날의 작업 사본(`src/` 기준, 커밋되지 않은 수정 포함)입니다.

## 동일성 확인

SPY에 확정 레시피를 원본(`src.pipeline.run_esn_ta_walkforward`)과 이 패키지(`legacy_v1.pipeline.run_walkforward`)로 각각 돌려, 예측 파일과 자산곡선 파일이 **바이트 단위로 같음**을 확인했습니다.

| SPY (2016-03~2026-09, 비용 각 0.05%) | Sharpe | 연수익 | 최대 낙폭 |
|---|---:|---:|---:|
| 레시피 (오버레이 없음) | 0.70 | 9.8% | −22.7% |
| 레시피 + 리스크 오버레이 (목표 변동성 10%) | 0.90 | 7.1% | −11.5% |

## 구성

| 폴더 | 파일 | 원본 |
|---|---|---|
| `data/` | `download.py`, `preprocess.py` | `src/data/` |
| `features/` | `ta_features.py` (ta 지표 20종), `ohlcv_features.py` | `src/indicators/` |
| `labels/` | `triple_barrier.py` (확정 라벨), `cpm_target.py` (CPM 분기용 함수 2개) | `src/cpm/triple_barrier.py`, `src/esn/dataset.py` |
| `model/` | `esn.py`, `metrics.py`, `scaling.py`, `thresholds.py` | `src/esn/`, `src/utils/scaling.py`, `src/pipeline/run_esn.py` |
| `backtest/` | `simple_backtest.py` (리스크 오버레이 포함) | `src/backtest/` |
| `pipeline/` | `production_config.py` (확정값), `run_walkforward.py` (실행), `screen_universe.py` | `src/pipeline/` |
| `relative/` | `relative_series.py` (상대강도 트랙) | `src/indicators/` |
| `utils/` | `io.py` | `src/utils/io.py` |

바뀐 것은 import 경로뿐입니다. 각 파일 첫 줄에 원본 경로를 적어 두었습니다.

## 확정 레시피 (`pipeline/production_config.py`)

- 입력: ta 지표 20종 (모멘텀 7, 추세 6, 변동성 4, 거래량 3), MinMax 스케일링(학습 구간에만 fit)
- 라벨: ATR 기반 트리플배리어, k = 2.0, 최대 보유 10일
- 모델: ESN, 시드 10개 앙상블(서비스용), 검증 구간 정확도로 threshold 선택
- walk-forward: 학습 5년 / 검증 1년 / 1년씩 이동
- 리스크 오버레이: 목표 변동성 10%, 리밸런싱 밴드 10%
- 비용: 수수료 0.05%, 슬리피지 0.05%

## 실행

프로젝트 루트에서:

```bash
# 한 종목 walk-forward (확정 레시피)
python -m legacy_v1.pipeline.run_walkforward --tp-path data/processed/SPY/1d/SPY_with_tp.csv \
    --target-method triple_barrier --train-years 5 --fee-rate 0.0005 --slippage-rate 0.0005 \
    --output-dir results/legacy_v1/SPY

# 리스크 오버레이 포함
python -m legacy_v1.pipeline.run_walkforward ... --risk-overlay --target-vol 0.10

# 여러 종목 스크리닝
python -m legacy_v1.pipeline.screen_universe --tickers AAPL MSFT --output results/legacy_v1/screening.csv
```

## 알려진 문제 (다시 살리기 전에 확인할 것)

2026-10-15에 예전 스크리닝 결과(2,740종목)를 같은 기간 "그 종목을 그냥 보유"와 비교했습니다(`results/_old_pipeline_vs_buyhold.csv`).

- 전략이 기간의 약 77% 동안 주식을 들고 있었고, 전략 Sharpe와 보유 Sharpe의 상관이 0.8입니다.
- 미국은 중앙값 기준으로 그냥 보유한 쪽이 더 좋았습니다(Sharpe 0.47 대 0.41).
- "active" 종목(Sharpe 0.40 이상)은 결과를 보고 고른 것이라 그 성과를 그대로 믿을 수 없습니다.
- 거래 구간이 2016~2026년이라, 이후 실험에서 남겨 둔 2019년 이후 구간을 이미 포함합니다.
- 리스크 오버레이의 효과가 ESN 신호에서 나오는지 오버레이 자체에서 나오는지는 아직 가리지 않았습니다. 같은 오버레이를 "그냥 보유"에 씌운 기준선이 필요합니다.

## 2026-11 추가: 체결 방식과 CPM 라벨

**체결 방식** (`--execution`)
- `same_close` (기본값): 예전 방식. 신호를 만든 그날 종가에 체결
- `next_open`: t일 신호를 t+1일 시가에 체결. 무작위 240종목에서 Sharpe 변화는 중앙값 0.000~0.002로 거의 없었습니다(`results/legacy_v1/execution_compare.csv`). 거래가 드물기 때문입니다.

**라벨** (`--target-method`)

| 값 | 내용 | 미래 정보 |
|---|---|---|
| `triple_barrier` | 확정 레시피 라벨 (ATR 2배, 10일) | 없음 (구간 경계에서 잘라냄) |
| `cpm` | 예전 CPM 삼각파 목표. 가격 파일의 `cpm_point_type` 열을 그대로 씀 | 있음. **사용 금지** (`--allow-lookahead` 없이는 실행 거부) |
| `cpm_global` | 같은 삼각파 목표지만 P·T를 전체 기간의 파레토 무릎으로 고름 | 있음. **사용 금지** (`--allow-lookahead` 없이는 실행 거부) |
| `cpm_causal` | 구간마다 학습 구간만으로 P·T를 파레토 무릎으로 고르고, 학습·검증 목표를 그 시점까지의 가격으로 만든 CPM으로 계산. 마지막 전환점과 그 이후는 확정되지 않아 제외 | 없음 |

- **P·T는 전체 기간으로 고르면 안 됩니다.** 그러면 이후 모든 거래 연도의 가격을 보고 라벨 설정을 정한 것이 됩니다. 비교와 결과에는 `cpm_causal`만 씁니다.
- 파레토 무릎: P ∈ {3, 4, 5, 6, 7, 8, 10, 12%}, T ∈ {2, 4, …, 18}의 72개 조합에서 "전환점 수는 적게, 잡아낸 움직임은 크게"의 파레토 최전선을 만들고, 양 끝을 잇는 선에서 가장 먼 점을 고릅니다(`labels/cpm.py`의 `select_pareto_knee`).
- 삼각파 목표: 저점 −1, 고점 +1, 사이는 직선으로 이은 값. ESN이 이 연속값을 회귀하고, 점수 < −threshold면 매수입니다.
- 예전 `src`의 causal CPM 스크립트(`run_esn_ta_causal_walkforward.py`)는 학습 구간 끝 이후를 마지막 값으로 채워, 검증 구간의 목표가 상수가 되는 문제가 있었습니다. `cpm_causal`은 이를 고쳤습니다.
