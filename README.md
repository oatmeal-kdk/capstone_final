# capstone2-legacy-v1

Phase 12 v1 레시피(ta 지표 20종 → ATR 트리플배리어 → ESN → 변동성 타깃 오버레이)와 그 재검토 실험을 capstone2 저장소에서 떼어 따로 관리하는 저장소입니다. 레시피와 실험 내용은 [legacy_v1/README.md](legacy_v1/README.md)에 있습니다.

## 폴더

| 경로 | 내용 | git |
|---|---|---|
| `legacy_v1/` | 파이프라인 패키지 (data, features, labels, model, backtest, pipeline, experiments) | 포함 |
| `legacy_v1/universe/` | pooled 실험의 시장별 종목 목록 (capstone2 `pipeline_v2` 패널의 열 순서 그대로) | 포함 |
| `data/universe/` | 스크리닝 결과 종목 목록 (`production_config.py`가 읽음) | 포함 |
| `data/processed/`, `data/labels/cpm/` | 가격 파일과 CPM 전환점 (약 1GB). 2026-11-03에 capstone2의 `data/`에서 복사 | 제외 |
| `results/` | 실험 결과 | 제외 |

## 준비

```bash
python -m pip install -r requirements.txt
```

`data/processed`와 `data/labels/cpm`은 git에 들어 있지 않습니다. 새로 받은 경우 capstone2에서 같은 경로로 복사합니다. capstone2 쪽 데이터가 이후에 갱신되어도 이 저장소에는 반영되지 않습니다.

가격 파일(`<종목>_with_tp.csv`)에 대해:
- `screen_universe.py`는 파일이 없는 종목을 yfinance로 받아 OHLCV만 있는 파일을 만듭니다. 확정 레시피(`triple_barrier`)와 `cpm_causal`은 이것으로 충분합니다.
- `cpm_point_type` 열은 capstone2의 `prepare_ticker_data.py`가 만든 것이고 이 저장소에서는 만들지 않습니다. 이 열을 쓰는 것은 `--target-method cpm`(미래 정보 포함, 사용 금지)뿐입니다. CPM 전환점은 `experiments/mark_cpm.py`가 `data/labels/cpm/`에 따로 만듭니다.

## 실행

모든 명령은 이 저장소 루트에서 실행합니다 (코드의 `data/...`, `results/...` 경로가 현재 폴더 기준입니다).

```bash
# 한 종목 walk-forward (확정 레시피)
python -m legacy_v1.pipeline.run_walkforward --tp-path data/processed/SPY/1d/SPY_with_tp.csv \
    --target-method triple_barrier --train-years 5 --fee-rate 0.0005 --slippage-rate 0.0005 \
    --output-dir results/legacy_v1/SPY

# 라벨 비교 (한 종목씩) / 여러 종목을 묶은 실험
python -m legacy_v1.experiments.compare_labels --per-market 30 --jobs 6
python -m legacy_v1.experiments.pooled --markets US KOSPI KOSDAQ --jobs 6
```
