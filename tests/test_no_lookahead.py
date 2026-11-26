"""Regression checks: inputs and labels use no data beyond what their definition allows.

These pin the properties the README states (section 3, "미래 정보"); they do not check result numbers.
"""

import numpy as np

from legacy_v1.features.ta_features import TA_FEATURE_COLUMNS, build_ta_features
from legacy_v1.labels.cpm import detect_turning_points
from legacy_v1.labels.triple_barrier import label_triple_barrier
from legacy_v1.pipeline.run_walkforward import _full_horizon_mask, _known_triangle


def test_ta_features_do_not_change_when_future_rows_are_added(ohlcv):
    cut = 300
    full = build_ta_features(ohlcv)[TA_FEATURE_COLUMNS].to_numpy(dtype=float)
    prefix = build_ta_features(ohlcv.iloc[:cut].copy())[TA_FEATURE_COLUMNS].to_numpy(dtype=float)
    np.testing.assert_allclose(prefix, full[:cut], rtol=1e-9, atol=1e-9, equal_nan=True)


def test_triple_barrier_label_sees_at_most_max_holding_future_rows(ohlcv):
    max_holding = 10
    full = label_triple_barrier(ohlcv, k=2.0, max_holding=max_holding).to_numpy()
    for t in range(30, len(ohlcv) - max_holding, 37):
        cut = ohlcv.iloc[: t + max_holding + 1].copy()     # rows up to t + max_holding only
        assert np.array_equal(label_triple_barrier(cut, k=2.0, max_holding=max_holding).to_numpy()[t], full[t], equal_nan=True)


def test_triple_barrier_rows_without_a_full_horizon_are_dropped():
    mask = _full_horizon_mask(50, 10)
    assert mask[:40].all() and not mask[40:].any()


def test_cpm_causal_target_is_unknown_from_the_last_reported_turning_point_on(ohlcv):
    close = ohlcv["Close"]
    p_value, t_value = 0.05, 4
    points = detect_turning_points(close.reset_index(drop=True), P=p_value, T=t_value)
    assert len(points) >= 4, "the random walk should give several turning points"
    target = _known_triangle(close, p_value, t_value)
    last_final = int(points[-2]["index"])        # the last reported point can still move, so the target stops at the one before
    assert np.isnan(target[last_final + 1:]).all()
    assert np.isnan(target[: int(points[0]["index"])]).all()
    known = target[~np.isnan(target)]
    assert known.size > 0 and np.all(np.abs(known) <= 1.0 + 1e-12)
