"""Turning-point fitness of a -1/0/+1 signal series (Lin, Yang & Song 2011, eqs. 1-7), with the gaps of the paper filled in.

Signal convention as in the rest of the project: -1 = buy, +1 = sell, 0 = nothing.

For every turning point T_i that has a turning point on both sides (the first and last points only bound their neighbours):
  zone Z_i      [ceil((T_{i-1} + T_i) / 2), ceil((T_i + T_{i+1}) / 2)). Zones do not overlap, so every signal belongs to exactly
                one turning point (the paper's window (T_{i-1}, T_{i+1}) counts a signal for two points).
  amplitude A_i bottom: max(close[T_{i-1}+1 .. T_{i+1}-1]) - close(T_i); top: close(T_i) - min(...)    (paper eqs. 3 and 6)
  accuracy      the right-direction signal in Z_i closest in price: min |close(S) - close(T_i)|          (paper eqs. 1 and 4)
                (buy at a bottom, sell at a top). No right-direction signal: A_i instead (missed chance, eqs. 3 and 6).
  wrong         the worst wrong-direction signal in Z_i, counted once:
                  sell in a bottom zone: 2 * (max(close[...]) - close(S))
                  buy in a top zone:     2 * (close(S) - min(close[...]))
                Equal to the paper's 2 * A_i (eqs. 2 and 5) for a signal at the turning point itself, smaller the further the price
                has already moved towards the next turning point. The paper's 5% cut-off (no penalty beyond it) is dropped.
  term_i        (accuracy or A_i) + wrong, divided by close(T_i) so that every turning point weighs the same whatever the price.
fitness = sum of term_i (smaller is better); baseline = sum of A_i / close(T_i), the fitness of a series with no signal at all;
normalized = fitness / baseline: 0 = every turning point hit exactly, 1 = as good as doing nothing, above 1 = worse than nothing.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def turning_point_fitness(close: np.ndarray, signal: np.ndarray, points: Sequence[dict]) -> dict:
    """Fitness of `signal` against the turning points `points` (dicts with "index" and "type" in {"bottom", "top"}).

    `close` and `signal` are full-length arrays and the point indices refer to them. Points must alternate bottom/top and be
    sorted, as detect_turning_points returns them.
    """
    close = np.asarray(close, dtype=float)
    signal = np.nan_to_num(np.asarray(signal, dtype=float), nan=0.0)
    idx = [int(p["index"]) for p in points]
    kinds = [str(p["type"]) for p in points]

    total = baseline = 0.0
    hits = wrong_points = evaluated = signals = 0
    for i in range(1, len(idx) - 1):
        prev, cur, nxt = idx[i - 1], idx[i], idx[i + 1]
        if not prev < cur < nxt:
            continue
        lo, hi = (prev + cur + 1) // 2, (cur + nxt + 1) // 2
        inner = close[prev + 1:nxt]
        price = close[cur]
        if inner.size == 0 or not np.isfinite(price) or price <= 0:
            continue
        zone_close, zone_signal = close[lo:hi], signal[lo:hi]
        if kinds[i] == "bottom":
            extreme = float(np.nanmax(inner))
            amplitude = extreme - price
            right, wrong = zone_signal == -1, zone_signal == 1
            wrong_penalty = 2.0 * (extreme - zone_close[wrong]) if wrong.any() else np.array([])
        else:
            extreme = float(np.nanmin(inner))
            amplitude = price - extreme
            right, wrong = zone_signal == 1, zone_signal == -1
            wrong_penalty = 2.0 * (zone_close[wrong] - extreme) if wrong.any() else np.array([])

        accuracy = float(np.min(np.abs(zone_close[right] - price))) if right.any() else amplitude
        worst_wrong = float(max(np.max(wrong_penalty), 0.0)) if wrong_penalty.size else 0.0
        total += (accuracy + worst_wrong) / price
        baseline += amplitude / price
        evaluated += 1
        hits += int(right.any())
        wrong_points += int(worst_wrong > 0)
        signals += int(right.sum() + wrong.sum())

    return {
        "fitness": total,
        "baseline": baseline,
        "normalized": total / baseline if baseline > 0 else float("nan"),
        "points": evaluated,
        "hit_rate": hits / evaluated if evaluated else float("nan"),
        "wrong_rate": wrong_points / evaluated if evaluated else float("nan"),
        "signals": signals,
    }
