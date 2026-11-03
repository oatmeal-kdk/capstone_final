# legacy_v1: the triangle-target functions of src/esn/dataset.py (copied unchanged).
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_triangle_target(
    df: pd.DataFrame,
    turning_points: list[dict[str, Any]] | pd.DataFrame,
) -> pd.Series:
    """Build a CPM turning-point triangle-wave teacher target in [-1, 1].

    Bottom/buy points are mapped to -1, top/sell points are mapped to 1, and
    values between adjacent turning points are linearly interpolated.
    This target is an offline supervised-learning label derived from CPM; it is
    not directly observable at a live trading decision point.
    """

    points = _normalize_turning_points(df, turning_points)
    if len(points) < 2:
        raise ValueError(
            "At least two CPM turning points are required to build triangle_target."
        )

    target = pd.Series(index=df.index, dtype=float, name="triangle_target")
    first_pos, first_value = points[0]
    last_pos, last_value = points[-1]
    target.iloc[: first_pos + 1] = first_value
    target.iloc[last_pos:] = last_value

    for (start_pos, start_value), (end_pos, end_value) in zip(points, points[1:]):
        if end_pos <= start_pos:
            continue
        target.iloc[start_pos : end_pos + 1] = np.linspace(
            start_value,
            end_value,
            end_pos - start_pos + 1,
        )
    return target.clip(-1.0, 1.0)


def _normalize_turning_points(
    df: pd.DataFrame,
    turning_points: list[dict[str, Any]] | pd.DataFrame,
) -> list[tuple[int, float]]:
    records = (
        turning_points.to_dict("records")
        if isinstance(turning_points, pd.DataFrame)
        else list(turning_points)
    )
    normalized: list[tuple[int, float]] = []
    for point in records:
        pos = _point_position(df, point)
        value = _point_value(point)
        if 0 <= pos < len(df):
            normalized.append((pos, value))
    normalized = sorted(set(normalized), key=lambda item: item[0])
    if len(normalized) < 2:
        raise ValueError(
            "Fewer than two valid turning points fall inside the ESN DataFrame index."
        )
    return normalized


def _point_position(df: pd.DataFrame, point: dict[str, Any]) -> int:
    if "index" in point:
        index_value = point["index"]
        if index_value in df.index:
            return int(df.index.get_loc(index_value))
        return int(index_value)
    for key in ["date", "Date", "datetime", "Datetime"]:
        if key in point:
            date_col = _date_column(df)
            if date_col is None:
                raise ValueError(f"Turning point has {key}, but df has no date column.")
            matches = np.flatnonzero(df[date_col].astype(str).to_numpy() == str(point[key]))
            if len(matches) == 0:
                raise ValueError(f"Turning point date not found in df[{date_col}]: {point[key]}")
            return int(matches[0])
    raise ValueError(f"Turning point missing index/date information: {point}")


def _point_value(point: dict[str, Any]) -> float:
    raw_type = str(point.get("type", point.get("point_type", point.get("signal", "")))).lower()
    if raw_type in {"bottom", "buy"}:
        return -1.0
    if raw_type in {"top", "sell"}:
        return 1.0
    raise ValueError(
        "Turning point type must be bottom/top or buy/sell; "
        f"got {point.get('type', point)}."
    )


def _date_column(df: pd.DataFrame) -> str | None:
    for col in ["Date", "date", "Datetime", "datetime"]:
        if col in df.columns:
            return col
    return None
