"""transforms.py — Goal 4 D4/D5 feature transforms & calibration (pure functions).

Standardized so HIGHER = BETTER FIT. Fit on the FULL-POOL frozen percentiles
(normalization_stats.json); never on a shortlist. Missing -> neutral, never negative
(the caller masks via the _present flags — these helpers operate on present values).

  * quantile_norm  — distribution-free rank-norm for skewed continuous features (saturate
                     beyond p1/p99). Robust to planted outliers; no tuning. (D4)
  * tail_anchor    — relevance calibration: pin the irrelevant BULK (~p_lo) -> 0, reward the
                     upper tail (~p_hi) -> 1. Preserves magnitude AND additive-positive. (D5)
  * trapezoid      — non-monotonic band membership (YOE, applied-ML years) -> [0,1]. (D6)
"""
from __future__ import annotations

import math


def _interp_percentiles(stats: dict) -> tuple[list[float], list[float]]:
    grid = [1, 5, 25, 50, 75, 95, 99]
    xs, ys = [], []
    for p in grid:
        key = f"p{p}"
        if key in stats:
            xs.append(stats[key]); ys.append(p / 100.0)
    return xs, ys


def quantile_norm(value: float, stats: dict, invert: bool = False) -> float:
    """Map ``value`` to [0,1] by piecewise-linear interpolation over frozen percentiles,
    saturating beyond p1/p99. ``invert`` for lower-is-better features."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    xs, ys = _interp_percentiles(stats)
    if len(xs) < 2:
        return 0.5
    if value <= xs[0]:
        q = ys[0]
    elif value >= xs[-1]:
        q = ys[-1]
    else:
        q = ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= value <= xs[i + 1]:
                span = xs[i + 1] - xs[i]
                frac = 0.0 if span <= 0 else (value - xs[i]) / span
                q = ys[i] + frac * (ys[i + 1] - ys[i])
                break
    return 1.0 - q if invert else q


def tail_anchor(value: float, stats: dict, lo_key: str = "p95", hi_key: str = "p99") -> float:
    """Relevance calibration: <= lo -> 0, >= hi -> 1, linear between. Pins the irrelevant
    bulk to 0 (NOT 0.5 like quantile-norm) and rewards only the upper tail (D5)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    lo, hi = stats.get(lo_key), stats.get(hi_key)
    if lo is None or hi is None or hi <= lo:
        return 0.0
    if value <= lo:
        return 0.0
    if value >= hi:
        return 1.0
    return (value - lo) / (hi - lo)


def trapezoid(value: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal membership: 0 below a, ramps to 1 over [a,b], 1 over [b,c], ramps to 0
    over [c,d], 0 above d. (a<=b<=c<=d). For soft bands like YOE 5-9 / ideal 6-8."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    if value <= a or value >= d:
        return 0.0
    if b <= value <= c:
        return 1.0
    if a < value < b:
        return (value - a) / (b - a) if b > a else 1.0
    return (d - value) / (d - c) if d > c else 1.0


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x
