"""Unit tests for src/transforms.py (pure; no artifacts needed)."""
import math

from src import transforms as T

STATS = {"p1": 0.0, "p5": 1.0, "p25": 2.0, "p50": 5.0, "p75": 10.0, "p95": 20.0, "p99": 40.0}


def test_quantile_norm_monotone_and_saturating():
    assert T.quantile_norm(-5, STATS) == 0.01      # below p1 -> p1 quantile
    assert T.quantile_norm(100, STATS) == 0.99     # above p99 -> p99 quantile
    assert T.quantile_norm(5, STATS) == 0.5        # at p50
    assert T.quantile_norm(2, STATS) < T.quantile_norm(10, STATS)  # monotone


def test_quantile_norm_invert():
    hi = T.quantile_norm(40, STATS)
    inv = T.quantile_norm(40, STATS, invert=True)
    assert abs((hi + inv) - 1.0) < 1e-9


def test_quantile_norm_missing_is_nan():
    assert math.isnan(T.quantile_norm(float("nan"), STATS))


def test_tail_anchor_pins_bulk_to_zero():
    # irrelevant bulk (<= p95) -> 0, not 0.5
    assert T.tail_anchor(5, STATS) == 0.0
    assert T.tail_anchor(20, STATS) == 0.0        # at p95
    assert T.tail_anchor(40, STATS) == 1.0        # at p99
    mid = T.tail_anchor(30, STATS)
    assert 0.0 < mid < 1.0


def test_trapezoid_band():
    # YOE-like band: ramp 4->5, plateau 5..9, ramp 9->11
    assert T.trapezoid(3, 4, 5, 9, 11) == 0.0
    assert T.trapezoid(7, 4, 5, 9, 11) == 1.0
    assert T.trapezoid(4.5, 4, 5, 9, 11) == 0.5
    assert T.trapezoid(10, 4, 5, 9, 11) == 0.5
    assert T.trapezoid(12, 4, 5, 9, 11) == 0.0


def test_clamp():
    assert T.clamp(5, 0, 1) == 1
    assert T.clamp(-1, 0, 1) == 0
    assert T.clamp(0.5, 0, 1) == 0.5
