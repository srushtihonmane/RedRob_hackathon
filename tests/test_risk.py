"""Goal 5 risk tests: honeypot H1/H2/H3 (OR), consulting gate + override, invariants."""
import datetime as dt
import os

import numpy as np
import pytest

from src import represent, risk
from src.ingest import parse_record
from tests import fixtures

REF = dt.date(2024, 6, 1)


@pytest.fixture(scope="module")
def fz():
    return represent.FrozenInputs.load(REF)


def _flags(rec, fz):
    return risk.risk_flags(represent.derive_structured(parse_record(rec), fz).values)


def test_honeypot_rules_or_fired(fz):
    assert _flags(fixtures.honeypot_h1(), fz)["honeypot_flag"]
    assert _flags(fixtures.honeypot_h2(), fz)["honeypot_flag"]
    assert _flags(fixtures.honeypot_h3(), fz)["honeypot_flag"]
    # a clean candidate is not flagged
    assert not _flags(fixtures.plain_language_tier5(), fz)["honeypot_flag"]
    # per-rule evidence recorded
    assert _flags(fixtures.honeypot_h1(), fz)["honeypot_reasons"][0][0] == "H1"


def test_consulting_gate_and_override(fz):
    only = _flags(fixtures.consulting_only(), fz)
    assert only["consulting_gate_flag"] and not only["consulting_gate_suppressed"]
    wp = _flags(fixtures.consulting_with_product(), fz)
    assert wp["consulting_gate_suppressed"] and not wp["consulting_gate_flag"]


@pytest.mark.skipif(not os.path.exists("data/sample/artifacts/features.npy"),
                    reason="sample artifacts not built")
def test_invariants_on_sample():
    from src import scoring
    b = scoring.load_bundle("data/sample/artifacts")
    r = risk.compute_risk(b)
    dy = b.col("yoe_minus_summed_tenure")
    # rule => condition
    assert np.all(dy[r["h1"]] > 2.0)
    assert np.all(dy[r["h2"]] < -2.0)
    assert np.all(b.col("expert_skill_zero_duration_count")[r["h3"]] >= 2)
    # near-empty buffer band (0.5, 2.0] — a wide dead zone makes the threshold robust.
    # (Exactly-empty is a FULL-POOL audit property, Goal 5 D6; the stratified sample may
    # hold a stray candidate, so assert near-empty rather than exactly zero.)
    band = (np.abs(dy) > 0.5) & (np.abs(dy) <= 2.0)
    assert band.mean() < 0.01, "buffer band should be near-empty (robust threshold)"
    # honeypot == OR of the three
    assert np.array_equal(r["honeypot_flag"], r["h1"] | r["h2"] | r["h3"])


@pytest.mark.skipif(not os.path.exists("data/sample/artifacts/features.npy"),
                    reason="sample artifacts not built")
def test_risk_artifact_written(tmp_path):
    from src import scoring
    b = scoring.load_bundle("data/sample/artifacts")
    summary = risk.write_risk_artifact(b, tmp_path)
    assert summary["n_rows"] == b.feats.shape[0]
    arr = np.load(tmp_path / "risk_flags.npy")
    assert arr.shape == (b.feats.shape[0], 6)
