"""Goal 3 builder/Sandbox-parity + REF_DATE tests (require the model; embeds a few docs)."""
import datetime as dt
import math
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.path.exists("data/sample/artifacts/repr_manifest.json"),
    reason="model/sample artifacts not available")

from src import build, represent          # noqa: E402
from src.common import read_json           # noqa: E402
from src.ingest import parse_record        # noqa: E402
from tests import fixtures                 # noqa: E402


def test_sandbox_parity_builder_matches_batch():
    """Per-candidate builder must produce identical raw features to the batch driver
    given frozen inputs (contracts.md C11)."""
    fz = represent.FrozenInputs.load(dt.date(2024, 6, 1))
    recs = [parse_record(fixtures.plain_language_tier5()),
            parse_record(fixtures.canonical_stuffer())]
    names, n, _ = build.build_features_and_embeddings(recs, fz, "data/sample/_parity_test")
    batch = np.load("data/sample/_parity_test/features.npy")
    for i, rec in enumerate(recs):
        b = build.builder(rec, fz)
        single = np.array([b["features"].values[k] for k in names], dtype=np.float32)
        # nan-aware equality
        both_nan = np.isnan(batch[i]) & np.isnan(single)
        assert np.all(both_nan | (batch[i] == single)), f"parity mismatch row {i}"


def test_ref_date_frozen_in_manifest():
    rm = read_json("data/sample/artifacts/repr_manifest.json")
    fm = read_json("data/sample/artifacts/feature_manifest.json")
    assert rm["ref_date"] == fm["ref_date"]
    # REF_DATE == max(last_active in sample) + 1 day
    from src.ingest import read_parquet_records
    recs = read_parquet_records("data/sample/artifacts/candidates.parquet")
    mx = max(r["redrob_signals"]["last_active_date"] for r in recs)
    expected = (mx + dt.timedelta(days=1)).isoformat()
    assert rm["ref_date"] == expected
