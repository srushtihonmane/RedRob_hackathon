"""Streaming/resumable builder parity: build_representation_streaming must produce the same
artifacts as the in-memory build_representation (the full-pool path used by precompute.py)."""
import datetime as dt
import json
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.path.exists("data/sample/artifacts/repr_manifest.json"),
                                reason="model/sample artifacts not available")

from src import build, ingest   # noqa: E402

REF = dt.date(2026, 5, 25)


def _build_both(tmp_path, n=15):
    src = tmp_path / "p.jsonl"
    with open("data/sample/sample.jsonl", encoding="utf-8") as f, open(src, "w", encoding="utf-8", newline="\n") as g:
        for i, line in enumerate(f):
            if i >= n:
                break
            g.write(line)
    mem, strm = tmp_path / "mem", tmp_path / "strm"
    ingest.ingest_file(src, mem, expected_n=n)
    ingest.ingest_file(src, strm, expected_n=n)
    build.build_representation(str(mem / "candidates.parquet"), str(mem / "candidate_ids.npy"),
                               str(mem), ref_date=REF)
    build.build_representation_streaming(str(strm / "candidates.parquet"),
                                         str(strm / "candidate_ids.npy"), str(strm), ref_date=REF,
                                         chunk=4)   # tiny chunk forces multiple flushes/checkpoints
    return mem, strm


def test_streaming_matches_in_memory(tmp_path):
    mem, strm = _build_both(tmp_path)
    fm = np.load(mem / "features.npy"); fs = np.load(strm / "features.npy")
    assert fm.shape == fs.shape
    assert np.allclose(np.nan_to_num(fm), np.nan_to_num(fs), atol=1e-3)
    em = np.load(mem / "embeddings_evidence.npy"); es = np.load(strm / "embeddings_evidence.npy")
    assert np.allclose(em, es, atol=1e-3)
    # same feature manifest column order + ref_date
    cm = json.load(open(mem / "feature_manifest.json")); cs = json.load(open(strm / "feature_manifest.json"))
    assert [c["name"] for c in cm["columns"]] == [c["name"] for c in cs["columns"]]
    assert cs["ref_date"] == REF.isoformat()
    # checkpoint file cleaned up on success
    assert not (strm / "_build_ckpt.json").exists()
