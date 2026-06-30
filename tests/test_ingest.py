"""Goal 1 tests: pure parse_record, streaming parquet round-trip, candidate_ids, integrity."""
import datetime as dt
import json

import numpy as np
import pytest

from src import ingest
from src.common import candidate_id_to_int
from tests import fixtures


def _approx_equal(a, b, tol=1e-4):
    """Recursive compare with float32 tolerance; dicts/lists deep-compared."""
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a == b
        return abs(float(a) - float(b)) <= tol
    if isinstance(a, dict):
        return set(a) == set(b) and all(_approx_equal(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_approx_equal(x, y, tol) for x, y in zip(a, b))
    return a == b


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_parse_record_normalizes_absent_arrays():
    raw = fixtures.base_candidate("CAND_0000123")
    del raw["certifications"]  # absent optional array
    raw["languages"] = []
    rec = ingest.parse_record(raw)
    assert rec["certifications"] == []      # absent -> []
    assert rec["languages"] == []
    assert isinstance(rec["career_history"][0]["start_date"], dt.date)
    assert rec["redrob_signals"]["skill_assessment_scores"] == {}


def test_parse_record_rejects_malformed():
    with pytest.raises(ValueError):
        ingest.parse_record({"candidate_id": "BADID"})
    bad = fixtures.base_candidate("CAND_0000001")
    bad["career_history"] = []
    with pytest.raises(ValueError):
        ingest.parse_record(bad)


def test_roundtrip_is_lossless(tmp_path):
    recs = list(fixtures.all_fixtures().values())
    src = tmp_path / "c.jsonl"
    _write_jsonl(src, recs)
    out = tmp_path / "art"
    manifest = ingest.ingest_file(src, out, expected_n=len(recs))

    assert manifest["rows_written"] == len(recs)
    assert manifest["quarantined"] == 0

    reloaded = ingest.read_parquet_records(out / "candidates.parquet")
    assert len(reloaded) == len(recs)
    for original, back in zip(recs, reloaded):
        expected = ingest.parse_record(original)
        assert back["candidate_id"] == expected["candidate_id"]
        assert _approx_equal(back, expected), f"round-trip mismatch for {back['candidate_id']}"


def test_candidate_ids_npy_aligns_with_source_order(tmp_path):
    recs = list(fixtures.all_fixtures().values())
    src = tmp_path / "c.jsonl"
    _write_jsonl(src, recs)
    out = tmp_path / "art"
    ingest.ingest_file(src, out, expected_n=len(recs))

    ids = np.load(out / "candidate_ids.npy")
    assert ids.dtype == np.int32
    assert ids.tolist() == [candidate_id_to_int(r["candidate_id"]) for r in recs]


def test_quarantine_not_drop_not_crash(tmp_path):
    recs = list(fixtures.all_fixtures().values())
    src = tmp_path / "c.jsonl"
    with open(src, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(recs[0]) + "\n")
        f.write("{ this is not valid json }\n")          # malformed
        f.write(json.dumps(recs[1]) + "\n")
        f.write(json.dumps(recs[0]) + "\n")               # duplicate id
    out = tmp_path / "art"
    manifest = ingest.ingest_file(src, out)  # no expected_n on a dirty sample
    assert manifest["rows_written"] == 2
    assert manifest["quarantined"] == 2
    assert (out / "quarantine.jsonl").exists()


def test_expected_n_mismatch_aborts(tmp_path):
    recs = list(fixtures.all_fixtures().values())
    src = tmp_path / "c.jsonl"
    _write_jsonl(src, recs)
    with pytest.raises(AssertionError):
        ingest.ingest_file(src, tmp_path / "art", expected_n=len(recs) + 1)


def test_manifest_has_provenance_fields(tmp_path):
    recs = list(fixtures.all_fixtures().values())
    src = tmp_path / "c.jsonl"
    _write_jsonl(src, recs)
    m = ingest.ingest_file(src, tmp_path / "art", expected_n=len(recs))
    for k in ("source_sha256", "parquet_sha256", "candidate_ids_sha256",
              "schema_fingerprint", "lib_versions", "rows_written"):
        assert k in m and m[k]
