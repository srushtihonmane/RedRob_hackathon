"""Unit tests for src/common.py — determinism primitives, identity, dates, IO."""
import datetime as dt

import pytest

from src import common


def test_candidate_id_roundtrip():
    for n in (1, 42, 4989, 9_999_999):
        cid = common.int_to_candidate_id(n)
        assert common.CANDIDATE_ID_RE.match(cid)
        assert common.candidate_id_to_int(cid) == n
    assert common.int_to_candidate_id(1) == "CAND_0000001"
    assert common.candidate_id_to_int("CAND_0004989") == 4989


def test_candidate_id_validation():
    assert common.is_valid_candidate_id("CAND_0000001")
    for bad in ("CAND_1", "cand_0000001", "CAND_00000001", "X", "CAND_000000A"):
        assert not common.is_valid_candidate_id(bad)
        with pytest.raises(ValueError):
            common.candidate_id_to_int(bad)


def test_stable_hash_is_deterministic_and_unsigned():
    h1 = common.stable_hash("CAND_0000001")
    h2 = common.stable_hash("CAND_0000001")
    assert h1 == h2
    assert 0 <= h1 <= 0xFFFFFFFF
    assert common.stable_hash("a") != common.stable_hash("b")
    assert common.stable_hash(b"bytes") == common.stable_hash("bytes")
    u = common.stable_unit_hash("CAND_0000001")
    assert 0.0 <= u < 1.0


def test_sha256_helpers(tmp_path):
    assert common.sha256_bytes(b"") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    assert common.sha256_file(p) == common.sha256_bytes(b"hello world")


def test_date_math_and_ref_date():
    ref = common.ref_date_from_dates([dt.date(2024, 1, 1), dt.date(2024, 6, 30), None])
    assert ref == dt.date(2024, 7, 1)  # max + 1 day
    assert common.days_between(ref, dt.date(2024, 6, 1)) == 30.0
    assert common.days_between(ref, None) is None
    assert common.parse_date("2024-06-30") == dt.date(2024, 6, 30)
    assert common.parse_date(None) is None
    assert common.parse_date("not-a-date") is None
    with pytest.raises(ValueError):
        common.ref_date_from_dates([None])


def test_json_roundtrip_is_byte_stable(tmp_path):
    obj = {"b": 2, "a": [1, 2, 3], "nested": {"z": 1, "y": 2}}
    p = tmp_path / "o.json"
    common.write_json(p, obj)
    again = tmp_path / "o2.json"
    common.write_json(again, obj)
    assert p.read_bytes() == again.read_bytes()  # deterministic (sorted keys)
    assert common.read_json(p) == obj


def test_configure_determinism_sets_env():
    common.configure_determinism()
    import os
    assert os.environ.get("PYTHONHASHSEED") == "0"
    assert os.environ.get("OMP_NUM_THREADS") == "1"
    settings = common.blas_thread_settings()
    assert settings["OMP_NUM_THREADS"] == "1"
