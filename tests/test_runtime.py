"""Goal 8 runtime tests: two-command repro, budget, ML policy, two-phase load, determinism,
build_manifest, runtime_report. Runs rank.py as a subprocess against the sample artifacts."""
import json
import os
import subprocess
import sys

import pytest

ART = "data/sample/artifacts"
POOL = "data/sample/sample.jsonl"
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ART, "features.npy")),
                                reason="sample artifacts not built")


def _run_rank(out, extra=None):
    cmd = [sys.executable, "rank.py", "--candidates", POOL, "--out", out, "--artifacts", ART]
    return subprocess.run(cmd + (extra or []), capture_output=True, text=True)


def test_F8_2_F8_6_budget_and_determinism(tmp_path):
    a, b = str(tmp_path / "a.csv"), str(tmp_path / "b.csv")
    r1 = _run_rank(a); r2 = _run_rank(b)
    assert r1.returncode == 0 and r2.returncode == 0, r1.stderr + r2.stderr
    # byte-identical (determinism)
    assert open(a, "rb").read() == open(b, "rb").read()
    # budget from runtime_report
    rep = json.load(open(os.path.join(ART, "runtime_report.json")))
    assert rep["total_wall_s"] < 300        # <= 5 min
    assert rep["peak_rss_mb"] < 16000       # <= 16 GB


def test_F8_4_official_validator_on_rank_output(tmp_path):
    out = str(tmp_path / "s.csv")
    assert _run_rank(out).returncode == 0
    v = subprocess.run([sys.executable, "validate_submission.py", out], capture_output=True, text=True)
    assert "Submission is valid." in v.stdout


def test_F8_3_runtime_loads_no_neural_model():
    code = ("import sys; import rank; "
            "m=[x for x in sys.modules if x.split('.')[0] in "
            "('torch','onnxruntime','fastembed','transformers','sentence_transformers')]; "
            "print('NEURAL:'+repr(m))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "NEURAL:[]" in r.stdout, r.stdout + r.stderr


def test_F8_4_binding_mismatch_fails_loudly(tmp_path):
    # point rank at the sample artifacts but a DIFFERENT candidates file -> binding fails loudly
    bogus = tmp_path / "other.jsonl"
    bogus.write_text('{"candidate_id":"CAND_0000001"}\n', encoding="utf-8")
    r = subprocess.run([sys.executable, "rank.py", "--candidates", str(bogus),
                        "--out", str(tmp_path / "x.csv"), "--artifacts", ART],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "binding failed" in r.stderr


def test_F8_5_build_manifest_sole_root(tmp_path):
    _run_rank(str(tmp_path / "s.csv"))
    m = json.load(open(os.path.join(ART, "build_manifest.json")))
    for k in ("candidates_sha256", "sub_manifests", "artifact_hashes", "runtime_report",
              "submission_sha256", "thread_settings"):
        assert k in m
    for sm in ("ingest_manifest", "feature_manifest", "repr_manifest", "risk_manifest", "jd_manifest"):
        assert sm in m["sub_manifests"]


def test_F8_7_runtime_report_stages(tmp_path):
    _run_rank(str(tmp_path / "s.csv"))
    rep = json.load(open(os.path.join(ART, "runtime_report.json")))
    stages = {s["stage"] for s in rep["stages"]}
    assert {"validate_and_bind", "load_artifacts", "score_pool", "write_csv"} <= stages
    assert "thread_settings" in rep
