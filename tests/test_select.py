"""Goal 6 selection tests: eligibility, micro-term tie-encoding, exactly-100, honeypot-free
top-100, and the OFFICIAL validate_submission.py on the produced CSV."""
import os
import subprocess
import sys

import numpy as np
import pytest

ART = "data/sample/artifacts"
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ART, "features.npy")),
                                reason="sample artifacts not built")

from src import risk, scoring, select   # noqa: E402


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    out = tmp_path_factory.mktemp("sub")
    csv = out / "submission.csv"
    sel = select.produce_submission(ART, str(csv))
    return {"sel": sel, "csv": str(csv), "out": str(out)}


def test_F6_5_exactly_100_rows(made):
    assert len(made["sel"].rows) == 100


def test_F6_1_selection_subset_of_eligible(made):
    b = scoring.load_bundle(ART)
    r = risk.compute_risk(b)
    hard = r["honeypot_flag"] | (r["consulting_gate_flag"] & ~r["consulting_gate_suppressed"])
    for row in made["sel"].rows:
        assert not hard[row["row_index"]]


def test_F6_2_printed_score_strictly_non_increasing_and_tiebreak(made):
    rows = made["sel"].rows
    scores = [r["printed_score"] for r in rows]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]                 # non-increasing
        if scores[i] == scores[i + 1]:                    # equal => candidate_id ascending
            assert rows[i]["candidate_id"] < rows[i + 1]["candidate_id"]


def test_F6_3_zero_honeypot_in_top100_and_exposure(made):
    b = scoring.load_bundle(ART)
    r = risk.compute_risk(b)
    flagged = sum(1 for row in made["sel"].rows if r["honeypot_flag"][row["row_index"]])
    assert flagged == 0                                   # detected honeypots excluded
    assert flagged / 100.0 < 0.10                         # exposure comfortably < 10%


def test_F6_6_selection_manifest(made):
    import json
    m = json.load(open(os.path.join(made["out"], "selection_manifest.json"), encoding="utf-8"))
    for k in ("selected", "eligible", "excluded", "exclusions_by_reason", "top_excluded"):
        assert k in m
    assert m["selected"] == 100


def test_F6_4_official_validator_passes(made):
    res = subprocess.run([sys.executable, "validate_submission.py", made["csv"]],
                         capture_output=True, text=True)
    assert "Submission is valid." in res.stdout, res.stdout + res.stderr


def test_F6_7_validation_suite_catches_bad_selection(made):
    b = scoring.load_bundle(ART)
    sres = scoring.score_pool(b)
    sel = select.select(b, sres)
    assert validate_ok(sel, b, sres)
    # corrupt: duplicate a rank -> suite must complain
    sel.rows[1]["rank"] = sel.rows[0]["rank"]
    assert not validate_ok(sel, b, sres)


def validate_ok(sel, b, sres):
    return len(select.validate_selection(sel, b, sres)) == 0


# ---- trap suite at the SUBMISSION level (FT.1-FT.4) -----------------------
def test_FT_traps_at_submission_level(made):
    ids = {r["candidate_id"]: r for r in made["sel"].rows}
    assert "CAND_9000002" in ids                              # FT.2 plain-language Tier-5 IN
    assert "CAND_9000001" not in ids                          # FT.1 keyword-stuffer OUT
    for hp in ("CAND_9000003", "CAND_9000004", "CAND_9000005"):
        assert hp not in ids                                  # FT.3 honeypots excluded
    # FT.4 active outranks inactive when both selected (here active is in; inactive may not be)
    if "CAND_9000009" in ids and "CAND_9000010" in ids:
        assert ids["CAND_9000009"]["rank"] < ids["CAND_9000010"]["rank"]
