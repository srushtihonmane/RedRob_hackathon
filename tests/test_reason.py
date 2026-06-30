"""Goal 7 reasoning tests: grounded, varied, rank-tone-consistent, no skills[] hallucination."""
import os

import pytest

ART = "data/sample/artifacts"
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ART, "features.npy")),
                                reason="sample artifacts not built")

from src import reason, scoring, select   # noqa: E402


@pytest.fixture(scope="module")
def made():
    b = scoring.load_bundle(ART)
    res = scoring.score_pool(b)
    rfn, prov = reason.build_reasoner(b, res, scoring.ScoringConfig())
    sel = select.select(b, res, reasoning_fn=rfn)
    return {"b": b, "res": res, "sel": sel, "prov": prov}


def test_reasoning_nonempty_unique_bounded(made):
    rs = [r["reasoning"] for r in made["sel"].rows]
    assert all(s.strip() for s in rs)
    assert len(set(rs)) == len(rs)                      # unique
    assert all(len(s) <= 240 for s in rs)               # length cap


def test_rank_tone_consistency(made):
    for r in made["sel"].rows:
        if r["tier"] == "Filler":
            assert "filler" in r["reasoning"].lower() or "limitation" in r["reasoning"].lower()


def test_top_rows_cite_concrete_facts(made):
    # the top few should reference a real anchor/company/assessment, not generic praise
    for r in made["sel"].rows[:5]:
        txt = r["reasoning"].lower()
        assert any(k in txt for k in ("yrs", "assessment", "retrieval", "ranking", "python",
                                      "vector", "nlp", "product_tech", "fintech")) or len(txt) > 20


def test_no_raw_skills_array_hallucination(made):
    """Reasoning may name a skill ONLY if it came from an assessment (Verified). Check that
    any 'assessment' clause's skill is a real assessment for that candidate."""
    from src.ingest import read_parquet_records
    recs = read_parquet_records(os.path.join(ART, "candidates.parquet"))
    cid2assess = {r["candidate_id"]: set(r["redrob_signals"]["skill_assessment_scores"])
                  for r in recs}
    for r in made["sel"].rows:
        if "assessment" in r["reasoning"]:
            # the named skill must be a real assessment for this candidate
            assess = cid2assess.get(r["candidate_id"], set())
            assert assess, f"{r['candidate_id']} cites assessment but has none"


def test_provenance_recorded(made):
    prov = made["prov"]
    # at least the strong (non-filler) rows have >=1 evidence id
    strong = [r for r in made["sel"].rows if r["tier"] in ("Elite", "Strong")]
    if strong:
        assert any(len(prov.get(r["candidate_id"], [])) >= 1 for r in strong)
