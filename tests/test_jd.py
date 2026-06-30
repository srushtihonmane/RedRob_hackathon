"""Goal 2 tests: jd_query.json validation + evidence-path resolution (model-independent).
The vector-generation tests live in test_jd_vectors.py (require the downloaded model)."""
import copy

from src import jd_build
from src.common import read_json

SCHEMA = read_json("candidate_schema.json")
JD = read_json("jd/jd_query.json")


def test_evidence_path_resolution():
    ok = ["profile.summary", "career_history[].description", "redrob_signals.skill_assessment_scores",
          "education[].tier", "skills[].name", "redrob_signals.open_to_work_flag",
          "profile.years_of_experience"]
    for p in ok:
        assert jd_build.evidence_path_exists(SCHEMA, p), p
    bad = ["profile.nonexistent", "career_history[].nope", "redrob_signals.foo",
           "skills[].xyz", "made_up.path"]
    for p in bad:
        assert not jd_build.evidence_path_exists(SCHEMA, p), p


def test_real_jd_query_validates_clean():
    errs = jd_build.validate_jd_query(JD, SCHEMA)
    assert errs == [], f"jd_query.json should validate clean, got: {errs}"


def test_jd_query_has_expected_structure():
    crits = {c["id"]: c for c in JD["criteria"]}
    must = [c for c in JD["criteria"] if c["criterion_type"] == "must_have"]
    assert len(must) >= 5
    # the four "absolutely need" + strong python
    for cid in ("production_embeddings_retrieval", "vector_db_hybrid_search_ops",
                "shipped_ranking_search_rec_at_scale", "ranking_eval_frameworks", "strong_python"):
        assert crits[cid]["criterion_type"] == "must_have"
    # consulting hard gate with override
    assert crits["consulting_entire_career"]["criterion_type"] == "hard_gate"
    assert crits["consulting_entire_career"]["override_condition"] == "any_product_stint"
    # honeypot umbrella present
    assert crits["profile_internally_consistent"]["criterion_type"] == "hard_gate"
    # atomic engagement signals exist
    assert any(c["id"].startswith("avail_") for c in JD["criteria"])
    # 5 ordered anti-profiles
    assert JD["antiprofile_labels"] == [a["label"] for a in JD["antiprofiles"]]
    assert len(JD["antiprofile_labels"]) == 5


def test_validation_catches_corruption():
    bad = copy.deepcopy(JD)
    bad["criteria"][0]["evidence_sources"].append("profile.does_not_exist")
    bad["criteria"][1]["id"] = bad["criteria"][0]["id"]      # duplicate id
    bad["criteria"][2]["criterion_type"] = "not_a_type"      # bad enum
    errs = jd_build.validate_jd_query(bad, SCHEMA)
    assert any("not in candidate_schema" in e for e in errs)
    assert any("duplicate criterion id" in e for e in errs)
    assert any("bad criterion_type" in e for e in errs)
