"""Validate the adversarial fixtures: schema-faithful AND carrying their intended trap
properties. These lock the fixtures so a later edit can't silently break the trap suite."""
from tests import fixtures
from src import common

REQ_TOP = {"candidate_id", "profile", "career_history", "education", "skills",
           "certifications", "languages", "redrob_signals"}
REQ_PROFILE = {"anonymized_name", "headline", "summary", "location", "country",
               "years_of_experience", "current_title", "current_company",
               "current_company_size", "current_industry"}
REQ_SIGNALS = {"profile_completeness_score", "signup_date", "last_active_date",
               "open_to_work_flag", "profile_views_received_30d", "applications_submitted_30d",
               "recruiter_response_rate", "avg_response_time_hours", "skill_assessment_scores",
               "connection_count", "endorsements_received", "notice_period_days",
               "expected_salary_range_inr_lpa", "preferred_work_mode", "willing_to_relocate",
               "github_activity_score", "search_appearance_30d", "saved_by_recruiters_30d",
               "interview_completion_rate", "offer_acceptance_rate", "verified_email",
               "verified_phone", "linkedin_connected"}
SIZE_ENUM = {"1-10", "11-50", "51-200", "201-500", "501-1000", "1001-5000", "5001-10000", "10001+"}
PROF_ENUM = {"beginner", "intermediate", "advanced", "expert"}
MODE_ENUM = {"remote", "hybrid", "onsite", "flexible"}


def test_all_fixtures_are_schema_faithful():
    for name, rec in fixtures.all_fixtures().items():
        assert REQ_TOP <= set(rec), f"{name}: missing top-level keys"
        assert common.is_valid_candidate_id(rec["candidate_id"]), name
        assert REQ_PROFILE <= set(rec["profile"]), f"{name}: missing profile keys"
        assert REQ_SIGNALS == set(rec["redrob_signals"]), f"{name}: signal key mismatch"
        assert rec["profile"]["current_company_size"] in SIZE_ENUM, name
        assert rec["redrob_signals"]["preferred_work_mode"] in MODE_ENUM, name
        assert len(rec["career_history"]) >= 1, name
        for role in rec["career_history"]:
            assert role["company_size"] in SIZE_ENUM, name
            assert isinstance(role["duration_months"], int)
        for sk in rec["skills"]:
            assert sk["proficiency"] in PROF_ENUM, name


def _summed_tenure_yrs(rec):
    return sum(r["duration_months"] for r in rec["career_history"]) / 12.0


def test_honeypot_fixtures_trip_their_rules():
    h1 = fixtures.honeypot_h1()
    assert h1["profile"]["years_of_experience"] - _summed_tenure_yrs(h1) > 2.0  # H1

    h2 = fixtures.honeypot_h2()
    assert _summed_tenure_yrs(h2) - h2["profile"]["years_of_experience"] > 2.0  # H2

    h3 = fixtures.honeypot_h3()
    zero_dur_expert = [s for s in h3["skills"]
                       if s["proficiency"] in ("advanced", "expert") and s["duration_months"] == 0]
    assert len(zero_dur_expert) >= 2  # H3


def test_stuffer_has_buzzwords_but_no_corroboration():
    s = fixtures.canonical_stuffer()
    assert s["profile"]["current_title"] == "HR Manager"
    skill_names = {sk["name"] for sk in s["skills"]}
    assert {"RAG", "Pinecone", "FAISS", "Embeddings"} <= skill_names  # buzzword stuffing
    assert s["redrob_signals"]["skill_assessment_scores"] == {}        # no Tier-1 corroboration
    assert s["redrob_signals"]["recruiter_response_rate"] >= 0.8       # "active but irrelevant"


def test_tier5_describes_system_without_buzzwords():
    t = fixtures.plain_language_tier5()
    blob = (t["profile"]["summary"] + " " +
            " ".join(r["description"] for r in t["career_history"])).lower()
    for buzz in ("rag", "pinecone", "ndcg", "faiss", "embedding"):
        assert buzz not in blob, f"tier5 should avoid the buzzword {buzz!r}"
    assert "recommendation" in blob or "search" in blob
    assert any(c in fixtures.PRODUCT for c in
               [r["company"] for r in t["career_history"]])  # product company


def test_consulting_fixtures():
    only = fixtures.consulting_only()
    assert all(r["company"] in fixtures.CONSULTING for r in only["career_history"])
    withprod = fixtures.consulting_with_product()
    assert any(r["company"] in fixtures.PRODUCT for r in withprod["career_history"])


def test_active_vs_inactive_same_fit_different_engagement():
    a, i = fixtures.active_strong(), fixtures.inactive_strong()
    assert a["career_history"] == i["career_history"]      # identical fit
    assert a["skills"] == i["skills"]
    assert a["redrob_signals"]["last_active_date"] > i["redrob_signals"]["last_active_date"]
    assert a["redrob_signals"]["recruiter_response_rate"] > i["redrob_signals"]["recruiter_response_rate"]
    assert a["redrob_signals"]["open_to_work_flag"] and not i["redrob_signals"]["open_to_work_flag"]
