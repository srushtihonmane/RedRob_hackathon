"""tests/fixtures.py — deterministic adversarial candidate fixtures.

Hand-crafted, schema-faithful synthetic records used by the trap/adversarial suite
(Goal 4 D9 / Goal 5 D6) and by unit tests across goals. These are NOT drawn from the real
pool — they are controlled inputs whose expected ranking behavior is known a priori:

  - canonical_stuffer        -> must rank DOWN / out of top-100 (AI skills, wrong title, no corroboration)
  - plain_language_tier5     -> must rank UP (built search/rec at a product co; no buzzwords)
  - honeypot_h1/h2/h3        -> must be EXCLUDED (logical impossibilities; OR-fired)
  - consulting_only          -> consulting gate FIRES (entire career services)
  - consulting_with_product  -> consulting gate SUPPRESSED (a product stint overrides)
  - cv_speech_primary        -> soft-penalized (CV/speech, no NLP/IR)
  - active_strong / inactive_strong -> identical fit; active must outrank inactive (engagement)

Each record validates against candidate_schema.json (required fields present, enums valid).
IDs use the CAND_90000xx band to avoid colliding with the real pool.
"""
from __future__ import annotations

import copy
from typing import Any

# Curated company-type anchors (mirrors Goal 3 company_table semantics).
CONSULTING = ["TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini",
              "HCL", "Mindtree", "Tech Mahindra", "Mphasis"]
PRODUCT = ["Razorpay", "Swiggy", "Flipkart", "CRED", "Zomato", "Meesho", "Ola", "Zoho"]


def _signals(**over: Any) -> dict[str, Any]:
    base = {
        "profile_completeness_score": 80.0,
        "signup_date": "2021-03-01",
        "last_active_date": "2024-05-01",
        "open_to_work_flag": True,
        "profile_views_received_30d": 40,
        "applications_submitted_30d": 5,
        "recruiter_response_rate": 0.6,
        "avg_response_time_hours": 12.0,
        "skill_assessment_scores": {},
        "connection_count": 300,
        "endorsements_received": 50,
        "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 30.0, "max": 45.0},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "github_activity_score": 40.0,
        "search_appearance_30d": 20,
        "saved_by_recruiters_30d": 3,
        "interview_completion_rate": 0.8,
        "offer_acceptance_rate": -1.0,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    }
    base.update(over)
    return base


def _role(company: str, title: str, start: str, end: str | None, months: int,
          industry: str, size: str, desc: str, current: bool = False) -> dict[str, Any]:
    return {
        "company": company, "title": title, "start_date": start, "end_date": end,
        "duration_months": months, "is_current": current, "industry": industry,
        "company_size": size, "description": desc,
    }


def _skill(name: str, prof: str, endorse: int, months: int) -> dict[str, Any]:
    return {"name": name, "proficiency": prof, "endorsements": endorse, "duration_months": months}


def base_candidate(cid: str, **over: Any) -> dict[str, Any]:
    """A complete, schema-valid baseline record. Override any top-level key via **over."""
    rec: dict[str, Any] = {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": "Test Person",
            "headline": "Software Engineer",
            "summary": "Software engineer with several years of experience building backend systems.",
            "location": "Bangalore",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "Software Engineer",
            "current_company": PRODUCT[0],
            "current_company_size": "1001-5000",
            "current_industry": "Software",
        },
        "career_history": [
            _role(PRODUCT[0], "Software Engineer", "2020-01-01", None, 52,
                  "Software", "1001-5000", "Built backend services.", current=True),
        ],
        "education": [
            {"institution": "IIT Bombay", "degree": "B.Tech", "field_of_study": "Computer Science",
             "start_year": 2013, "end_year": 2017, "grade": "8.5 CGPA", "tier": "tier_1"},
        ],
        "skills": [_skill("Python", "advanced", 30, 60)],
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": _signals(),
    }
    for k, v in over.items():
        rec[k] = v
    return rec


# --------------------------------------------------------------------------- #
# Adversarial fixtures                                                         #
# --------------------------------------------------------------------------- #
def canonical_stuffer() -> dict[str, Any]:
    """HR Manager listing every AI buzzword as a skill, with zero corroboration."""
    rec = base_candidate("CAND_9000001")
    rec["profile"].update({
        "headline": "HR Manager | People Operations | Talent",
        "summary": "Experienced HR Manager leading talent acquisition, people operations, "
                   "and employee engagement programs across fast-growing teams.",
        "current_title": "HR Manager", "current_industry": "Human Resources",
        "years_of_experience": 6.0,
    })
    rec["career_history"] = [
        _role(PRODUCT[1], "HR Manager", "2019-06-01", None, 60, "Human Resources",
              "1001-5000", "Led recruiting, onboarding, and HR operations for the org.",
              current=True),
    ]
    # The trap: a long AI skills list with no career/assessment backing.
    rec["skills"] = [
        _skill("RAG", "expert", 20, 30), _skill("Pinecone", "expert", 15, 24),
        _skill("FAISS", "advanced", 10, 20), _skill("Embeddings", "expert", 25, 30),
        _skill("LLM", "expert", 30, 28), _skill("Vector Search", "advanced", 12, 22),
        _skill("Recommendation Systems", "expert", 18, 26), _skill("NDCG", "advanced", 5, 18),
        _skill("Transformers", "expert", 22, 24),
    ]
    # Active + responsive (the "active but irrelevant" sample_submission trap).
    rec["redrob_signals"] = _signals(recruiter_response_rate=0.85, last_active_date="2024-05-20",
                                     skill_assessment_scores={})
    return rec


def plain_language_tier5() -> dict[str, Any]:
    """Built a recommendation/search system at a product company — described plainly,
    no 'RAG'/'Pinecone'/'NDCG' buzzwords. Should rank UP."""
    rec = base_candidate("CAND_9000002")
    rec["profile"].update({
        "headline": "Senior Software Engineer — Search & Discovery",
        "summary": "Senior engineer who built and ran the candidate-facing search and "
                   "recommendation system serving millions of users. I owned relevance "
                   "quality end to end: how we matched queries to items, how we measured "
                   "ranking quality offline against live engagement, and how we refreshed "
                   "the similarity index as the catalog changed.",
        "current_title": "Senior Software Engineer", "current_industry": "Internet",
        "years_of_experience": 7.0, "location": "Pune",
    })
    rec["career_history"] = [
        _role(PRODUCT[0], "Senior Software Engineer", "2021-01-01", None, 40, "Internet",
              "1001-5000",
              "Owned the product's search and recommendation stack. Built the retrieval "
              "layer that finds similar items from dense vectors plus a keyword index, "
              "and the ranking model that orders results. Set up offline evaluation that "
              "tracked ranking quality and correlated it with A/B engagement.",
              current=True),
        _role(PRODUCT[2], "Software Engineer", "2017-06-01", "2020-12-01", 42, "Internet",
              "5001-10000",
              "Worked on the recommendations team improving how relevant items were ranked "
              "for each user; ran experiments and tuned the matching pipeline at scale."),
    ]
    rec["skills"] = [
        _skill("Python", "expert", 60, 84), _skill("Search", "advanced", 40, 60),
        _skill("Recommendation Systems", "advanced", 35, 48), _skill("Spark", "advanced", 20, 40),
    ]
    rec["redrob_signals"] = _signals(
        recruiter_response_rate=0.7, last_active_date="2024-05-25", open_to_work_flag=True,
        skill_assessment_scores={"Python": 90.0, "Information Retrieval": 84.0})
    return rec


def honeypot_h1() -> dict[str, Any]:
    """yoe (17) far exceeds summed career tenure (3 yr): yoe - tenure = 14 > 2. AI/ML title."""
    rec = base_candidate("CAND_9000003")
    rec["profile"].update({
        "headline": "Principal Recommendation Systems Engineer",
        "current_title": "Recommendation Systems Engineer", "years_of_experience": 17.0,
        "summary": "Recommendation systems engineer.",
    })
    rec["career_history"] = [
        _role(PRODUCT[0], "Recommendation Systems Engineer", "2021-06-01", None, 36,
              "Internet", "1001-5000", "Worked on recommendation ranking.", current=True),
    ]
    return rec


def honeypot_h2() -> dict[str, Any]:
    """summed career tenure (12 yr) far exceeds yoe (2): tenure - yoe = 10 > 2."""
    rec = base_candidate("CAND_9000004")
    rec["profile"].update({"years_of_experience": 2.0})
    rec["career_history"] = [
        _role(PRODUCT[0], "Engineer", "2018-01-01", None, 72, "Internet", "1001-5000",
              "Engineering.", current=True),
        _role(PRODUCT[2], "Engineer", "2012-01-01", "2017-12-01", 72, "Internet", "5001-10000",
              "Engineering."),
    ]
    return rec


def honeypot_h3() -> dict[str, Any]:
    """>=2 advanced/expert skills with duration_months == 0 (use 3)."""
    rec = base_candidate("CAND_9000005")
    rec["skills"] = [
        _skill("Kubernetes", "expert", 10, 0), _skill("Rust", "advanced", 5, 0),
        _skill("GraphQL", "expert", 8, 0), _skill("Python", "advanced", 30, 60),
    ]
    return rec


def consulting_only() -> dict[str, Any]:
    """Entire career at consulting firms — gate FIRES (no product stint)."""
    rec = base_candidate("CAND_9000006")
    rec["profile"].update({"current_company": CONSULTING[0], "current_industry": "IT Services"})
    rec["career_history"] = [
        _role(CONSULTING[0], "Consultant", "2020-01-01", None, 52, "IT Services", "10001+",
              "Delivered client projects.", current=True),
        _role(CONSULTING[1], "Associate", "2016-01-01", "2019-12-01", 48, "IT Services", "10001+",
              "Client delivery."),
    ]
    return rec


def consulting_with_product() -> dict[str, Any]:
    """Mostly consulting but one product stint — gate SUPPRESSED by override."""
    rec = consulting_only()
    rec["candidate_id"] = "CAND_9000007"
    rec["career_history"].append(
        _role(PRODUCT[0], "Software Engineer", "2013-01-01", "2015-12-01", 36, "Internet",
              "1001-5000", "Built product features at a product company."))
    return rec


def cv_speech_primary() -> dict[str, Any]:
    """Computer-vision/speech specialist with no NLP/IR — soft negative archetype."""
    rec = base_candidate("CAND_9000008")
    rec["profile"].update({
        "headline": "Computer Vision Engineer | Perception",
        "current_title": "Computer Vision Engineer", "current_industry": "Robotics",
        "summary": "Computer vision and speech engineer specializing in object detection, "
                   "image segmentation, and speech recognition for perception systems.",
    })
    rec["career_history"] = [
        _role(PRODUCT[0], "Computer Vision Engineer", "2020-01-01", None, 52, "Robotics",
              "1001-5000", "Built object detection and image segmentation pipelines; "
              "deployed speech recognition for in-car assistants.", current=True),
    ]
    rec["skills"] = [
        _skill("OpenCV", "expert", 40, 60), _skill("YOLO", "expert", 30, 48),
        _skill("Speech Recognition", "advanced", 25, 40), _skill("Python", "advanced", 30, 60),
    ]
    rec["redrob_signals"] = _signals(
        skill_assessment_scores={"OpenCV": 88.0, "Computer Vision": 85.0, "Speech Recognition": 80.0})
    return rec


def _strong_relevant(cid: str) -> dict[str, Any]:
    rec = plain_language_tier5()
    rec["candidate_id"] = cid
    return rec


def active_strong() -> dict[str, Any]:
    """Strong + highly engaged (recent activity, high response, open to work)."""
    rec = _strong_relevant("CAND_9000009")
    rec["redrob_signals"] = _signals(
        recruiter_response_rate=0.9, last_active_date="2024-05-28", open_to_work_flag=True,
        avg_response_time_hours=3.0, interview_completion_rate=0.95,
        skill_assessment_scores={"Python": 90.0, "Information Retrieval": 84.0})
    return rec


def inactive_strong() -> dict[str, Any]:
    """Same strong fit but disengaged (stale, low response, not open) — must rank below active."""
    rec = _strong_relevant("CAND_9000010")
    rec["redrob_signals"] = _signals(
        recruiter_response_rate=0.05, last_active_date="2023-09-01", open_to_work_flag=False,
        avg_response_time_hours=120.0, interview_completion_rate=0.2,
        skill_assessment_scores={"Python": 90.0, "Information Retrieval": 84.0})
    return rec


ALL_FIXTURES: dict[str, Any] = {
    "canonical_stuffer": canonical_stuffer,
    "plain_language_tier5": plain_language_tier5,
    "honeypot_h1": honeypot_h1,
    "honeypot_h2": honeypot_h2,
    "honeypot_h3": honeypot_h3,
    "consulting_only": consulting_only,
    "consulting_with_product": consulting_with_product,
    "cv_speech_primary": cv_speech_primary,
    "active_strong": active_strong,
    "inactive_strong": inactive_strong,
}


def all_fixtures() -> dict[str, dict[str, Any]]:
    """Materialize every fixture as a fresh dict (deep-copied so callers can mutate safely)."""
    return {name: copy.deepcopy(fn()) for name, fn in ALL_FIXTURES.items()}
