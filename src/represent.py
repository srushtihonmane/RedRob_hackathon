"""represent.py — Goal 3: candidate representation (structured features + corroboration).

Governing principle (Goal 3 D4): EXPOSE INPUTS, NOT VERDICTS. Raw/continuous derived values,
deterministic flags, pool statistics. ALL thresholds/weights/transforms/gate-firing/
corroboration-counting defer to Goals 4/5/6.

This module owns the model-INDEPENDENT structured features (G1-G4 components, G6-G8, the 23
signals, and the static corroboration channels {assessment, title, skillmeta}). The
model-DEPENDENT features (G5 archetype cosines, G9 role-relevance, G10 identity/evidence
divergence) and the embeddings/BM25/snippets are assembled by the builder/driver after
embedding (see build.py).

NaN+_present contract (Goal 3 D7, binding): missing => value is NaN AND companion `_present`==0;
invariant `_present==0 => value MUST be NaN`. No sentinels, no imputation. Built by
construction via ``FeatureRow.add_nullable``.
"""
from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Any

from .common import days_between, parse_date, read_json

NAN = float("nan")

TIER1_CITIES = {"bangalore", "bengaluru", "mumbai", "delhi", "new delhi", "gurgaon", "gurugram",
                "noida", "pune", "hyderabad", "chennai", "kolkata", "ahmedabad"}
NOIDA_PUNE = {"noida", "pune"}
FIELD_RELEVANT = ("computer", "cs", "information", "data", "machine learning",
                  "artificial intelligence", " ai", "ml", "electronic", "electrical",
                  "software", "statistic", "mathematic")
WORK_MODES = ["remote", "hybrid", "onsite", "flexible"]


# --------------------------------------------------------------------------- #
# Frozen inputs (pool-independent given these — Sandbox parity, contracts C11) #
# --------------------------------------------------------------------------- #
@dataclass
class FrozenInputs:
    ref_date: _dt.date
    company_types: dict[str, str]            # company name -> type
    consulting_firms: set[str]
    concepts: dict[str, dict]                # concept registry .concepts
    model: dict | None = None
    jd_ideal: Any = None                     # [1,384] (added for embedding layer)
    jd_antiprofile: Any = None               # [A,384]
    antiprofile_labels: list[str] = field(default_factory=list)

    @staticmethod
    def load(ref_date: _dt.date, company_table_path="reference/company_table.json",
             concept_registry_path="reference/concept_registry.json",
             jd_query_path="jd/jd_query.json", jd_dir="jd") -> "FrozenInputs":
        import os
        ct = read_json(company_table_path)
        cr = read_json(concept_registry_path)
        jd = read_json(jd_query_path)
        jd_ideal = jd_anti = None
        ip = os.path.join(jd_dir, "jd_ideal.npy")
        ap = os.path.join(jd_dir, "jd_antiprofile.npy")
        if os.path.exists(ip) and os.path.exists(ap):
            import numpy as np
            jd_ideal = np.load(ip)
            jd_anti = np.load(ap)
        return FrozenInputs(
            ref_date=ref_date,
            company_types={k.lower(): v for k, v in ct["companies"].items()},
            consulting_firms={c.lower() for c in ct["consulting_firms"]},
            concepts=cr["concepts"],
            model=jd["model"],
            jd_ideal=jd_ideal,
            jd_antiprofile=jd_anti,
            antiprofile_labels=jd["antiprofile_labels"],
        )

    def company_type(self, name: str) -> str:
        return self.company_types.get((name or "").strip().lower(), "unknown")


# --------------------------------------------------------------------------- #
# Feature row builder — enforces the NaN/_present invariant by construction    #
# --------------------------------------------------------------------------- #
class FeatureRow:
    def __init__(self):
        self.values: dict[str, float] = {}
        self.nullable: set[str] = set()

    def add(self, name: str, value: float) -> None:
        self.values[name] = float(value)

    def add_bool(self, name: str, value: bool) -> None:
        self.values[name] = 1.0 if value else 0.0

    def add_nullable(self, name: str, value: float | None) -> None:
        """Missing -> NaN value + {name}_present==0; present -> value + _present==1."""
        present = value is not None and not (isinstance(value, float) and math.isnan(value))
        self.values[name] = float(value) if present else NAN
        self.values[name + "_present"] = 1.0 if present else 0.0
        self.nullable.add(name)


# --------------------------------------------------------------------------- #
# Group derivations                                                            #
# --------------------------------------------------------------------------- #
def _months(role: dict) -> float:
    return float(role.get("duration_months") or 0)


def _g1_experience(r: FeatureRow, rec: dict) -> None:
    prof = rec["profile"]
    roles = rec["career_history"]
    durations = [_months(x) for x in roles]
    summed = sum(durations) / 12.0
    r.add("yoe", float(prof.get("years_of_experience") or 0.0))
    r.add("summed_tenure_yrs", summed)
    r.add("n_roles", float(len(roles)))
    cur = [x for x in roles if x.get("is_current")]
    r.add("current_role_tenure_mo", float(max((_months(x) for x in cur), default=0.0)))
    r.add("longest_role_tenure_mo", float(max(durations, default=0.0)))
    durations_sorted = sorted(durations)
    n = len(durations_sorted)
    median = durations_sorted[n // 2] if n else 0.0
    if n and n % 2 == 0:
        median = (durations_sorted[n // 2 - 1] + durations_sorted[n // 2]) / 2.0
    r.add("median_role_tenure_mo", float(median))
    r.add("mean_role_tenure_mo", float(sum(durations) / n) if n else 0.0)


def _g2_jobhop(r: FeatureRow, rec: dict) -> None:
    roles = rec["career_history"]
    durations = [_months(x) for x in roles]
    n = len(roles)
    yoe = float(rec["profile"].get("years_of_experience") or 0.0)
    short = sum(1 for d in durations if d < 18)
    r.add("count_stints_under_18mo", float(short))
    r.add("frac_short_stints", float(short / n) if n else 0.0)
    r.add("switches_per_year", float((n - 1) / yoe) if yoe > 0 else 0.0)
    exits = [_months(x) for x in roles if not x.get("is_current")]
    r.add("mean_tenure_at_exit", float(sum(exits) / len(exits)) if exits else 0.0)


def _g3_company(r: FeatureRow, rec: dict, fz: FrozenInputs) -> None:
    roles = rec["career_history"]
    types = [fz.company_type(x.get("company", "")) for x in roles]
    durations = [_months(x) for x in roles]
    total = sum(durations) or 1.0
    any_product = any(t == "product_tech" for t in types)
    services = sum(d for d, t in zip(durations, types) if t == "services_consulting")
    product = sum(d for d, t in zip(durations, types) if t == "product_tech")
    r.add_bool("any_product_stint", any_product)
    r.add("frac_career_services", float(services / total))
    r.add_bool("is_consulting_entire_career",
               bool(roles) and all(t == "services_consulting" for t in types))
    r.add("product_tenure_yrs", float(product / 12.0))
    sizes = {"1-10": 1, "11-50": 2, "51-200": 3, "201-500": 4, "501-1000": 5,
             "1001-5000": 6, "5001-10000": 7, "10001+": 8}
    prod_sizes = [sizes.get(x.get("company_size"), 0) for x, t in zip(roles, types) if t == "product_tech"]
    r.add("max_company_size_at_product", float(max(prod_sizes, default=0)))
    by_emp: dict[str, float] = {}
    for x, d in zip(roles, durations):
        by_emp[x.get("company", "")] = by_emp.get(x.get("company", ""), 0.0) + d
    r.add("largest_employer_fraction", float(max(by_emp.values(), default=0.0) / total))


def _concept_text_hit(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)


def _g4_applied_ml(r: FeatureRow, rec: dict, fz: FrozenInputs) -> None:
    """Keyword-based ML-role tenure components (the embedding-weighted version is added later)."""
    ml_concepts = ("retrieval_embeddings", "vector_db", "ranking_eval", "learning_to_rank",
                   "llm_finetune", "nlp_ir")
    kws = []
    for c in ml_concepts:
        kws += fz.concepts[c]["keywords"]
    ml_tenure = 0.0
    ml_count = 0
    for x in rec["career_history"]:
        blob = f"{x.get('title','')} {x.get('description','')}"
        if _concept_text_hit(blob, kws):
            ml_tenure += _months(x)
            ml_count += 1
    r.add("kw_ml_role_tenure_yrs", float(ml_tenure / 12.0))
    r.add("ml_role_count", float(ml_count))


def _degree_level(degree: str) -> int:
    d = (degree or "").lower()
    if "phd" in d or "ph.d" in d or "doctor" in d:
        return 3
    if d.startswith("m") or "master" in d or "mtech" in d or "m.tech" in d or "mba" in d or "m.e" in d or "ms" in d:
        return 2
    if d.startswith("b") or "bachelor" in d or "btech" in d or "b.tech" in d or "b.e" in d:
        return 1
    return 0


def _g6_education(r: FeatureRow, rec: dict, fz: FrozenInputs) -> None:
    edu = rec["education"]
    tier_map = {"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1, "unknown": 0}
    tiers = [tier_map.get((e.get("tier") or "unknown"), 0) for e in edu]
    r.add("highest_tier_ordinal", float(max(tiers, default=0)))
    r.add_bool("has_tier1", any(t == 4 for t in tiers))
    levels = [_degree_level(e.get("degree", "")) for e in edu]
    r.add("highest_degree_level", float(max(levels, default=0)))
    r.add_bool("has_phd", any(l == 3 for l in levels))
    relevant = any(any(k in (e.get("field_of_study") or "").lower() for k in FIELD_RELEVANT) for e in edu)
    r.add_bool("field_relevance", relevant)
    grad_years = [e.get("end_year") for e in edu if e.get("end_year")]
    r.add_nullable("latest_grad_year", float(max(grad_years)) if grad_years else None)


def _timeline_overlap_months(roles: list[dict]) -> float:
    spans = []
    for x in roles:
        sd = parse_date(x.get("start_date")) if isinstance(x.get("start_date"), str) else x.get("start_date")
        ed = x.get("end_date")
        ed = parse_date(ed) if isinstance(ed, str) else ed
        if sd is None:
            continue
        spans.append((sd, ed))
    spans.sort(key=lambda s: s[0])
    overlap = 0.0
    for i in range(len(spans) - 1):
        s1, e1 = spans[i]
        s2, _ = spans[i + 1]
        if e1 is not None and e1 > s2:
            overlap += (e1 - s2).days / 30.0
    return overlap


def _g7_consistency(r: FeatureRow, rec: dict, fz: FrozenInputs) -> None:
    yoe = float(rec["profile"].get("years_of_experience") or 0.0)
    summed = sum(_months(x) for x in rec["career_history"]) / 12.0
    r.add("yoe_minus_summed_tenure", yoe - summed)
    zero_dur_expert = sum(1 for s in rec["skills"]
                          if s.get("proficiency") in ("advanced", "expert")
                          and (s.get("duration_months") == 0))
    r.add("expert_skill_zero_duration_count", float(zero_dur_expert))
    career_months = sum(_months(x) for x in rec["career_history"])
    max_skill_dur = max((float(s.get("duration_months") or 0) for s in rec["skills"]), default=0.0)
    r.add_bool("skill_duration_exceeds_career_flag", max_skill_dur > career_months + 1e-9)
    r.add("timeline_overlap_months", _timeline_overlap_months(rec["career_history"]))
    grad_years = [e.get("end_year") for e in rec["education"] if e.get("end_year")]
    if grad_years:
        yrs_since = fz.ref_date.year - max(grad_years)
        r.add_nullable("gradyear_vs_yoe_gap", float(yrs_since - yoe))
    else:
        r.add_nullable("gradyear_vs_yoe_gap", None)


def _g8_location(r: FeatureRow, rec: dict) -> None:
    loc = (rec["profile"].get("location") or "").lower()
    country = (rec["profile"].get("country") or "").lower()
    r.add_bool("is_noida_pune", any(c in loc for c in NOIDA_PUNE))
    r.add_bool("is_tier1_indian_city", any(c in loc for c in TIER1_CITIES))
    r.add_bool("is_india", country == "india")


def _signals(r: FeatureRow, rec: dict, fz: FrozenInputs) -> None:
    s = rec["redrob_signals"]
    for k in ("profile_completeness_score", "profile_views_received_30d",
              "applications_submitted_30d", "recruiter_response_rate", "avg_response_time_hours",
              "connection_count", "endorsements_received", "notice_period_days",
              "search_appearance_30d", "saved_by_recruiters_30d", "interview_completion_rate"):
        r.add("sig_" + k, float(s.get(k) or 0.0))
    # sentinel -1 -> NaN + present flag
    gh = s.get("github_activity_score")
    r.add_nullable("sig_github_activity_score", None if gh is None or gh < 0 else float(gh))
    oa = s.get("offer_acceptance_rate")
    r.add_nullable("sig_offer_acceptance_rate", None if oa is None or oa < 0 else float(oa))
    for k in ("open_to_work_flag", "willing_to_relocate", "verified_email", "verified_phone",
              "linkedin_connected"):
        r.add_bool("sig_" + k, bool(s.get(k)))
    mode = s.get("preferred_work_mode")
    for m in WORK_MODES:
        r.add_bool(f"sig_work_mode_{m}", mode == m)
    sal = s.get("expected_salary_range_inr_lpa") or {}
    smin, smax = sal.get("min"), sal.get("max")
    has_sal = smin is not None and smax is not None
    r.add_bool("sig_salary_present", has_sal)
    r.add_nullable("sig_salary_min", float(smin) if has_sal else None)
    r.add_nullable("sig_salary_max", float(smax) if has_sal else None)
    r.add_nullable("sig_salary_mid", float((smin + smax) / 2.0) if has_sal else None)
    la = parse_date(s.get("last_active_date")) if isinstance(s.get("last_active_date"), str) else s.get("last_active_date")
    su = parse_date(s.get("signup_date")) if isinstance(s.get("signup_date"), str) else s.get("signup_date")
    r.add_nullable("sig_last_active_recency_days", days_between(fz.ref_date, la))
    r.add_nullable("sig_account_tenure_days", days_between(fz.ref_date, su))


def _corroboration(r: FeatureRow, rec: dict, fz: FrozenInputs) -> None:
    assess = rec["redrob_signals"].get("skill_assessment_scores") or {}
    skills = rec["skills"]
    titles = " ".join([rec["profile"].get("current_title", "")]
                      + [x.get("title", "") for x in rec["career_history"]]).lower()
    for cname, cdef in fz.concepts.items():
        # Tier-1 Verified: assessment max over the concept's assessment skills.
        scores = [assess[a] for a in cdef["assessment_skills"] if a in assess]
        r.add_nullable(f"assess_{cname}_max", float(max(scores)) if scores else None)
        # Tier-3 metadata: max skill duration among skills[] whose name matches concept keywords.
        kws = cdef["keywords"]
        durs = [float(s.get("duration_months") or 0) for s in skills
                if _concept_text_hit(s.get("name", ""), kws)]
        r.add_nullable(f"skillmeta_{cname}_max_duration", float(max(durs)) if durs else None)
        # Title hit (career/current titles).
        r.add_bool(f"title_hit_{cname}", _concept_text_hit(titles, kws))


def derive_structured(rec: dict, fz: FrozenInputs) -> FeatureRow:
    """All model-INDEPENDENT features for one candidate. Pure given frozen inputs."""
    r = FeatureRow()
    _g1_experience(r, rec)
    _g2_jobhop(r, rec)
    _g3_company(r, rec, fz)
    _g4_applied_ml(r, rec, fz)
    _g6_education(r, rec, fz)
    _g7_consistency(r, rec, fz)
    _g8_location(r, rec)
    _signals(r, rec, fz)
    _corroboration(r, rec, fz)
    return r
