"""scoring.py — Goal 4: the relevance engine (4A retrieval -> 4B criterion eval -> 4C fusion
-> 4D calibration -> 4E modifiers). Emits a per-candidate fit bundle (contracts.md C7).

LOAD-BEARING design (faithful to Goal 4):
  * Criterion-satisfaction is the FIT SPINE; BM25/dense are evidence SOURCES, not additive
    channels (a high dense cosine alone can never inflate fit).
  * Corroboration-first: trust-tiered noisy-OR (Verified assessment / Demonstrated career+dense
    / Declared) x a soft corroboration gate. Tier-3-only evidence attenuates toward a floor.
  * Must-haves combine via a weighted geometric-mean SOFT-AND (any near-zero must-have drags
    the block down) -> this naturally pins the irrelevant bulk near 0 (tail-anchor property).
  * engagement_modifier in [0.7,1.1] and risk_modifier in [0.5,1.0] are bounded, separable,
    and APPLIED IN GOAL 6 — never fused into fit. low-fit x high-engagement stays low.

This module loads precomputed artifacts and does pure array arithmetic — NO neural model
(Runtime ML Policy). It is imported by rank.py.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import bm25 as bm25mod
from . import transforms as T
from .common import read_json

# Which concept(s) corroborate each criterion (implementation detail of 4B).
CRITERION_CONCEPTS = {
    "production_embeddings_retrieval": ["retrieval_embeddings"],
    "vector_db_hybrid_search_ops": ["vector_db"],
    "shipped_ranking_search_rec_at_scale": ["ranking_eval"],
    "ranking_eval_frameworks": ["ranking_eval"],
    "strong_python": ["strong_python"],
    "llm_finetuning": ["llm_finetune"],
    "learning_to_rank": ["learning_to_rank"],
    "distributed_systems_scale": ["distributed_systems"],
    "nlp_ir": ["nlp_ir"],
}
CORE_DENSE_CONCEPTS = {"retrieval_embeddings", "vector_db", "ranking_eval", "nlp_ir"}


@dataclass
class ScoringConfig:
    importance_weights: dict = field(default_factory=lambda: {
        "critical": 1.0, "high": 0.6, "medium": 0.35, "low": 0.15})
    theta_assess: float = 60.0
    tier_weight_verified: float = 1.0
    tier_weight_demonstrated: float = 0.85
    tier_weight_declared: float = 0.32
    bm25_sat_c: float = 3.0
    dense_lo_key: str = "p75"
    dense_hi_key: str = "p99"
    soft_and_eps: float = 0.05
    nh_each: float = 0.12
    nh_cap: float = 0.30
    cm_clamp_lo: float = 0.85
    cm_clamp_hi: float = 1.15
    engagement_slope: float = 0.6
    engagement_lo: float = 0.7
    engagement_hi: float = 1.1
    risk_floor: float = 0.5
    risk_w_geometry: float = 0.5
    risk_w_archetype: float = 0.8
    tier_elite: float = 0.60
    tier_strong: float = 0.40
    tier_plausible: float = 0.20


# --------------------------------------------------------------------------- #
# Artifact bundle                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class Bundle:
    candidate_ids: np.ndarray
    feats: np.ndarray
    fidx: dict
    stats: dict
    emb_identity: np.ndarray
    emb_evidence: np.ndarray
    bm25_W: sp.csr_matrix
    bm25_vocab: dict
    jd: dict
    jd_ideal: np.ndarray
    jd_antiprofile: np.ndarray
    antiprofile_labels: list
    art_dir: str

    def col(self, name: str) -> np.ndarray:
        return self.feats[:, self.fidx[name]]


def load_bundle(art_dir: str, jd_dir: str = "jd") -> Bundle:
    art = Path(art_dir)
    fm = read_json(art / "feature_manifest.json")
    fidx = {c["name"]: c["index"] for c in fm["columns"]}
    bm = art / "bm25_index"
    return Bundle(
        candidate_ids=np.load(art / "candidate_ids.npy"),
        feats=np.load(art / "features.npy"),
        fidx=fidx,
        stats=read_json(art / "normalization_stats.json"),
        emb_identity=np.load(art / "embeddings_identity.npy"),
        emb_evidence=np.load(art / "embeddings_evidence.npy"),
        bm25_W=sp.load_npz(bm / "matrix.npz"),
        bm25_vocab=json.load(open(bm / "vocab.json", encoding="utf-8")),
        jd=read_json(Path(jd_dir) / "jd_query.json"),
        jd_ideal=np.load(Path(jd_dir) / "jd_ideal.npy"),
        jd_antiprofile=np.load(Path(jd_dir) / "jd_antiprofile.npy"),
        antiprofile_labels=read_json(Path(jd_dir) / "jd_query.json")["antiprofile_labels"],
        art_dir=art_dir,
    )


# --------------------------------------------------------------------------- #
# Channels (vectorized, full-pool, cheap)                                      #
# --------------------------------------------------------------------------- #
def _bm25_concept_scores(b: Bundle) -> dict:
    """Per-concept BM25 score (N,) by scoring only that concept's phrase-joined terms."""
    phrases = bm25mod.build_phrase_list()
    out = {}
    for cname, cdef in read_json("reference/concept_registry.json")["concepts"].items():
        terms = set()
        for kw in cdef["keywords"]:
            terms.update(bm25mod.tokenize(kw, phrases))
        q = np.zeros(b.bm25_W.shape[1], dtype=np.float32)
        for t in terms:
            if t in b.bm25_vocab:
                q[b.bm25_vocab[t]] = 1.0
        out[cname] = np.asarray(b.bm25_W @ q).ravel()
    return out


def _saturate(x: np.ndarray, c: float) -> np.ndarray:
    return x / (x + c)


# --------------------------------------------------------------------------- #
# 4A Retrieval — recall-first union shortlist                                  #
# --------------------------------------------------------------------------- #
def retrieve_shortlist(b: Bundle, k_per_channel: int = 3000) -> np.ndarray:
    n = b.feats.shape[0]
    if n <= k_per_channel:
        return np.arange(n)
    ideal = b.jd_ideal[0]
    dense_ev = b.emb_evidence @ ideal
    dense_id = b.emb_identity @ ideal
    qall = bm25mod.query_terms_from_jd()
    qv = np.zeros(b.bm25_W.shape[1], dtype=np.float32)
    for t in qall:
        if t in b.bm25_vocab:
            qv[b.bm25_vocab[t]] = 1.0
    bm25_all = np.asarray(b.bm25_W @ qv).ravel()

    def topk(arr):
        return np.argpartition(-arr, k_per_channel)[:k_per_channel]

    members = set(topk(dense_ev)) | set(topk(dense_id)) | set(topk(bm25_all))
    # structural prefilter: strong non-text evidence (any assessment present, product ML tenure)
    struct = np.zeros(n, dtype=bool)
    for cname in CRITERION_CONCEPTS:
        for c in CRITERION_CONCEPTS[cname]:
            key = f"assess_{c}_max_present"
            if key in b.fidx:
                struct |= b.col(key) > 0
    struct |= b.col("product_tenure_yrs") > 1.0
    members |= set(np.nonzero(struct)[0].tolist())
    return np.array(sorted(members), dtype=np.int64)


# --------------------------------------------------------------------------- #
# 4B Criterion evaluation (per concept, vectorized over the pool)              #
# --------------------------------------------------------------------------- #
def _concept_evidence(b: Bundle, cfg: ScoringConfig, bm25c: dict, dense_support: np.ndarray):
    """Return per-concept dict: {concept: {'sat':(N,), 'sources':(N,) count}}."""
    out = {}
    for c in set(sum(CRITERION_CONCEPTS.values(), [])):
        n = b.feats.shape[0]
        assess_present = b.col(f"assess_{c}_max_present") if f"assess_{c}_max_present" in b.fidx else np.zeros(n)
        assess_max = b.col(f"assess_{c}_max") if f"assess_{c}_max" in b.fidx else np.full(n, np.nan)
        verified = np.where((assess_present > 0) & (np.nan_to_num(assess_max) >= cfg.theta_assess),
                            np.clip(np.nan_to_num(assess_max) / 100.0, 0, 1), 0.0)
        lex = _saturate(bm25c.get(c, np.zeros(n)), cfg.bm25_sat_c)
        title = b.col(f"title_hit_{c}") if f"title_hit_{c}" in b.fidx else np.zeros(n)
        demonstrated = np.maximum(lex, 0.6 * title)
        if c in CORE_DENSE_CONCEPTS:
            demonstrated = np.maximum(demonstrated, dense_support)
        skillmeta_present = b.col(f"skillmeta_{c}_max_duration_present") if f"skillmeta_{c}_max_duration_present" in b.fidx else np.zeros(n)
        declared = 0.4 * skillmeta_present
        wv, wd, wde = cfg.tier_weight_verified, cfg.tier_weight_demonstrated, cfg.tier_weight_declared
        sat = 1.0 - (1 - wv * verified) * (1 - wd * demonstrated) * (1 - wde * declared)
        nsrc = (verified > 0.1).astype(float) + (demonstrated > 0.1).astype(float) + (declared > 0.1).astype(float)
        out[c] = {"sat": sat, "sources": nsrc, "verified": verified,
                  "demonstrated": demonstrated, "declared": declared}
    return out


def _criterion_sat(b: Bundle, crit: dict, concept_ev: dict, cfg: ScoringConfig) -> np.ndarray:
    cs = CRITERION_CONCEPTS.get(crit["id"])
    min_src = crit.get("corroboration", {}).get("min_sources", 1)
    if cs:
        sats = []
        for c in cs:
            ev = concept_ev[c]
            gate = np.clip(ev["sources"] / max(min_src, 1), 0, 1)
            gate = gate * gate * (3 - 2 * gate)  # smoothstep
            sats.append(ev["sat"] * gate)
        return np.maximum.reduce(sats)
    return _structured_criterion(b, crit, cfg)


def _structured_criterion(b: Bundle, crit: dict, cfg: ScoringConfig) -> np.ndarray:
    """Criteria without a concept: structured/signal evidence (hr_tech, open_source, modifiers)."""
    n = b.feats.shape[0]
    cid = crit["id"]
    if cid == "hr_tech_marketplace":
        return np.zeros(n)  # industry-based; low weight nice-to-have, handled as 0 default
    if cid == "open_source_contributions":
        g = b.col("sig_github_activity_score_present") * np.clip(np.nan_to_num(b.col("sig_github_activity_score")) / 100.0, 0, 1)
        return g
    return np.zeros(n)


# --------------------------------------------------------------------------- #
# 4C Fusion + 4D Calibration                                                   #
# --------------------------------------------------------------------------- #
def _context_modifier(b: Bundle, cfg: ScoringConfig) -> np.ndarray:
    yoe = b.col("yoe")
    exp_band = np.array([T.trapezoid(v, 4, 6, 8, 10) for v in yoe])      # 5-9 soft, 6-8 ideal
    prod = b.col("product_tenure_yrs")
    aml = np.array([T.trapezoid(v, 1, 4, 6, 10) for v in prod])          # ideal 4-5
    noida = b.col("is_noida_pune"); tier1 = b.col("is_tier1_indian_city")
    relocate = b.col("sig_willing_to_relocate")
    loc = np.clip(0.5 + 0.5 * noida + 0.2 * tier1 + 0.15 * relocate, 0, 1)
    notice = b.col("sig_notice_period_days")
    notice_mod = np.clip(1.0 - np.maximum(notice - 30, 0) / 180.0 * 0.5, 0.5, 1.0)
    cm = 1.0 + 0.10 * (exp_band - 0.5) + 0.10 * (aml - 0.5) + 0.05 * (loc - 0.5) + 0.05 * (notice_mod - 1.0)
    return np.clip(cm, cfg.cm_clamp_lo, cfg.cm_clamp_hi)


def fuse_and_calibrate(b: Bundle, cfg: ScoringConfig, crit_sats: dict):
    n = b.feats.shape[0]
    crit_by_id = {c["id"]: c for c in b.jd["criteria"]}
    # Must-have soft-AND (weighted geometric mean).
    mh = [c for c in b.jd["criteria"] if c["criterion_type"] == "must_have"]
    wsum = np.zeros(n); logsum = np.zeros(n)
    for c in mh:
        w = cfg.importance_weights[c["importance"]]
        s = np.maximum(crit_sats[c["id"]], cfg.soft_and_eps)
        logsum += w * np.log(s); wsum += w
    mh_soft = np.exp(logsum / np.maximum(wsum, 1e-9))
    # Nice-to-have bonus.
    nh = [c for c in b.jd["criteria"] if c["criterion_type"] == "nice_to_have"]
    nh_bonus = np.zeros(n)
    for c in nh:
        nh_bonus += cfg.nh_each * crit_sats[c["id"]]
    nh_bonus = np.minimum(nh_bonus, cfg.nh_cap)
    cm = _context_modifier(b, cfg)
    structured = mh_soft * (1 + nh_bonus) * cm
    fit = np.clip(structured, 0, 1)
    return fit, mh_soft, nh_bonus, cm


def fit_tier(fit: np.ndarray, cfg: ScoringConfig) -> np.ndarray:
    out = np.empty(fit.shape[0], dtype=object)
    for i, v in enumerate(fit):
        out[i] = ("Elite" if v >= cfg.tier_elite else "Strong" if v >= cfg.tier_strong
                  else "Plausible" if v >= cfg.tier_plausible else "Filler")
    return out


# --------------------------------------------------------------------------- #
# 4E Modifiers                                                                 #
# --------------------------------------------------------------------------- #
def engagement_modifier(b: Bundle, cfg: ScoringConfig) -> np.ndarray:
    rr = b.col("sig_recruiter_response_rate")
    la = b.col("sig_last_active_recency_days")
    la_q = np.array([T.quantile_norm(v, b.stats["sig_last_active_recency_days"], invert=True)
                     if not math.isnan(v) else 0.5 for v in la])
    otw = b.col("sig_open_to_work_flag")
    ic = b.col("sig_interview_completion_rate")
    art = b.col("sig_avg_response_time_hours")
    art_q = np.array([T.quantile_norm(v, b.stats["sig_avg_response_time_hours"], invert=True)
                      if not math.isnan(v) else 0.5 for v in art])
    E = 0.34 * rr + 0.26 * la_q + 0.16 * otw + 0.12 * ic + 0.12 * art_q
    E_med = (0.34 * 0.5 + 0.26 * 0.5 + 0.16 * (b.stats["sig_open_to_work_flag"].get("mean", 0.35))
             + 0.12 * b.stats["sig_interview_completion_rate"].get("p50", 0.5) / 1.0 + 0.12 * 0.5)
    raw = 1.0 + (E - E_med) * cfg.engagement_slope
    return np.clip(raw, cfg.engagement_lo, cfg.engagement_hi)


def risk_modifier(b: Bundle, cfg: ScoringConfig):
    n = b.feats.shape[0]
    ev_ideal = b.col("evidence_cos_ideal")
    # stuffer geometry: lexical >> dense, plus identity/evidence divergence
    qall = bm25mod.query_terms_from_jd()
    qv = np.zeros(b.bm25_W.shape[1], dtype=np.float32)
    for t in qall:
        if t in b.bm25_vocab:
            qv[b.bm25_vocab[t]] = 1.0
    lex = _saturate(np.asarray(b.bm25_W @ qv).ravel(), cfg.bm25_sat_c)
    dense_n = np.array([T.tail_anchor(v, b.stats["evidence_cos_ideal"], cfg.dense_lo_key, cfg.dense_hi_key) for v in ev_ideal])
    geometry = np.maximum(lex - dense_n, 0) + np.clip(b.col("identity_evidence_divergence"), 0, 1) * 0.5
    # archetype resemblance EXCESS over ideal (penalize resembling a negative archetype > the ideal)
    excess = np.zeros(n)
    for label in b.antiprofile_labels:
        ec = b.col(f"evidence_cos_{label}")
        excess = np.maximum(excess, np.maximum(ec - ev_ideal, 0))
    penalty = cfg.risk_w_geometry * geometry + cfg.risk_w_archetype * excess
    rm = np.clip(1.0 - penalty, cfg.risk_floor, 1.0)
    return rm, {"geometry": geometry, "archetype_excess": excess}


# --------------------------------------------------------------------------- #
# Full pipeline                                                                #
# --------------------------------------------------------------------------- #
def score_pool(b: Bundle, cfg: ScoringConfig | None = None, shortlist: np.ndarray | None = None):
    cfg = cfg or ScoringConfig()
    bm25c = _bm25_concept_scores(b)
    dense_support = np.array([T.tail_anchor(v, b.stats["evidence_cos_ideal"], cfg.dense_lo_key, cfg.dense_hi_key)
                              for v in b.col("evidence_cos_ideal")])
    concept_ev = _concept_evidence(b, cfg, bm25c, dense_support)
    crit_sats = {c["id"]: _criterion_sat(b, c, concept_ev, cfg) for c in b.jd["criteria"]
                 if c["criterion_type"] in ("must_have", "nice_to_have")}
    fit, mh_soft, nh_bonus, cm = fuse_and_calibrate(b, cfg, crit_sats)
    tier = fit_tier(fit, cfg)
    eng = engagement_modifier(b, cfg)
    rm, rsub = risk_modifier(b, cfg)
    # corroboration breadth = number of must-have concepts with >=1 verified/demonstrated source
    breadth = np.zeros(b.feats.shape[0])
    for c in CORE_DENSE_CONCEPTS | {"strong_python", "learning_to_rank", "llm_finetune", "distributed_systems"}:
        if c in concept_ev:
            breadth += (concept_ev[c]["sources"] >= 1).astype(float)
    return {
        "fit_score": fit, "fit_tier": tier, "mh_soft": mh_soft, "nh_bonus": nh_bonus,
        "cm": cm, "engagement_modifier": eng, "risk_modifier": rm, "risk_sub": rsub,
        "crit_sats": crit_sats, "concept_ev": concept_ev, "corroboration_breadth": breadth,
    }
