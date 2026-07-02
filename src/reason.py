"""reason.py — Goal 7: deterministic evidence->narrative compiler for the reasoning column.

A PURE PROJECTION of the already-decided evidence contract (fit bundle + Goal 3 snippets):
never reads raw candidate data, never selects evidence beyond what the scorer surfaced.
7A Planner role-tags/orders the evidence into an IR; 7B Realizer renders a rank-band-toned
string (<=~240 char). Citeable whitelist only (contracts C10): anchor literals, assessment
skill+score (Verified), career narrative spans (Demonstrated), redrob_signals. Raw skills[]
tokens are NEVER cited. Every rendered clause binds to an evidence_id (provenance).
"""
from __future__ import annotations

import math

from .common import stable_unit_hash
from .scoring import CRITERION_CONCEPTS

CONCEPT_PHRASE = {
    "retrieval_embeddings": "embeddings-based retrieval",
    "vector_db": "vector search / hybrid retrieval",
    "ranking_eval": "ranking & evaluation",
    "learning_to_rank": "learning-to-rank",
    "strong_python": "Python",
    "llm_finetune": "LLM fine-tuning",
    "distributed_systems": "distributed systems at scale",
    "nlp_ir": "NLP / IR",
}
LEAD_VARIANTS = ["{t}, {y} yrs at {c}", "{t} ({y} yrs, {c})", "{t} at {c}, {y} yrs"]
MAX_LEN = 240


def _short_label(desc: str, limit: int = 40) -> str:
    """Shorten a criterion description at a WORD boundary (no mid-word cuts)."""
    desc = (desc or "").rstrip(".")
    if len(desc) <= limit:
        return desc
    cut = desc[:limit]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(",;: ")


def _snip_for_concept(snippets, concept):
    verified = [s for s in snippets if s.get("concept") == concept and s["source_tier"] == "verified"]
    demo = [s for s in snippets if s.get("concept") == concept and s["source_tier"] == "demonstrated"]
    return (verified or demo or [None])[0]


def build_reasoner(bundle, res, cfg):
    """Return reasoning_fn(row_index, ctx) -> str and a provenance dict {candidate_id: [ids]}."""
    crit_by_id = {c["id"]: c for c in bundle.jd["criteria"]}
    mh_ids = [c["id"] for c in bundle.jd["criteria"] if c["criterion_type"] == "must_have"]
    nh_ids = [c["id"] for c in bundle.jd["criteria"] if c["criterion_type"] == "nice_to_have"]
    iw = cfg.importance_weights if cfg else {"critical": 1.0, "high": 0.6, "medium": 0.35, "low": 0.15}
    provenance: dict = {}

    def plan(i, ctx):
        snippets = ctx["snippets"]
        anchor = next((s for s in snippets if s["source_type"] == "anchor"), None)
        # strengths: satisfied criteria ranked by fit_contribution = w * sat
        cand = []
        for cid in mh_ids + nh_ids:
            sat = float(res["crit_sats"][cid][i]) if cid in res["crit_sats"] else 0.0
            if sat < 0.3:
                continue
            w = iw[crit_by_id[cid]["importance"]]
            concept = (CRITERION_CONCEPTS.get(cid) or [None])[0]
            snip = _snip_for_concept(snippets, concept) if concept else None
            cand.append({"cid": cid, "sat": sat, "contribution": w * sat, "concept": concept,
                         "snip": snip,
                         "jd": CONCEPT_PHRASE.get(concept) or _short_label(crit_by_id[cid]["description"])})
        cand.sort(key=lambda d: -d["contribution"])
        strengths = cand[:2]
        # concern ladder (deterministic, first applicable)
        concern = None
        if ctx["filler"]:
            concern = {"text": "below the relevant tier — included as filler", "eid": None}
        else:
            notice = bundle.col("sig_notice_period_days")[i]
            weakest = min(((cid, float(res["crit_sats"][cid][i])) for cid in mh_ids
                           if cid in res["crit_sats"]), key=lambda kv: kv[1], default=None)
            if not math.isnan(notice) and notice > 60:
                concern = {"text": f"notice period {int(notice)} days", "eid": None}
            elif weakest and weakest[1] < 0.5:
                c = (CRITERION_CONCEPTS.get(weakest[0]) or [None])[0]
                concern = {"text": f"lighter demonstrated evidence of {CONCEPT_PHRASE.get(c, 'the core stack')}",
                           "eid": None}
        return {"anchor": anchor, "strengths": strengths, "concern": concern, "tier": ctx["tier"]}

    def realize(ir, cid):
        eids = []
        a = ir["anchor"]
        raw = (a or {}).get("raw", {})
        lead = LEAD_VARIANTS[int(stable_unit_hash(cid) * len(LEAD_VARIANTS))]
        s1 = lead.format(t=raw.get("current_title", "Candidate"), y=raw.get("years_of_experience", "?"),
                         c=raw.get("current_company", ""))
        if a:
            eids.append(a["evidence_id"])
        # strengths with JD link + citation (dedupe identical clauses / repeated snippets)
        clauses, seen_clause, seen_eid = [], set(), set()
        for st in ir["strengths"]:
            snip = st["snip"]
            if snip and snip["source_tier"] == "verified" and "raw" in snip:
                score = snip["raw"].get("score")
                score_s = f"{round(float(score))}" if score is not None else ""
                clause = f"{snip['raw'].get('skill','')} assessment {score_s}".strip()
                eid = snip["evidence_id"]
            elif snip and snip["source_tier"] == "demonstrated":
                clause = f"{st['jd']} ({raw.get('current_company','')})"
                eid = snip["evidence_id"]
            else:
                clause, eid = st["jd"], None
            if clause in seen_clause or (eid and eid in seen_eid):
                continue
            seen_clause.add(clause)
            if eid:
                seen_eid.add(eid); eids.append(eid)
            clauses.append(clause)
        if clauses:
            s1 = s1 + " — " + "; ".join(clauses[:2])
        # tone + concern (S2)
        tier = ir["tier"]
        s2 = ""
        if ir["concern"]:
            lead_word = {"Elite": "Note", "Strong": "Caveat", "Plausible": "Concern",
                         "Filler": "Limitation"}.get(tier, "Note")
            s2 = f" {lead_word}: {ir['concern']['text']}."
            if ir["concern"]["eid"]:
                eids.append(ir["concern"]["eid"])
        text = (s1.rstrip(". ") + "." + s2).strip()
        text = " ".join(text.split())
        if len(text) > MAX_LEN:
            text = text[:MAX_LEN - 1].rstrip() + "."
        return text, eids

    def reasoning_fn(i, ctx):
        ir = plan(i, ctx)
        cid = None
        a = ir["anchor"]
        if a:
            cid = a["evidence_id"].split(":")[0]
        text, eids = realize(ir, cid or "CAND_0000000")
        provenance[cid] = eids
        return text

    return reasoning_fn, provenance
