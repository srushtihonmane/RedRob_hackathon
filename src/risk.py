"""risk.py — Goal 5: the hard, decisive RISK layer (binary detection only; never scores/sorts).

Owns honeypot detection (H1/H2/H3, OR-fired — logical impossibilities) and the one
deterministically-detectable hard gate (consulting_entire_career) with its product-stint
override. Emits PRIMITIVE FACTS only; Goal 6 composes the decision (hard_gate_flag).

Every rule is a per-candidate logical test over Goal 3 features -> reusable + Sandbox-safe
(contracts.md C11). The observed pool counts (e.g. 69 honeypots) are an offline AUDIT
statistic, not part of the per-candidate function (Goal 5 D6).

  H1: yoe - summed_tenure > 2.0      (experience far exceeds career tenure)
  H2: summed_tenure - yoe > 2.0      (career tenure far exceeds experience)
  H3: >=2 advanced/expert skills with duration_months == 0
  consulting_gate: every stint services-consulting AND no product stint (override: any_product_stint)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import write_json, sha256_file

H_THRESHOLD = 2.0


def compute_risk(bundle) -> dict:
    """Vectorized risk primitives over the whole pool, from the feature matrix."""
    col = bundle.col
    dy = col("yoe_minus_summed_tenure")
    h1 = dy > H_THRESHOLD
    h2 = dy < -H_THRESHOLD
    h3 = col("expert_skill_zero_duration_count") >= 2
    honeypot = h1 | h2 | h3
    # Gate fires only on entire-career consulting (which already implies no product stint).
    consulting_gate = col("is_consulting_entire_career") > 0.5
    # Suppression is the explainability case: consulting-DOMINANT career rescued by a product
    # stint (the JD's "currently at a consulting firm but have prior product experience").
    any_product = col("any_product_stint") > 0.5
    consulting_dominant = col("frac_career_services") >= 0.5
    consulting_suppressed = consulting_dominant & any_product
    return {
        "honeypot_flag": honeypot, "h1": h1, "h2": h2, "h3": h3,
        "consulting_gate_flag": consulting_gate,
        "consulting_gate_suppressed": consulting_suppressed,
    }


def risk_flags(feat_row: dict) -> dict:
    """Per-candidate risk primitives from a feature dict (Sandbox path)."""
    dy = feat_row["yoe_minus_summed_tenure"]
    h1 = dy > H_THRESHOLD
    h2 = dy < -H_THRESHOLD
    h3 = feat_row["expert_skill_zero_duration_count"] >= 2
    consulting_entire = feat_row["is_consulting_entire_career"] > 0.5
    any_product = feat_row["any_product_stint"] > 0.5
    consulting_dominant = feat_row["frac_career_services"] >= 0.5
    reasons = []
    if h1: reasons.append(("H1", "yoe_minus_summed_tenure", float(dy)))
    if h2: reasons.append(("H2", "summed_tenure_minus_yoe", float(-dy)))
    if h3: reasons.append(("H3", "expert_skill_zero_duration_count",
                           float(feat_row["expert_skill_zero_duration_count"])))
    return {
        "honeypot_flag": bool(h1 or h2 or h3), "honeypot_reasons": reasons,
        "consulting_gate_flag": bool(consulting_entire),
        "consulting_gate_suppressed": bool(consulting_dominant and any_product),
    }


def stuffer_suspect(bundle, scoring_res) -> np.ndarray:
    """Runtime diagnostic only (Goal 5 D4): high lexical-dense divergence AND zero Tier-1/2
    corroboration AND title/domain mismatch. NEVER used for exclusion."""
    geom = scoring_res["risk_sub"]["geometry"]
    breadth = scoring_res["corroboration_breadth"]
    return (geom > 0.3) & (breadth < 1)


def write_risk_artifact(bundle, out_dir: str, git_commit: str | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    r = compute_risk(bundle)
    arr = np.stack([r["honeypot_flag"], r["h1"], r["h2"], r["h3"],
                    r["consulting_gate_flag"], r["consulting_gate_suppressed"]], axis=1).astype(np.int8)
    np.save(out / "risk_flags.npy", arr)
    summary = {
        "schema_version": 1, "n_rows": int(arr.shape[0]),
        "columns": ["honeypot_flag", "h1", "h2", "h3", "consulting_gate_flag",
                    "consulting_gate_suppressed"],
        "counts": {k: int(r[k].sum()) for k in ("honeypot_flag", "h1", "h2", "h3",
                                                "consulting_gate_flag", "consulting_gate_suppressed")},
        "risk_flags_sha256": sha256_file(out / "risk_flags.npy"),
        "git_commit": git_commit,
    }
    write_json(out / "risk_manifest.json", summary)
    return summary
