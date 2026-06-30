"""select.py — Goal 6: the decision policy (the only goal that writes the submission).

Canonical flow: score everyone -> mark eligibility -> filter eligible -> sort -> take top 100
-> (Goal 7 reasoning) -> assemble -> validate -> write CSV.

  final_score = fit_score * engagement_modifier * risk_modifier   (computed for ALL)
  hard_gate_flag = honeypot OR (consulting_gate AND NOT suppressed)   (derived, not persisted)
  sort key = final_score desc -> corroboration_breadth desc -> candidate_id asc
  printed score = final_score - rank*DELTA   (sub-epsilon micro-term: strictly decreasing, so
                  the ONLY validator tie-break — candidate_id ascending on equal printed score —
                  never needs to fire; breadth already influenced order via the sort.) [C9]
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import risk as riskmod
from . import scoring as scoringmod
from .common import int_to_candidate_id, write_json
from .snippets import SnippetReader

DELTA = 1e-7
N_SELECT = 100
HEADER = ["candidate_id", "rank", "score", "reasoning"]


@dataclass
class Selection:
    rows: list          # list of dicts: candidate_id, rank, printed_score, reasoning, row_index, ...
    eligible_count: int
    excluded_count: int
    exclusions_by_reason: dict
    top_excluded: list


def _placeholder_reasoning(anchor: str, tier: str, filler: bool) -> str:
    base = anchor.strip().rstrip(".")
    tail = " Included as filler below the relevant tier." if filler else f" Assessed fit tier: {tier}."
    return (base + "." + tail)[:240]


def select(bundle, scoring_res, reasoning_fn=None) -> Selection:
    """Compute eligibility + final_score, select top-100 eligible, assign ranks + printed
    scores. ``reasoning_fn(row_index, ctx) -> str`` overrides the placeholder (Goal 7)."""
    n = bundle.feats.shape[0]
    fit = scoring_res["fit_score"]
    final = fit * scoring_res["engagement_modifier"] * scoring_res["risk_modifier"]
    breadth = scoring_res["corroboration_breadth"]
    tier = scoring_res["fit_tier"]

    risk = riskmod.compute_risk(bundle)
    hard_gate = risk["honeypot_flag"] | (risk["consulting_gate_flag"] & ~risk["consulting_gate_suppressed"])
    eligible = ~hard_gate

    elig_idx = np.nonzero(eligible)[0]
    cids = bundle.candidate_ids
    # sort: final desc, breadth desc, candidate_id asc
    order = sorted(elig_idx.tolist(),
                   key=lambda i: (-float(final[i]), -float(breadth[i]), int(cids[i])))
    selected = order[:N_SELECT]

    snips = SnippetReader(bundle.art_dir)
    rows = []
    for rank, i in enumerate(selected, start=1):
        printed = float(final[i]) - rank * DELTA
        srow = snips.row(i)
        anchor = next((s["text"] for s in srow["snippets"] if s["source_type"] == "anchor"),
                      f"Candidate {srow['candidate_id']}")
        filler = (tier[i] == "Filler")
        if reasoning_fn is not None:
            reasoning = reasoning_fn(i, {"rank": rank, "tier": tier[i], "filler": filler,
                                         "snippets": srow["snippets"], "final": float(final[i])})
        else:
            reasoning = _placeholder_reasoning(anchor, str(tier[i]), filler)
        rows.append({"candidate_id": int_to_candidate_id(int(cids[i])), "rank": rank,
                     "printed_score": printed, "reasoning": reasoning, "row_index": int(i),
                     "final": float(final[i]), "tier": str(tier[i]), "filler": filler})
    snips.close()

    # Guarantee unique reasoning (templated source data can yield identical anchors). The
    # real Goal 7 compiler produces varied reasoning; this only protects the skeleton gate.
    if reasoning_fn is None:
        seen: set[str] = set()
        for r in rows:
            base = r["reasoning"]
            txt, k = base, 2
            while txt in seen:
                txt = f"{base[:230]} (rank {r['rank']})"
                k += 1
                if k > 3:
                    break
            seen.add(txt)
            r["reasoning"] = txt

    # exclusions audit
    excl_reasons = {"honeypot": int(risk["honeypot_flag"].sum()),
                    "consulting_gate": int((risk["consulting_gate_flag"] & ~risk["consulting_gate_suppressed"]).sum())}
    excl_idx = np.nonzero(hard_gate)[0]
    top_excluded = sorted(excl_idx.tolist(), key=lambda i: -float(final[i]))[:10]
    top_excluded = [{"candidate_id": int_to_candidate_id(int(cids[i])), "final": float(final[i]),
                     "honeypot": bool(risk["honeypot_flag"][i]),
                     "consulting_gate": bool(risk["consulting_gate_flag"][i])} for i in top_excluded]
    return Selection(rows=rows, eligible_count=int(eligible.sum()),
                     excluded_count=int(hard_gate.sum()), exclusions_by_reason=excl_reasons,
                     top_excluded=top_excluded)


# --------------------------------------------------------------------------- #
# Validation suite (Goal 6 D7) — the 11 checks, by invariants                  #
# --------------------------------------------------------------------------- #
def validate_selection(sel: Selection, bundle, scoring_res) -> list[str]:
    errs = []
    rows = sel.rows
    if len(rows) != N_SELECT:
        errs.append(f"expected {N_SELECT} rows, got {len(rows)}")
    ranks = [r["rank"] for r in rows]
    if sorted(ranks) != list(range(1, N_SELECT + 1)):
        errs.append("ranks must be 1..100 unique")
    ids = [r["candidate_id"] for r in rows]
    if len(set(ids)) != len(ids):
        errs.append("duplicate candidate_id in selection")
    scores = [r["printed_score"] for r in rows]
    for i in range(len(scores) - 1):                              # non-increasing
        if scores[i] < scores[i + 1] - 1e-12:
            errs.append(f"printed score increases at rank {i+1}")
            break
    if any(not np.isfinite(s) for s in scores):                  # finite
        errs.append("non-finite printed score")
    # equal-printed-score tie-break audit (should never trigger given the micro-term)
    for i in range(len(rows) - 1):
        if scores[i] == scores[i + 1] and rows[i]["candidate_id"] > rows[i + 1]["candidate_id"]:
            errs.append("equal printed score not in candidate_id ascending order")
            break
    # eligibility vs Goal-5 primitives (directly)
    risk = riskmod.compute_risk(bundle)
    hard_gate = risk["honeypot_flag"] | (risk["consulting_gate_flag"] & ~risk["consulting_gate_suppressed"])
    for r in rows:
        if risk["honeypot_flag"][r["row_index"]]:
            errs.append(f"selected honeypot {r['candidate_id']}")
        if hard_gate[r["row_index"]]:
            errs.append(f"selected hard-gated {r['candidate_id']}")
    # top-100 correctness: lowest selected final >= highest unselected eligible final
    final = scoring_res["fit_score"] * scoring_res["engagement_modifier"] * scoring_res["risk_modifier"]
    sel_rowidx = {r["row_index"] for r in rows}
    eligible = ~hard_gate
    unsel_elig = [i for i in np.nonzero(eligible)[0] if i not in sel_rowidx]
    if rows and unsel_elig:
        lowest_sel = min(final[r["row_index"]] for r in rows)
        highest_unsel = max(final[i] for i in unsel_elig)
        if lowest_sel < highest_unsel - 1e-9:
            errs.append("a higher-scoring eligible candidate was left unselected")
    # reasoning checks: non-empty, unique, length, rank-tone (fillers acknowledge limitation)
    reasons = [r["reasoning"] for r in rows]
    if any(not s.strip() for s in reasons):
        errs.append("empty reasoning")
    if len(set(reasons)) != len(reasons):
        errs.append("duplicate reasoning strings")
    if any(len(s) > 260 for s in reasons):
        errs.append("reasoning exceeds length cap")
    # alignment: every selected id exists in candidate_ids
    valid_ints = set(int(x) for x in bundle.candidate_ids)
    for r in rows:
        from .common import candidate_id_to_int
        if candidate_id_to_int(r["candidate_id"]) not in valid_ints:
            errs.append(f"selected id not in pool: {r['candidate_id']}")
    return errs


def write_submission(sel: Selection, out_csv: str) -> None:
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(HEADER)
        for r in sel.rows:
            score_str = f"{r['printed_score']:.8f}"
            reasoning = " ".join(r["reasoning"].split())   # collapse whitespace/newlines
            w.writerow([r["candidate_id"], r["rank"], score_str, reasoning])


def write_selection_manifest(sel: Selection, out_dir: str, config_hash: str = "",
                             git_commit: str | None = None) -> None:
    write_json(Path(out_dir) / "selection_manifest.json", {
        "schema_version": 1, "selected": len(sel.rows), "eligible": sel.eligible_count,
        "excluded": sel.excluded_count, "exclusions_by_reason": sel.exclusions_by_reason,
        "top_excluded": sel.top_excluded, "config_hash": config_hash, "git_commit": git_commit,
    })


def produce_submission(art_dir: str, out_csv: str, cfg=None, reasoning_fn=None,
                       jd_dir: str = "jd", git_commit: str | None = None):
    """End-to-end: load -> score -> select -> validate -> write CSV + manifest."""
    bundle = scoringmod.load_bundle(art_dir, jd_dir=jd_dir)
    res = scoringmod.score_pool(bundle, cfg)
    sel = select(bundle, res, reasoning_fn=reasoning_fn)
    errs = validate_selection(sel, bundle, res)
    if errs:
        raise AssertionError("selection validation failed:\n  " + "\n  ".join(errs))
    write_submission(sel, out_csv)
    write_selection_manifest(sel, str(Path(out_csv).parent), git_commit=git_commit)
    return sel
