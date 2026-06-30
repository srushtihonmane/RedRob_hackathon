"""snippets.py — Goal 3 (G7/G8 addenda): row-addressable evidence_snippets sidecar.

Per candidate, a small curated set of criterion-linked narrative spans with STABLE
evidence_ids (the handles Goal 4's evidence trail and Goal 7's renderer reference). Stored
as a JSONL blob + an offsets index so the runtime materializes ONLY the top-100 rows
(O(100) random access; never reopens raw parquet) — Goal 8 D5.

Citeable sources only (contracts C10): anchor literals, skill_assessment_scores (Verified),
career narrative spans (Demonstrated). Raw skills[] tokens are NEVER snippets.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .represent import FrozenInputs

MAX_SNIPPETS = 8


def _sentence_with(text: str, keywords: list[str]) -> str | None:
    for sent in (text or "").replace("\n", " ").split(". "):
        low = sent.lower()
        if any(k in low for k in keywords):
            s = sent.strip()
            return s if len(s) <= 240 else s[:237] + "..."
    return None


def _fmt_yoe(yoe) -> str:
    try:
        return f"{round(float(yoe), 1):g}"
    except (TypeError, ValueError):
        return str(yoe)


def extract_snippets(rec: dict, fz: FrozenInputs) -> list[dict]:
    cid = rec["candidate_id"]
    p = rec["profile"]
    yoe = _fmt_yoe(p.get("years_of_experience"))
    company = p.get("current_company", "")
    snips: list[dict] = [{
        "evidence_id": f"{cid}:anchor",
        "concept": None, "source_type": "anchor", "source_tier": "verified",
        "field_path": "profile",
        "text": f"{p.get('current_title','')}, {yoe} yrs at {company} "
                f"({fz.company_type(company)}).",
        "raw": {"current_title": p.get("current_title", ""), "years_of_experience": yoe,
                "current_company": company, "company_type": fz.company_type(company)},
    }]
    assess = rec["redrob_signals"].get("skill_assessment_scores") or {}
    used_concepts: set[str] = set()
    for cname, cdef in fz.concepts.items():
        # Verified: best assessment for this concept.
        present = [(a, assess[a]) for a in cdef["assessment_skills"] if a in assess]
        if present:
            skill, score = max(present, key=lambda kv: kv[1])
            snips.append({
                "evidence_id": f"{cid}:assess:{cname}",
                "concept": cname, "source_type": "assessment", "source_tier": "verified",
                "field_path": "redrob_signals.skill_assessment_scores",
                "text": f"Scored {score:g} on the {skill} assessment.",
                "raw": {"skill": skill, "score": float(score)},
            })
            used_concepts.add(cname)
        # Demonstrated: a career span mentioning the concept.
        for i, role in enumerate(rec["career_history"]):
            span = _sentence_with(role.get("description", ""), cdef["keywords"])
            if span:
                snips.append({
                    "evidence_id": f"{cid}:career:{cname}:{i}",
                    "concept": cname, "source_type": "career", "source_tier": "demonstrated",
                    "field_path": f"career_history[{i}].description",
                    "text": span,
                })
                used_concepts.add(cname)
                break
        if len(snips) >= MAX_SNIPPETS:
            break
    return snips[:MAX_SNIPPETS]


def build_sidecar(records, fz: FrozenInputs, out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blob_path = out / "evidence_snippets.jsonl"
    off_path = out / "evidence_snippets_offsets.npy"
    offsets = []
    pos = 0
    with open(blob_path, "wb") as f:
        for rec in records:
            obj = {"candidate_id": rec["candidate_id"], "snippets": extract_snippets(rec, fz)}
            line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
            offsets.append((pos, len(line)))
            f.write(line)
            pos += len(line)
    np.save(off_path, np.asarray(offsets, dtype=np.int64))
    return {"n_rows": len(offsets), "bytes": pos}


class SnippetReader:
    """Random-access reader: materialize a single row's snippets without a full load."""
    def __init__(self, out_dir: str):
        out = Path(out_dir)
        self._blob = open(out / "evidence_snippets.jsonl", "rb")
        self._off = np.load(out / "evidence_snippets_offsets.npy")

    def __len__(self) -> int:
        return len(self._off)

    def row(self, i: int) -> dict:
        start, length = int(self._off[i, 0]), int(self._off[i, 1])
        self._blob.seek(start)
        return json.loads(self._blob.read(length).decode("utf-8"))

    def close(self):
        self._blob.close()
