"""make_sample.py — build a deterministic, stratified ~1.5k candidate sample for fast,
representative end-to-end validation (full 100k embed deferred).

Strata (so every code path is exercised and the top-100 is meaningful):
  - real honeypots (H1/H2/H3 logical impossibilities)         cap 50
  - real keyword-positive (ranking/search/retrieval/eval...)  cap 300
  - real consulting-entire-career                             cap 80
  - random background (deterministic by stable_hash)          fill to N
  - the 10 synthetic adversarial fixtures (known-answer)      appended

Determinism: selection uses stable_hash (no RNG). Output preserves real-source order, then
appends the synthetic fixtures. Usage:
  python scripts/make_sample.py --candidates candidates.jsonl --out data/sample/sample.jsonl --n 1500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import stable_hash  # noqa: E402
from tests import fixtures  # noqa: E402

CONSULTING = {"infosys", "wipro", "tcs", "tata consultancy services", "capgemini", "hcl",
              "mindtree", "accenture", "cognizant", "tech mahindra", "mphasis"}
KW = ("ranking", "search", "recommend", "retrieval", "embedding", "vector", "ndcg", "mrr",
      "learning to rank", "information retrieval", "relevance", "hybrid search")


def _summed_tenure_yrs(rec) -> float:
    return sum((r.get("duration_months") or 0) for r in rec.get("career_history") or []) / 12.0


def _is_honeypot(rec) -> bool:
    yoe = rec.get("profile", {}).get("years_of_experience") or 0.0
    sm = _summed_tenure_yrs(rec)
    if yoe - sm > 2.0 or sm - yoe > 2.0:
        return True
    z = sum(1 for s in rec.get("skills") or []
            if s.get("proficiency") in ("advanced", "expert") and (s.get("duration_months") == 0))
    return z >= 2


def _is_keyword_positive(rec) -> bool:
    parts = [rec.get("profile", {}).get("summary", ""), rec.get("profile", {}).get("headline", "")]
    for r in rec.get("career_history") or []:
        parts.append(r.get("title", ""))
        parts.append(r.get("description", ""))
    blob = " ".join(parts).lower()
    return any(k in blob for k in KW)


def _is_consulting_only(rec) -> bool:
    comps = [r.get("company", "").strip().lower() for r in rec.get("career_history") or []]
    return bool(comps) and all(c in CONSULTING for c in comps)


def build_sample(candidates_path, out_path, n=1500, bg_cut=40):
    caps = {"honeypot": 50, "keyword": 300, "consulting": 80}
    counts = {k: 0 for k in caps}
    kept: list[tuple[int, str]] = []          # (source_index, raw_line) for priority strata
    background: list[tuple[int, int, str]] = []  # (hashval, source_index, raw_line)
    seen: set[str] = set()

    with open(candidates_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = rec.get("candidate_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            bucket = None
            if _is_honeypot(rec) and counts["honeypot"] < caps["honeypot"]:
                bucket = "honeypot"
            elif _is_keyword_positive(rec) and counts["keyword"] < caps["keyword"]:
                bucket = "keyword"
            elif _is_consulting_only(rec) and counts["consulting"] < caps["consulting"]:
                bucket = "consulting"
            if bucket:
                counts[bucket] += 1
                kept.append((idx, line))
            elif stable_hash(cid) % 1000 < bg_cut:
                background.append((stable_hash(cid), idx, line))

    # Fill background deterministically by stable_hash order to reach N total reals.
    need = max(0, n - len(kept))
    background.sort()
    bg = [(idx, line) for _h, idx, line in background[:need]]
    reals = sorted(kept + bg, key=lambda t: t[0])  # restore source order

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as g:
        for _idx, line in reals:
            g.write(line if line.endswith("\n") else line + "\n")
        for rec in fixtures.all_fixtures().values():   # appended known-answer fixtures
            g.write(json.dumps(rec) + "\n")

    total = len(reals) + len(fixtures.ALL_FIXTURES)
    print(f"sample written: {out_path}")
    print(f"  strata: {counts} | background={len(bg)} | fixtures={len(fixtures.ALL_FIXTURES)}")
    print(f"  total rows: {total}")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--out", default="data/sample/sample.jsonl")
    ap.add_argument("--n", type=int, default=1500)
    args = ap.parse_args()
    build_sample(args.candidates, args.out, args.n)


if __name__ == "__main__":
    main()
