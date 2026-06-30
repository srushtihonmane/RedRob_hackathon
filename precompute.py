#!/usr/bin/env python3
"""precompute.py — the UNBUDGETED offline phase (Goal 8). Builds every artifact the budgeted
ranking step loads. May exceed 5 minutes (embedding the pool); ships as the first repro command:

    python precompute.py --candidates ./candidates.jsonl --artifacts ./artifacts
    python rank.py       --candidates ./candidates.jsonl --out ./submission.csv

Steps: ingest (Goal 1) -> ensure JD vectors (Goal 2) -> representation: embeddings/features/
BM25/snippets (Goal 3) -> risk flags (Goal 5) -> build_manifest (sole root).
"""
import argparse
import os
import subprocess
import sys

from src import build, ingest, risk, scoring
from src import jd_build
from src.runtime_io import write_build_manifest
from src.common import int_to_candidate_id  # noqa: F401  (kept for parity utilities)


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--jd", default="jd")
    ap.add_argument("--expected-n", type=int, default=None,
                    help="assert exact row count (e.g. 100000 for the full pool)")
    args = ap.parse_args()
    gc = _git_commit()

    print("[1/5] ingest ...")
    ingest.ingest_file(args.candidates, args.artifacts, expected_n=args.expected_n, git_commit=gc)

    print("[2/5] JD vectors ...")
    if not (os.path.exists(os.path.join(args.jd, "jd_ideal.npy"))
            and os.path.exists(os.path.join(args.jd, "jd_antiprofile.npy"))):
        jd_build.build(out_dir=args.jd, git_commit=gc)
    else:
        print("      (jd vectors present; skipping)")

    print("[3/5] representation (embeddings/features/bm25/snippets) ... this is the heavy step")
    parquet = os.path.join(args.artifacts, "candidates.parquet")
    ids = os.path.join(args.artifacts, "candidate_ids.npy")
    # REF_DATE via a streaming pre-scan (never materializes the full table).
    ref_date = build.compute_ref_date(ingest.iter_parquet_records(parquet))
    print(f"      REF_DATE = {ref_date} (max last_active + 1 day)")
    build.build_representation_streaming(parquet, ids, args.artifacts, ref_date=ref_date,
                                         jd_query_path=os.path.join(args.jd, "jd_query.json"),
                                         git_commit=gc)

    print("[4/5] risk flags ...")
    bundle = scoring.load_bundle(args.artifacts, jd_dir=args.jd)
    risk.write_risk_artifact(bundle, args.artifacts, git_commit=gc)

    print("[5/5] build_manifest ...")
    write_build_manifest(args.artifacts, args.candidates,
                         os.path.join(args.artifacts, "build_manifest.json"),
                         git_commit=gc, lib_versions={"python": sys.version.split()[0]})
    print(f"Precompute complete -> {args.artifacts}")


if __name__ == "__main__":
    main()
