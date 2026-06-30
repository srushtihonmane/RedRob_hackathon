#!/usr/bin/env python3
"""rank.py — the BUDGETED ranking step (Goal 8). Loads precomputed artifacts and produces the
submission CSV with deterministic retrieve/filter/score/rank/reason/CSV. CPU-only, no network,
NO neural model (Runtime ML Policy). Single command (Stage-3 reproduction):

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Defaults to the full-pool artifacts in ./artifacts (built by precompute.py). Use --artifacts
to point elsewhere (e.g. the sample).
"""
import os
# Determinism: pin single-thread BLAS BEFORE numpy imports (Goal 8 D6).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import argparse
import subprocess
import sys

from src import scoring, select
from src import reason as reasonmod
from src.runtime_io import RuntimeReport, validate_and_bind, write_build_manifest
from src.common import sha256_file


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--jd", default="jd")
    ap.add_argument("--no-bind", action="store_true", help="skip artifact<->pool hash binding")
    args = ap.parse_args()

    rep = RuntimeReport()
    # Phase 1: cheap validation + binding (loud, ~100ms) BEFORE big matrices.
    with rep.stage("validate_and_bind"):
        cpath = None if args.no_bind else args.candidates
        validate_and_bind(args.artifacts, cpath)
    # Phase 2: load matrices + score + select + reason + write.
    with rep.stage("load_artifacts"):
        bundle = scoring.load_bundle(args.artifacts, jd_dir=args.jd)
    cfg = scoring.ScoringConfig()
    with rep.stage("score_pool"):
        res = scoring.score_pool(bundle, cfg)
    with rep.stage("reason_and_select"):
        rfn, prov = reasonmod.build_reasoner(bundle, res, cfg)
        sel = select.select(bundle, res, reasoning_fn=rfn)
        errs = select.validate_selection(sel, bundle, res)
        if errs:
            print("VALIDATION FAILED:\n  " + "\n  ".join(errs), file=sys.stderr)
            sys.exit(1)
    with rep.stage("write_csv"):
        select.write_submission(sel, args.out)
        select.write_selection_manifest(sel, str(os.path.dirname(args.out) or "."),
                                        git_commit=_git_commit())

    report = rep.finalize(os.path.join(args.artifacts, "runtime_report.json"))
    write_build_manifest(args.artifacts, args.candidates,
                         os.path.join(args.artifacts, "build_manifest.json"),
                         submission_sha256=sha256_file(args.out), git_commit=_git_commit(),
                         lib_versions={"python": sys.version.split()[0]})
    print(f"Wrote {args.out} | {len(sel.rows)} rows | "
          f"{report['total_wall_s']:.2f}s wall | {report['peak_rss_mb']:.0f}MB peak RSS")


if __name__ == "__main__":
    main()
