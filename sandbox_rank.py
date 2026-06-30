#!/usr/bin/env python3
"""sandbox_rank.py — the §10.5 Sandbox runtime (Goal 8 D8). Ranks a <=100-candidate sample
END-TO-END for the demo/sandbox. Unlike rank.py it runs the Goal 1->3->5 builders ON THE FLY
(candidate-dependent embedding is permitted HERE), then applies the SHIPPED FROZEN pool-stats
(REF_DATE + normalization_stats from the released artifacts) -- never recomputing them from the
sample. That round-trip actively proves the transforms are pool-frozen.

    python sandbox_rank.py --sample sample100.jsonl --artifacts artifacts --out ranked.csv

Constraints: <=5 min CPU, no hosted-LLM calls. May carry the embedding model (precompute deps).
Shares Goals 4->7 ranking logic byte-for-byte with rank.py.
"""
import argparse
import datetime as dt
import json
import os
import shutil
import sys
import tempfile

from src import build, ingest, scoring, select
from src import reason as reasonmod
from src.common import read_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True, help="<=100-candidate JSONL")
    ap.add_argument("--artifacts", default="artifacts", help="shipped full-pool artifacts (frozen stats)")
    ap.add_argument("--jd", default="jd")
    ap.add_argument("--out", default="ranked.csv")
    args = ap.parse_args()

    # Parse the sample (pure parse_record; Sandbox path).
    recs = [ingest.parse_record(json.loads(l)) for l in open(args.sample, encoding="utf-8") if l.strip()]
    if len(recs) > 100:
        raise SystemExit("sandbox accepts <=100 candidates")
    print(f"[sandbox] parsed {len(recs)} candidates")

    # FROZEN pool statistics from the shipped artifacts (NEVER recomputed from the sample).
    frozen_ref = dt.date.fromisoformat(read_json(os.path.join(args.artifacts, "repr_manifest.json"))["ref_date"])
    print(f"[sandbox] using FROZEN REF_DATE={frozen_ref} from shipped artifacts")

    tmp = tempfile.mkdtemp(prefix="sandbox_")
    try:
        # Build representation for the sample using the frozen REF_DATE (skips pool pre-scan).
        import numpy as np
        ids = np.array([int(r["candidate_id"].split("_")[1]) for r in recs], dtype=np.int32)
        np.save(os.path.join(tmp, "candidate_ids.npy"), ids)
        import pyarrow.parquet as pq
        from src.ingest import ARROW_SCHEMA, _batch_to_arrow
        w = pq.ParquetWriter(os.path.join(tmp, "candidates.parquet"), ARROW_SCHEMA)
        w.write_batch(_batch_to_arrow(recs, 0)); w.close()
        build.build_representation(os.path.join(tmp, "candidates.parquet"),
                                   os.path.join(tmp, "candidate_ids.npy"), tmp,
                                   ref_date=frozen_ref,
                                   jd_query_path=os.path.join(args.jd, "jd_query.json"))
        # Apply SHIPPED frozen normalization_stats for scoring calibration (the proof step).
        shutil.copy(os.path.join(args.artifacts, "normalization_stats.json"),
                    os.path.join(tmp, "normalization_stats.json"))

        bundle = scoring.load_bundle(tmp, jd_dir=args.jd)
        cfg = scoring.ScoringConfig()
        res = scoring.score_pool(bundle, cfg)
        rfn, _ = reasonmod.build_reasoner(bundle, res, cfg)
        n_sel = min(100, len(recs))
        # select() takes top-100; with <100 candidates it returns all eligible ranked.
        from src import risk as riskmod
        sel = select.select(bundle, res, reasoning_fn=rfn)
        select.write_submission(sel, args.out)
        print(f"[sandbox] wrote {args.out} | {len(sel.rows)} ranked rows")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
