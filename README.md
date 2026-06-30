# Redrob Intelligent Candidate Discovery & Ranking

Ranks the top-100 best-fit candidates from `candidates.jsonl` (100k) for the released Senior-AI-
Engineer JD. The system mirrors the JD's own "v2": **embeddings + hybrid retrieval + corroboration-
first re-ranking**, with explicit trap/honeypot handling — not keyword matching.

## Reproduce the submission (two commands)

```bash
# 1) PRECOMPUTE (unbudgeted, offline) — builds all artifacts (embeddings, features, BM25,
#    snippets, manifests). Needs network ONCE to fetch the embedding model. May exceed 5 min.
python precompute.py --candidates ./candidates.jsonl --artifacts ./artifacts --expected-n 100000

# 2) RANK (budgeted) — loads precomputed artifacts and writes the CSV. CPU-only, no network,
#    <=5 min, <=16 GB. This is the single Stage-3 reproduction command.
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

`rank.py` measured at **~0.1–2 s wall / <1.5 GB peak** (it is a frozen scorer: loads artifacts,
does deterministic array arithmetic, never loads a neural model). Output passes
`validate_submission.py`. Two runs are byte-identical.

## Setup

```bash
pip install -r requirements.runtime.txt       # lean, CPU-only ranking step
pip install -r requirements.precompute.txt     # + embedding stack (precompute / sandbox only)
```

Python 3.10. The ranking step uses only numpy/scipy/pyarrow/pandas/joblib.

## Sandbox / demo (§10.5)

Rank a ≤100-candidate sample end-to-end (builds features on the fly, applies the **shipped frozen
pool statistics** — proving the transforms are pool-frozen):

```bash
python sandbox_rank.py --sample your_sample.jsonl --artifacts ./artifacts --out ranked.csv
```

A self-contained Docker recipe (the canonical Stage-3 image) is in `Dockerfile.runtime`:

```bash
docker build -f Dockerfile.runtime -t redrob-ranker .
docker run --rm -v "$PWD:/work" -w /work redrob-ranker \
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

## Architecture (offline ↔ runtime boundary)

| Phase | What | Where |
|---|---|---|
| **Offline (precompute)** | ingest → JD typed-criteria + ideal/anti-profile vectors → candidate embeddings (identity + query-aware evidence), BM25 (narrative-only), 132 structured features, corroboration channels, evidence snippets, risk flags | `precompute.py`, `src/{ingest,jd_build,represent,build,bm25,snippets,risk}.py` |
| **Runtime (rank)** | load artifacts → recall-first union shortlist → trust-tiered **corroboration-first** criterion satisfaction → must-have soft-AND fusion → tail-anchored fit + tier → bounded engagement/risk modifiers → eligibility gates → top-100 → deterministic reasoning → CSV | `rank.py`, `src/{scoring,risk,select,reason,runtime_io}.py` |

Key design (Stage 4/5 defense):
- **Corroboration-first**: a single self-reported field never satisfies a criterion; ≥1 Verified
  (assessment) or Demonstrated (career + dense-evidence) source is required. Declared-only evidence
  attenuates toward a floor. This sinks keyword-stuffers and the undetectable honeypots alike.
- **Soft-AND must-have spine**: a high dense cosine alone can never inflate fit; the irrelevant
  bulk is pinned near 0 (the JD's "narrow profile").
- **Honeypots**: 3 disjoint logical-impossibility rules (H1/H2/H3, OR-fired) → excluded; top-100
  honeypot exposure ≪ 10%.
- **Engagement/risk** are bounded, separable multipliers ([0.7,1.1] / [0.5,1.0]) applied at
  selection — a perfect-on-paper but inactive candidate is down-weighted; engagement can never
  manufacture a top pick.
- **Determinism**: frozen artifacts + frozen pool stats (REF_DATE, normalization, calibration),
  stable hashing, single-thread BLAS, `candidate_id`-ascending tie-break. No clocks/RNG/network.

## Artifacts & manifests

`build_manifest.json` is the sole root (hash chain + every sub-manifest + `runtime_report.json` +
`submission_sha256`). Each goal emits its own sub-manifest. See `Architecture/` for the full design.

## Tests

```bash
python -m pytest -q tests/        # trap suite, invariants, budget, determinism, official validator
```
