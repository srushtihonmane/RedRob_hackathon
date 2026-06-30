# artifacts.md — Artifact inventory (derived index)

> Derived index of every persisted artifact: producer goal, format/shape, and consumers.
> When this disagrees with a goal file on a *schema*, see `contracts.md` (authoritative);
> this file is a navigation aid.

## Offline / precompute artifacts

| Artifact | Producer | Format / shape | Consumed by |
|---|---|---|---|
| `candidates.parquet` | Goal 1 | nested `list<struct>` + `map`, 1 row/candidate, source order | Goal 3 (iterate in order) |
| `candidate_ids.npy` | Goal 1 | int32 `[N]`, 7-digit part | all goals (alignment key) |
| `ingest_manifest.json` | Goal 1 | json (rows, quarantine, sha256, schema fp, versions, git) | Goal 8 root manifest |
| `quarantine.jsonl` (opt) | Goal 1 | jsonl (line# + error) | human review gate |
| `jd_query.json` | Goal 2 | json (typed criteria, lexical groups, narratives, labels) | Goals 3, 4, 5, 7 |
| `jd_ideal.npy` | Goal 2 | f32 `[1×384]` L2-norm | Goal 3 (role_relevance), Goal 4 |
| `jd_antiprofile.npy` | Goal 2 | f32 `[A×384]` L2-norm, row=label order | Goal 3 (archetype cosines), Goal 4 |
| `jd_manifest.json` | Goal 2 | json (sha256, model+rev, dim, prefix, git) | Goal 8 root manifest |
| `embeddings_identity.npy` | Goal 3 | f32 `[N×384]` L2-norm | Goal 4 (retrieval, archetype) |
| `embeddings_evidence.npy` | Goal 3 | f32 `[N×384]` L2-norm | Goal 4 (retrieval, satisfaction) |
| `features.npy` | Goal 3 | f32 `[N×F]` (NaN+_present) | Goals 4, 5 |
| `features.parquet` | Goal 3 | named cols, native nulls (debug) | inspection/audit |
| `feature_manifest.json` | Goal 3 | json (per-col schema, registries, archetype order) | Goals 4, 5, 8 |
| `bm25_index/` | Goal 3 | CSR `.npz` + `idf.npy` + `doclen.npy` + `vocab.json` | Goal 4 (lexical channel) |
| `normalization_stats.json` | Goal 3 | per-feature full-pool percentiles | Goal 4 (transforms) |
| `evidence_snippets` sidecar | Goal 3 | row-addressable blob + offset index, stable `evidence_id` | Goals 4, 7 |
| `repr_manifest.json` | Goal 3 | json (sha256 all + upstream, registries, model rev) | Goal 8 root manifest |
| pool distribution stats (BM25/dense/CE) | Goal 4 offline | json/npy | Goal 4 runtime (tail-anchored calibration) |
| `ce_features.npy` (if enabled) | Goal 4 offline | f32 `[N×C]`, 2-D row-aligned | Goal 4 runtime (lookup only) |
| `risk_flags` artifact | Goal 5 | per-candidate primitives + evidence, aligned | Goal 6 |
| `company_table.json`, `concept_registry` | Goal 3 (curated) | json | Goals 3, 4, 5 |

## Runtime / output artifacts

| Artifact | Producer | Format | Notes |
|---|---|---|---|
| fit bundle (per shortlisted candidate) | Goal 4 runtime | in-memory / sidecar | see `contracts.md` C7 |
| `submission.csv` | Goal 6 | CSV `candidate_id,rank,score,reasoning` | the deliverable; passes `validate_submission.py` |
| `selection_manifest.json` | Goal 6 | json (counts, exclusions, top-excluded) | gate-decision defense |
| `reasoning` column | Goal 7 | text, ≤~240 char, QUOTE_ALL | folded into submission.csv |
| `reasoning_provenance.json` | Goal 7 | clause→`evidence_id` | grounding audit |
| `reasoning_audit_report.json` | Goal 7 | flagged rows | human review |
| `runtime_report.json` | Goal 8 | per-stage timings, peak RSS, bytes, flags | Stage-3/5 evidence |
| `build_manifest.json` | Goal 8 | **sole root** — hash chain + all sub-manifests | organizer entry point |

## Entrypoints (Goal 8 D9)
- `precompute.py --candidates candidates.jsonl …` → builds all offline artifacts (unbudgeted).
- `rank.py --artifacts … --out submission.csv` → the budgeted ranking step (≤5 min).
- Competition Runtime Dockerfile (= the Stage-3 reproduction image).
- Two pinned `requirements` files (lean runtime set; precompute/sandbox heavier set).
