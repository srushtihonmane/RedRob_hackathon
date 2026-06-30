# runtime.md — Runtime envelope, ML policy, determinism, reproduction (authoritative, cross-cutting)

> **Read every session.** This is the cross-cutting runtime contract that binds all goals.
> Full design rationale and profiling numbers live in `Goal8.md`.

## R1 — The hard envelope (the ONLY hard runtime limits)
The **ranking step** (`rank.py`, the budgeted phase) must satisfy, and nothing more is mandated:
- **CPU only** (no GPU)
- **No network**
- **≤ 5 min** wall-clock
- **≤ 16 GB** RAM (exceeding 16 GB = DQ)
- **≤ 5 GB** disk

Measured headroom (Goal 8 profiling, conservative hardware): ~1–2 s cold wall-clock, ~1.3 GB peak. The budget is comfortable; correctness and determinism are the real constraints.

## R2 — Runtime ML Policy (behavioral, not a package allowlist)
The ranking step is a **frozen scorer, not an ML pipeline.** It must **not instantiate, load, download, or execute any neural model or inference engine** — transformers, sentence-transformers, Torch, ONNX, GGUF, or any other — to compute candidate-dependent outputs at runtime. **ALL** neural computation (embedding generation, reranking, calibration fitting, model inference) happens **offline**. At runtime the system only **loads precomputed artifacts** (embeddings, indexes, feature matrices, lookup tables, calibration params) and performs deterministic **retrieval / filter / score / rank / reason / CSV**.

**Dependencies are constrained by behavior, not a fixed package list.** Any CPU-only, offline, deterministic library is permitted (numpy, scipy, pyarrow, pandas, joblib, …). A lean, model-free dependency set is **preferred** (smaller, more deterministic Stage-3 image) but **not mandated**. *(This supersedes any "numpy/scipy only" phrasing in the goal files.)*

**Decision rules** (resolve anything unspecified): when in doubt push computation into **precompute, not runtime**; between two rule-compliant implementations prefer the **simpler** one. Runtime = a lightweight artifact loader + scorer.

## R3 — Determinism (the primary correctness guarantee)
- No randomness, clocks, locale dependence, or runtime-derived statistics.
- Stable hashing only (`zlib.crc32`/`hashlib`, never salted `hash()`); `PYTHONHASHSEED=0`.
- Deterministic collection ordering (`np.unique` for unions, never set-iteration order).
- Frozen artifacts + frozen pool statistics (REF_DATE, normalization, calibration anchors).
- Deterministic tie-break hierarchy ending in `candidate_id` ascending (see `contracts.md` C9).
- Single-thread BLAS in the canonical reproduction image — a hardening choice that removes residual float-reduction jitter, **not** the primary guarantee (that's fixed artifacts + deterministic logic + deterministic tie-breaking).

## R4 — Two runtimes (shared ranking logic, separate dependency sets)
- **Competition Runtime** — the artifact-only ranking step (the submission + Stage-3 reproduction). Obeys R1 + R2. Deps = lean model-free CPU/offline set.
- **Sandbox Runtime** — end-to-end demo on a ≤100 arbitrary sample (§10.5). Constraints = ≤5 min CPU, no hosted-LLM calls. May carry models/heavier deps. Runs the Goal 1→3→5 builders on the sample (candidate-dependent embedding permitted HERE only), then applies **shipped frozen pool-stats** — never recomputes them from the sample (this proves the transforms are pool-frozen). See `contracts.md` C11.
- They **share ranking logic (Goals 4→7 byte-identical)** but **not dependency sets**.

## R5 — Two-phase load order (loud, cheap failure)
1. **Validation load** — read the *small* artifacts first (manifests, frozen stats, `jd_query`, `candidate_ids`); run **hash + alignment + shape checks** + the artifact↔pool binding assertion.
2. **Only on success**, load the large matrices (full-load densely-scanned embeddings ×2 + features; mmap sparse/selective: BM25 CSC, the row-addressable snippet sidecar) and execute.

A broken/misaligned/stale artifact fails in ~100 ms, before hundreds of MB load. Competition Runtime **never loads raw `candidates.parquet`/JSONL** (snippets + features replace it).

## R6 — Reproduction (Stage 3)
- **Two-command repro:** `precompute.py --candidates …` (unbudgeted) → `rank.py --artifacts … --out submission.csv` (budgeted). Both documented + shipped in the README; `rank.py` is the single command Stage 3 reproduces.
- **`build_manifest.json` is the sole root** (see `contracts.md` C2) referencing every sub-manifest + `runtime_report.json` + `submission_sha256`.
- **Reproduction tiers:** ranking-step from shipped artifacts = bit-identical (canonical Docker) / semantic (else); full-from-raw (`precompute`→`rank`) = semantic-equivalent (embedding float nondeterminism documented; shipped artifacts are canonical).
- Runtime always attempts to produce the CSV; timing/memory ceilings are enforced in CI, not by self-aborting (a valid CSV <300 s still scores). A hard ~8 GB memory check stays.

## R7 — Build discipline (process)
- Push all heavy work into the unbudgeted offline phase; only per-candidate work is budget-constrained.
- **Profile after each subsystem** (ingest, representation, BM25, scoring, risk, selection, reasoning, full ranking step): measure wall-clock + peak RSS, fix the top bottleneck, record in `runtime_report.json` — don't defer performance to the end.
- Emit `runtime_report.json` every run; manifests + `runtime_report.json` = the Stage-3/Stage-5 evidence package.
