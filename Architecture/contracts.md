# contracts.md — Cross-goal interface contracts (authoritative)

> **Read every session.** This file owns the seams BETWEEN goals. Each `GoalN.md` owns a
> goal's internals; where a goal file and this file disagree on an interface, **this file
> wins**. Numbers/encodings still live in the goal files; this file pins the *contracts*
> that keep the 8 goals composable and aligned.

## C1 — Alignment & identity (the spine)
- **N = 100,000** candidates, no drops (malformed → quarantine, not delete — Goal 1 D5).
- **Canonical row order = source order** of `candidates.jsonl` (row *i* = *i*-th line). No sort pass. (Goal 1 D4)
- **`candidate_ids.npy`** (int32 of the 7-digit part) is the authoritative alignment key. **Every** artifact (embeddings ×2, `features.npy`/`.parquet`, risk flags, snippets, bm25 rows) is row-aligned to it by construction.
- `candidate_id` surface form matches `^CAND_[0-9]{7}$` (Goal 1 D5; validator-enforced).
- **Startup assertion (loud fail):** all artifacts have length N and `candidate_ids` line up before any matrix loads. (Goal 8 D5/D9)
- **`REF_DATE` = max(`last_active_date` in pool) + 1 day**, computed once offline, **frozen in manifests**, used pool-wide for all date math. Never `today()`/build-time. (Goal 3 D5) Sandbox uses the shipped frozen REF_DATE, never recomputes. (Goal 8 D8)

## C2 — Manifest & authenticity chain
- **`build_manifest.json` is the SOLE root** an organizer opens first; it references every sub-manifest + report by path and carries the hash chain + provenance (`candidates_sha256`, `goalN_manifest`, `runtime_report`, `submission_sha256`, `git_commit`, `lib_versions`, `model_revision`, `thread_settings`, timestamps). (Goal 8 D9)
- Each goal emits its own sub-manifest (sha256 of its artifacts + upstream hashes, versions, git commit, timestamp) — Goals 1–8 mirror the same discipline.
- **Artifact↔pool binding assertion** at startup: loaded artifacts' source-hash matches the candidates file (satisfies §10.3 "cover exactly the released pool").

## C3 — Embedding model (shared by JD and candidate sides)
- **`bge-small-en-v1.5`, 384-d, L2-normalized**, with a **query/passage prefix convention**. JD side uses the *query* prefix (Goal 2 D5); candidate side uses the *passage* prefix (Goal 3 D2). Pin model revision in manifests.

## C4 — Goal 2 → 3/4: the JD query artifact
- **`jd_query.json`** (canonical, human-readable): typed-criteria catalog (declarative *what*, no numeric weights), lexical concept groups + conservative synonyms, the ideal-candidate narrative text, anti-profile archetype narratives **+ ordered labels**, divergence-input declaration.
- **`jd_ideal.npy`** `[1×384]` f32 L2-norm. **`jd_antiprofile.npy`** `[A×384]` f32 L2-norm — **row order = the archetype label order declared in `jd_query.json`** (Goal 3 must consume vectors in that exact order).
- Typed-criterion record fields (Goal 2 D1) are the schema Goals 4/5/7 bind to: `id`, `criterion_type`, `polarity`, `importance`, `description` (= the `jd_label` Goal 7 renders), `evidence_sources[]` (must be real paths in `candidate_schema.json`), `corroboration{min_sources}`, `match_modes[]`, `target_kind`, optional `override_condition`, `provenance`.

## C5 — Goal 3 → 4/5/7: candidate representation
- **`embeddings_identity.npy`, `embeddings_evidence.npy`** `[N×384]` f32 L2-norm. (Goal 3 D2)
- **`features.npy`** `[N×F]` f32 (runtime scorer artifact) + **`features.parquet`** (named, debug) + **`feature_manifest.json`** (per-column: index, name, group, dtype, source paths, derivation+version, nullable+present-flag, signal role, corroboration concept+source, advisory scale; concept-registry & company_table refs; **anti-profile archetype label order = `jd_antiprofile.npy` row order**). (Goal 3 D1/D8)
- **NaN+`_present` contract (binding):** missing ⇒ value is NaN **and** companion `_present`==0; invariant `_present==0 ⇒ value MUST be NaN`; no sentinels, no imputation. **Consumers must mask via flags / nan-ops — never do arithmetic on raw NaN.** (Goal 3 D7)
- **`normalization_stats.json`** — per-feature full-pool stats {count, missing, min/max/mean/std, p1..p99}. **Goal 3 ships RAW features; Goal 4 owns all transforms** using these frozen stats. (Goal 3 D7 / Goal 4 D4)
- **`bm25_index/`** = CSR doc-term `.npz` + `idf.npy` + `doclen.npy` + `vocab.json`; corpus = narrative fields only, **skills[] array excluded** (diluted ~12k×/skill). (Goal 3 D3)
- **`evidence_snippets` sidecar** — criterion-linked narrative spans with **stable `evidence_id`s**, **row-addressable** for O(100) random access at runtime (runtime never reopens raw parquet). (Goal 3 G7/G8 addenda)
- **Corroboration sources** exposed as static {assessment, title, skillmeta}; Goal 4 adds {dense, lexical} at runtime and counts `min_sources`. **Goal 3 never computes corroboration counts.** (Goal 3 D6)

## C6 — The scoring hierarchy (no overlap)
- **Goal 4 = relevance** (soft). **Goal 5 = risk** (hard detection). **Goal 6 = decision** (gates + selection).
- `final_score = fit_score × engagement_modifier × risk_modifier`, computed for all; **selection restricted to the eligible subset**. (Goal 6 D-flow)
- **Soft risk shapes the score inside Goal 4 (D8, floor 0.5, monotone, no overrides); hard risk excludes inside Goal 6.** Different mechanisms, deliberately different magnitudes, no double-count. (Goal 4 D8 / Goal 5 intro)

## C7 — Goal 4 → 6/7: the fit bundle
Per shortlisted candidate: `fit_score`; component subscores; **per-criterion satisfaction degree + evidence trail** where each trail atom carries `{criterion_id, jd_label, class, importance, satisfaction_degree, fit_contribution (= normalized wᵢ·satᵢ), evidence:[{evidence_id, source_tier ∈ verified/demonstrated/declared, source_type, field_path, raw_value, surface_snippet}]}`; bounded **`engagement_modifier`** ∈ asymmetric [0.7, 1.1]; bounded **`risk_modifier`** ∈ [0.5, 1.0] (+ sub-components); semantic **`fit_tier` ∈ {Elite, Strong, Plausible, Filler}** (calibrated-space label, **not** a gating decision); **corroboration-breadth count**. (Goal 4 D1/D6/D6.5/D7/D8 + Goal 7 addendum)

## C8 — Goal 5 → 6: risk primitives
Stored per candidate (offline, aligned): `honeypot_flag`, `honeypot_reasons[]` (rule id + triggering values), `consulting_gate_flag`, `consulting_gate_suppressed`, `suppression_evidence[]`. Runtime-only diagnostic `stuffer_suspect_flag` rides in the Goal 4 bundle. **Goal 5 stores primitives only; it never composes the decision.** (Goal 5 D6)
- **Goal 6 derives (not persisted):** `hard_gate_flag = honeypot_flag OR (consulting_gate_flag AND NOT consulting_gate_suppressed)`. (Goal 5 D6 / Goal 6 D1)

## C9 — Goal 6 → output: the submission
- `submission.csv` columns **exactly** `candidate_id,rank,score,reasoning`; exactly 100 data rows; ranks 1–100 unique; ids unique & valid; **score non-increasing with rank**. (submission_spec §2–3; `validate_submission.py`)
- **Tie-break (validator ground truth):** on an exactly-equal printed `score`, ordering MUST be **`candidate_id` ascending** (`validate_submission.py` lines 136–144 — stricter than prose §3). Corroboration breadth influences order by being **folded into the printed-score micro-term**, not as a post-score reorder. **`fit_score` is never a tie-break.** (Goal 6 D2/D4)
- `selection_manifest.json`: counts, exclusions-by-reason, top-excluded, config hash, git, timestamp.

## C10 — Goal 7: reasoning grounding
- `reasoning` is a **pure projection** of the already-decided evidence contract — Goal 7 never reads raw candidate data and never selects evidence. (Goal 7 D1)
- **Citeable whitelist:** anchor literals; `skill_assessment_scores` (named skills allowed **only** here, Verified tier); career narrative snippets; `redrob_signals` values. **Raw `skills[]` tokens are NEVER citeable** (hallucination per Stage-4 rubric). Every cited token resolves to an `evidence_id`. (Goal 7 D4)
- Side artifacts: `reasoning_provenance.json` (clause→`evidence_id`), `reasoning_audit_report.json`.

## C11 — Builder-refactor for the two runtimes (Goal 8)
Per-candidate logic is factored as **pool-independent reusable functions** so the Sandbox Runtime can derive identical outputs for a ≤100 sample from frozen inputs:
- Goal 1: `parse_record(raw) → canonical struct` (separate from the streaming writer). Pool-level integrity stays in the offline driver only.
- Goal 3: `builder(candidate, frozen_inputs)` taking **REF_DATE, jd_ideal/jd_antiprofile, company_table, concept_registry, model** as frozen params (skips the pass-1 REF_DATE pre-scan; uses frozen REF_DATE).
- Goal 5: `risk_flags(candidate_features)` (every rule a per-candidate logical test).
- Pool-level stats (REF_DATE, normalization, tail-anchor calibration, concept freq, anti-profiles) are **shipped frozen and never recomputed from a sample** — Sandbox applying them *proves* they are pool-frozen. (Goal 8 D8)

## C12 — Consolidated carry-overs (gathered from the goal files)
- **G2→G3:** model locked (C3); BM25/dense text from narrative fields, not skills[]; features must supply evidence for every criterion's `evidence_sources`; `redrob_signals` feed atomic engagement criteria.
- **G3→G4:** raw features + frozen stats (G4 owns transforms); corroboration static sources {assessment,title,skillmeta} (G4 adds {dense,lexical}); two embeddings + identity_evidence_divergence + dual archetype cosines; G4 applied-ML components are inputs not a verdict; NaN+`_present` masking.
- **G4→G5/6/7:** G5 owns hard divergence/stuffer threshold + disqualifier overrides; G6 gets `fit×engagement×risk` + enforces gates + honeypot-constrained top-100; G7 consumes the per-criterion evidence trail; all audit-tuned numerics resolved in the Goal 4 D9 loop.
- **G5→G6/7:** G6 derives `hard_gate_flag` + owns eligibility/twin safeguard; G7 consumes suppression evidence + diagnostics for survivors.
- **G6→:** G7 reasoning consumed at assembly (validation #8/#11); G6 logic frozen across all 3 submissions (only Goal 4 weights vary {safe, tuned, final}).
- **G7→G3/4/6:** G4 emits `fit_contribution` + `fit_tier` + threads `evidence_id`/`source_tier`/`surface_snippet`/`jd_label`; G3 ships the `evidence_snippets` sidecar; G6 consumes `fit_tier` with `filler_flag ≡ fit_tier==Filler`.
