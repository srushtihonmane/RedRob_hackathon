# Architecture — Index (canonical entry point)

> This folder is the canonical, split version of the original `brainstorm_handoff.md`
> (kept as an archive). Read order for any build session:
> 1. `overview.md` (this is PART A — shared context, traps, constraints, decided architecture)
> 2. `contracts.md` (authoritative for all INTER-goal interfaces — read every session)
> 3. `runtime.md` (the runtime envelope + ML policy + reproduction — read every session)
> 4. `artifacts.md` (inventory of every artifact: producer, format, consumers)
> 5. `GoalN.md` for the goal you are implementing this session
>
> Ownership rule: `contracts.md` owns the seams BETWEEN goals; each `GoalN.md` owns the
> internal design of that goal; `runtime.md` owns cross-cutting runtime/determinism;
> `artifacts.md` is the derived index. If a goal file and contracts.md ever disagree on
> an interface, contracts.md wins.

## Files
- [overview.md](overview.md) — shared context + status summary + build order
- [contracts.md](contracts.md) — cross-goal interface contracts (alignment, manifests, schemas, boundaries)
- [runtime.md](runtime.md) — runtime envelope, Runtime ML Policy, determinism, two-command repro
- [artifacts.md](artifacts.md) — full artifact inventory
- [Goal1.md](Goal1.md) — Data ingestion
- [Goal2.md](Goal2.md) — JD understanding
- [Goal3.md](Goal3.md) — Candidate representation
- [Goal4.md](Goal4.md) — Candidate↔JD scoring (with internal module decomposition)
- [Goal5.md](Goal5.md) — Trap handling
- [Goal6.md](Goal6.md) — Ranking & selection
- [Goal7.md](Goal7.md) — Reasoning generation
- [Goal8.md](Goal8.md) — Compute optimization

---

# Redrob Ranking Challenge — Brainstorm Handoff

> **How to use this file.** We're brainstorming each goal in its own chat. In any goal chat, paste **PART A (Shared Context)** first, then paste the **one PART B goal section** you want to work on. PART A gives a cold chat everything it needs; PART B says where that goal stands and what's still open.

---

# PART A — Shared Context (paste in every goal chat)

## The task
Redrob "Intelligent Candidate Discovery & Ranking Challenge." Given a **single fixed job description** and a pool of **100,000 candidates** (`candidates.jsonl`, ~487 MB), produce a CSV ranking the **top 100 best-fit candidates**, with `candidate_id, rank, score, reasoning`. Scored once against a hidden ground truth after submissions close.

## Why this JD is special
The JD is the *winners' actual job*. Its "What you'd actually be doing" section tells you the company's real stack and objective, and your ranker should mirror it:
- Current system = **BM25 + rule-based scoring** ("working but not great") → your baseline.
- v2 they want = **embeddings + hybrid retrieval + (LLM) re-ranking** → your architecture.
- Eval = **offline benchmarks + A/B + recruiter-feedback loops**.
- The optimized metric = **recruiter-engagement / hireability**, NOT raw skill-match. So the behavioral signals are part of *relevance*, not just a fraud filter.

## What the JD actually wants (the typed criteria model)
- **Must-haves (hard, high weight):** production embeddings-retrieval (handled drift/index-refresh/regression); vector-DB or hybrid-search ops (Pinecone/FAISS/OpenSearch/etc.); strong Python; ranking eval frameworks (NDCG/MRR/MAP, offline↔online, A/B); **shipped an end-to-end ranking/search/rec system at a *product* company, at scale** (the spine).
- **Nice-to-haves (low weight):** LLM fine-tuning (LoRA/QLoRA/PEFT); learning-to-rank (XGBoost/neural); HR-tech; distributed systems; OSS.
- **Disqualifiers / strong negatives:** pure research, no production; "AI" = <12 mo LangChain+OpenAI with no pre-LLM ML; senior who hasn't coded in 18 mo ("moved to architecture"); title-chaser (~1.5-yr hops for title); career *entirely* at consulting firms (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini); CV/speech/robotics primary with no NLP/IR; 5+ yrs closed-source only with no external validation.
- **Context modifiers:** experience 5–9 yrs is a **soft band** (ideal 6–8), not a gate; location Noida/Pune or willing to relocate from a Tier-1 Indian city; notice period (≤30 ideal); **availability** (active, responsive, open_to_work).
- **Ideal profile:** 6–8 yrs total, 4–5 in applied ML at product (not services) companies; shipped a ranking/search/rec system at scale; has defensible opinions on retrieval/eval/LLM; in or willing to relocate to Noida/Pune; active on the platform.

## Candidate data shape (`candidate_schema.json`)
- `profile`: headline, summary, location, country, years_of_experience, current_title, current_company, current_company_size, current_industry.
- `career_history[]`: company, title, start_date, end_date, duration_months, is_current, industry, company_size, description.
- `education[]`: institution, degree, field_of_study, start_year, end_year, grade, **tier** (tier_1..tier_4/unknown).
- `skills[]`: name, **proficiency** (beginner/intermediate/advanced/expert), **endorsements**, **duration_months**.
- `certifications[]`, `languages[]`.
- **`redrob_signals` (23 behavioral signals):** profile_completeness_score, signup_date, last_active_date, open_to_work_flag, profile_views_received_30d, applications_submitted_30d, recruiter_response_rate, avg_response_time_hours, **skill_assessment_scores** (dict skill→0-100), connection_count, endorsements_received, notice_period_days, expected_salary_range_inr_lpa{min,max}, preferred_work_mode, willing_to_relocate, github_activity_score (-1 if none), search_appearance_30d, saved_by_recruiters_30d, interview_completion_rate, offer_acceptance_rate (-1 if none), verified_email, verified_phone, linkedin_connected.

## The traps (the contest is decided here)
1. **Keyword stuffers** — list every AI buzzword; wrong title/no corroboration. (The provided `sample_submission.csv` falls for this: it ranks an HR Manager #1 on "AI skills count.")
2. **Plain-language Tier-5s** — never write "RAG"/"Pinecone" but career history shows they built the system. Must **not** be penalized for missing buzzwords.
3. **Behavioral twins** — near-duplicate profiles/signal envelopes.
4. **Honeypots (~80, "subtly impossible")** — e.g. 8 yrs at a company founded 3 yrs ago; "expert" in 10 skills with 0 months used. Forced to relevance tier 0 in ground truth. **Honeypot rate >10% in your top 100 = disqualification (Stage 3).**

## Hard constraints & facts
- **Scoring:** `Final = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`. P@10 "relevant" = tier 3+. **Half the score is the top 10.**
- **Compute (ranking step only):** ≤5 min wall-clock, ≤16 GB RAM, **CPU only**, **no network**, ≤5 GB disk.
- **Precompute is ALLOWED (resolved from spec §10.3):** embeddings/indexes/weights may be built offline (may exceed 5 min) and shipped as artifacts; the *ranking step* must load—not recompute—them and finish in budget. Ship the precompute script too (Stage 4 checks git authenticity). Artifact must cover exactly the released pool.
- **No labels, no live leaderboard, 3 submissions max** (last valid counts). You must be your own offline benchmark.
- **Pipeline stages:** 1 format-validation → 2 scoring → 3 code reproduction + honeypot check → 4 manual review (reasoning quality, git authenticity, code quality) → 5 defend-your-work interview. AI tools allowed but AI-only fails at 3–5.

## Cross-cutting principles
1. **JD is singular & offline** → understanding it can be arbitrarily heavy, done once.
2. **Only per-candidate work is budget-constrained** → push everything possible into the offline phase.
3. **Pure similarity loses; pure rules miss plain-language fits** → the design rewards **learned matching + explicit logic/consistency checks layered together**. That layering is the "real engineering" being scored at Stages 3–5.
4. **Never trust one field** → require ≥2 corroborating sources; catch honeypots by *contradiction*.
5. **Defensibility counts** (Stage 5) → prefer transparent, explainable components where accuracy is comparable.

## The decided overall architecture
**Offline phase (unbudgeted):** stream `candidates.jsonl` once → build structured features + the 23 signals + sparse lexical doc + **dense embeddings** → persist as compact Arrow/Parquet + `.npy`/FAISS index. Encode the JD once into a 3-channel query (lexical terms / dense "ideal-candidate" embedding / structured gates+weights).
**In-budget ranking step:** load artifacts → **hybrid retrieve** (BM25 ∪ dense kNN) → **gate** (honeypot/consistency + hard disqualifiers floor candidates) → **re-rank** (fused, calibrated score with engagement multiplier) → **constrained top-100** (enforce honeypot rate ≪10%) → **generate reasoning** from the per-criterion evidence trail.

---

---

## Status summary
**Fully detailed & locked:** **Goal 1** (data ingestion, D1–D5), **Goal 2** (JD understanding, D1–D8), **Goal 3** (candidate representation, D1–D9), **Goal 4** (candidate↔JD scoring, D1–D9 incl. D6.5), **Goal 5** (trap handling, D1–D6), **Goal 6** (ranking & selection, D1–D7), **Goal 7** (reasoning generation, D1–D8), and **Goal 8** (compute optimization, D1–D9). **All 8 goals now fully detailed & locked.**
**Goal 8 → upstream carry-backs (RESOLVED, verified deliverable, folded into Goals 1/3/5):** two named runtimes — **Competition Runtime** (artifact-only, numpy/scipy, the 5-min reproduction contract) and **Sandbox Runtime** (end-to-end on a ≤100 sample, may carry models) — share ranking logic (Goals 4→7) but not deps. This requires Goals 1/3/5 builders to be **reusable `builder(candidate, frozen_inputs)` functions** and the Goal 3 `evidence_snippets` sidecar to be **row-addressable** (O(100) access). Verified against each goal's locks: **no conflicts** — Goal 1 splits parser from writer; Goal 3 is already raw-feature + REF_DATE-frozen; Goal 5 rules are per-candidate logical tests. Single root **`build_manifest.json`** is the sole top-level entry point referencing all sub-manifests + `runtime_report.json` + `submission_sha256`.
**Goal 7 → upstream carry-backs (RESOLVED, folded into Goals 3/4/6):** Goal 4 emits per-criterion `fit_contribution` (normalized `wᵢ·satᵢ`) + semantic `fit_tier ∈ {Elite,Strong,Plausible,Filler}` (calibrated-space thresholds; relevance label, not gating) + threads `evidence_id`/`source_tier`/`surface_snippet`/`jd_label` on the evidence trail; **Goal 3** adds an `evidence_snippets` sidecar (criterion-linked spans + stable `evidence_id`s, overriding D1's read-on-demand for citations); **Goal 6** consumes `fit_tier`, with `filler_flag ≡ fit_tier==Filler`. No conflicts; tier↔rank divergence bounded and on the Stage-4 watch-list.
**Still open:** numeric tuning executed alongside the build (Goal 4 D9): the importance ladder, θ_assess, dense-hit cutoff, soft-AND ε, attenuation floor, risk weights/floor, engagement slope, and the diagnostic stuffer threshold. The offline eval/self-labeling harness is **decided** as Goal 4 D9 (audit-driven optimization + a ~250–400 hand-labeled tripwire used as a regression alarm only). (Goal 6's `hard_gate_flag` derivation and the twin-safeguard question are now resolved — see Goal 6 D1/D5.)
**Recommended order:** build (Goals 1→3 offline artifacts → Goal 4/5 scoring+risk → Goal 6 selection → Goal 7 reasoning) + the D9 audit/tuning loop.

> **Goal 2 → Goal 3 carry-overs (read before Goal 3 chat):** (1) Embedding model is **locked**: `bge-small-en-v1.5`, 384-d, L2-normalized, with a query/passage-prefix convention — candidate side must use the *passage* prefix to match the JD's *query* prefix. (2) Candidate BM25 doc and dense text must come from **narrative fields (summary, headline, career titles+descriptions), not the `skills[]` array** (diluted ~12k×/skill). (3) Goal 3 features must supply evidence for the Goal 2 criteria's `evidence_sources` (incl. `skill_assessment_scores` for corroboration, derived `applied_ml_years_at_product`, tenure/title-chaser stats, summed-tenure-vs-YOE for honeypot input). (4) `redrob_signals` feed the atomic engagement criteria. (5) Profiling script lives in session scratch — re-runnable; key distributions captured in the Goal 2 section above.

> **Note on Goal 1 ↔ Goal 3 boundary:** Goal 1 is now scoped to **pure ingestion only** (faithful nested Parquet of raw data). All derivation/representation — embeddings, numeric feature matrix, lexical doc — lives in **Goal 3** and is built by iterating Goal 1's canonical table in source row order.
