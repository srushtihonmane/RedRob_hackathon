# Build Handoff v2 — Review + Latest Looping Method + Single Prompt

*Supersedes `BUILD_HANDOFF.md`. The earlier file's "FATAL tie-break" issue is now **resolved in the plan itself** (Goal 6 D2/D4 + `Architecture/contracts.md` C9) and re-verified against `validate_submission.py`. This version also uses Anthropic's **newer** harness guidance (Mar 24, 2026), which postdates the article the v1 handoff cited.*

---

## 1. Critical review — only fatal / rule-level issues

I read all 8 goals in `brainstorm_handoff.md`, the `Architecture/` split (overview, contracts, runtime, artifacts, Goal1–8), `submission_spec.docx`, `validate_submission.py`, and `candidate_schema.json`.

**Verdict: no remaining fatal flaws and no hackathon-rule violations.** The plan is unusually complete and internally consistent, and every hard rule I could check is satisfied: exactly-100 rows, `score` non-increasing with rank, ranking step is CPU-only / no-network / ≤5 min / ≤16 GB / ≤5 GB disk, precompute-is-allowed (§10.3), honeypot ≤10%-in-top-100 (§7), ~80-honeypot framing, 3-submission cap, reasoning rubric, no special-casing of honeypots (general logic, not an ID blocklist).

### The one previously-"fatal" issue is already fixed — confirmed
- `validate_submission.py` lines 136–144 reject any equal-`score` pair not in **`candidate_id`-ascending** order. This is **stricter than prose spec §3** ("secondary signal *or* candidate_id ascending") and, as the Stage-1 auto-validator, is **ground truth**.
- The current plan already conforms: **Goal 6 D2** folds corroboration breadth into a sub-epsilon **printed-score micro-term** (`printed = final_score − rank·δ`), which makes printed scores strictly decreasing with rank → **no exact ties remain**, so the validator's tie-break branch never fires; **Goal 6 D4 (REVISED)** sets the only equal-score tie-break to `candidate_id` ascending; **contracts.md C9** pins this as authoritative. I traced the math: for any δ>0, `printed` is strictly decreasing whenever `final_score` is non-increasing, so the validator's `s1 < s2` and `s1 == s2 ∧ c1 > c2` checks both pass. ✅ Keep it; don't reintroduce a breadth-first post-score reorder.

### Two operational must-ships (not design flaws, but they fail you if dropped)
1. **§10.5 sandbox / demo link.** "Submissions without a working sandbox link are **flagged at Stage 1**." Goal 8 D8 leaves hosting "not locked." The accepted fallback is a `docker run` recipe in the README whose image **builds and runs unmodified** — make this a tracked deliverable, not an afterthought.
2. **Submission packaging.** The CSV filename must be your **registered participant ID** (`<participant_id>.csv`), and the repo root needs `submission_metadata.yaml` (from the template). `rank.py` can emit `submission.csv`; just rename at upload.

### Minor defensibility notes (optional)
- Vendor/cache the `bge-small-en-v1.5` weights with the precompute code so the offline step is reproducible without network too (Stage-3/5 defense).
- Keep the Stage-5 framing that H1/H2/H3 are **general logical-consistency checks**, with corroboration-first scoring as the *primary* honeypot defense — that's what makes "we don't special-case honeypots" true.

Net: build it. The architecture is sound and rule-compliant as written.

---

## 2. The looping method (Anthropic's latest, June 2026)

There are now **two** relevant Anthropic engineering posts, and the newer one matters:

- **"Effective harnesses for long-running agents"** (Nov 26, 2025) — the **two-agent** harness with a public, runnable quickstart. An **initializer agent** sets up the environment once; a **coding agent** makes incremental progress every session. Core artifacts: a JSON **`feature_list.json`** (every feature an end-to-end test, all `passes:false`), an **`init.sh`** smoke test, a **`claude-progress.txt`** log, and **git** commits. Rules that make it work: one feature at a time, **self-verify end-to-end with real tests**, flip `passes:true` only after it truly passes, and **never edit or delete a test to make it pass**.
- **"Harness design for long-running application development"** (Mar 24, 2026) — the **newer** evolution. A **GAN-inspired three-agent** architecture (**planner → generator → evaluator**). Two durable lessons:
  1. **Separate generation from evaluation.** Agents "reliably skew positive when grading their own work," so a *separate, skeptical evaluator* beats self-evaluation — even on verifiable tasks. Tuning a standalone skeptic is far more tractable than making a builder critical of itself.
  2. **Strip non-load-bearing scaffolding.** "Every component encodes an assumption about what the model can't do on its own"; on stronger models they dropped context-resets and even the sprint construct, running one continuous session with auto-compaction. The guiding rule (from *Building Effective Agents*): *use the simplest thing that works; add complexity only where the task exceeds what the model does reliably solo.*

### Is the latest method suitable here? Partly — take the valuable half.
The Mar-2026 harness was built for **subjective, hard-to-verify** web-app quality, which is why it needs an LLM evaluator clicking through a live UI with Playwright. **Your task is the opposite: it's richly machine-verifiable.** You already ship the exact "end-to-end tests" the harness verifies against:
- `validate_submission.py` (the official auto-validator),
- the trap/adversarial unit suite (stuffer ranked down, Tier-5 up, each honeypot archetype excluded — Goal 4 D9 / Goal 5 D6),
- invariant asserts (honeypot rules, alignment, NaN/`_present` contract),
- runtime budget checks (wall-clock + peak RSS, CPU-only, no-network),
- the two-command reproduction.

So the right design is: **use the Nov-2025 two-agent loop as the spine** (it's public, single-prompt-friendly, and maps perfectly because your validators *are* the tests), and **borrow exactly one upgrade from Mar-2026: keep evaluation separate from generation.** Concretely:
- **Objective gates** → the machine validators are the external skeptic. The builder may *run* them but the build's `passes` flips only on their real output, and the builder may **never** weaken them.
- **Subjective gates** (the Goal 4 D9 top-k human audit and the Stage-4 reasoning quality) → run a **separate "skeptical reviewer"** pass (a subagent, or a fresh session with an adversarial prompt) that tries to *falsify* "this top-100 is good" and "this reasoning is grounded," and feeds failures back as new `passes:false` features. Don't let the builder grade its own ranking taste.
- You do **not** need the planner agent (your `Architecture/` spec already is the plan — the initializer just decomposes it into `feature_list.json`) and you do **not** need Playwright (no UI). That's the "strip non-load-bearing scaffolding" lesson applied.

### How to actually run it
- **Simplest (recommended):** one continuous **Claude Code** session pointed at this folder; paste the single prompt in §3. On current Opus models, auto-compaction + the file-based handoff (progress + `feature_list.json` + git) is enough — no manual context-reset orchestration. When context gets large, `/clear` and paste the same prompt; the agent re-orients from the files.
- **Most autonomous (multi-session):** drop the same prompt into the quickstart loop at `github.com/anthropics/claude-quickstarts/tree/main/autonomous-coding`; its script re-invokes the coding agent until every feature passes. Reference loop runner: `github.com/anthropics/cwc-long-running-agents`.
- **Separated evaluator:** either spawn it as a subagent from inside the loop (Task tool / Claude Code subagent), or run a dedicated "review" session between build sessions. Keep its prompt skeptical and its findings written to a file the builder reads.

---

## 3. The single prompt (paste into Claude Code)

> Give Claude Code access to this folder, then paste everything in the block below. Run PHASE 0 once; it then loops PHASE 1 until the stop condition.

```
You are building, validating, and iterating an end-to-end candidate-ranking system
AUTONOMOUSLY until it is competition-ready. Work feature-by-feature, self-verify with
real tests, and do not stop until every feature passes and a valid submission.csv is
reproducibly produced within budget.

SPEC & GROUND TRUTH (the plan lives in the Architecture/ folder):
- Architecture/overview.md   <- shared context (task, JD, traps, constraints, decided
                                architecture, status). Read first.
- Architecture/contracts.md  <- AUTHORITATIVE for all INTER-goal interfaces (alignment,
                                manifests, schemas, scoring boundary, tie-break). Read
                                EVERY session. If a GoalN.md and contracts.md disagree on
                                an interface, contracts.md wins.
- Architecture/runtime.md    <- runtime envelope, Runtime ML Policy, determinism,
                                two-command repro. Read EVERY session.
- Architecture/artifacts.md  <- inventory of every artifact (producer, format, consumers).
- Architecture/GoalN.md      <- the detailed design for the goal you implement this
                                session (read only the relevant one).
- submission_spec.docx, job_description.docx, candidate_schema.json
- validate_submission.py     <- the official auto-validator. GROUND TRUTH; it OVERRIDES any
                                prose in the spec or the plan wherever they disagree.
- Data: candidates.jsonl (~487MB, 100k rows), sample_candidates.json,
        sample_submission.csv, candidate_schema.json, submission_metadata_template.yaml
(brainstorm_handoff.md is the original monolith, retained only as an archive — build from
Architecture/.)

NON-NEGOTIABLE INVARIANTS (already settled in contracts.md — enforce, don't re-litigate):
1. TIE-BREAK: on EQUAL printed `score`, ordering MUST be `candidate_id` ASCENDING ONLY,
   because validate_submission.py (lines 136-144) rejects anything else. Corroboration
   breadth is folded into the printed score as a deterministic sub-epsilon micro-term
   (printed = final_score - rank*delta) so breadth still influences order and printed
   scores are effectively unique; NEVER do a post-score reorder. validate_submission.py
   MUST pass on the final CSV.
2. RUNTIME ML POLICY (behavior, not a package blocklist): the ranking step is a FROZEN
   SCORER, not an ML pipeline. It must NOT instantiate, load, download, or execute any
   neural model or inference engine (transformers, sentence-transformers, Torch, ONNX,
   GGUF, etc.) to compute candidate-dependent outputs at runtime. ALL neural computation
   (embeddings, reranking, calibration fitting, any inference) happens OFFLINE in
   precompute. At runtime: load precomputed artifacts + deterministic
   retrieval/filter/score/rank/reason/CSV only.
3. DETERMINISM: PYTHONHASHSEED=0, stable hashing (zlib.crc32/hashlib, never salted hash()),
   single-thread BLAS in the repro image, no clocks/locale/RNG. Pin every dependency version.
4. TWO-COMMAND REPRO: precompute.py (unbudgeted) -> rank.py --candidates candidates.jsonl
   --out submission.csv (budgeted: CPU-only, no-network, <=5 min, <=16 GB, <=5 GB disk).

FAITHFULNESS vs SIMPLIFICATION: treat the plan's ARCHITECTURE, DATA CONTRACTS, and
LOAD-BEARING decisions as binding (the offline<->runtime boundary, corroboration-first
scoring, the trap/risk handling, two-command reproduction, determinism) — Stages 4-5 judge
architectural faithfulness. You MAY simplify or collapse INTERNAL implementation details
(a single D sub-decision is not automatically an immutable test) when ALL of: correctness
is preserved, runtime/memory is unchanged or better, and the architecture stays faithful.
Log every such deviation in claude-progress.txt with a one-line rationale. Don't gold-plate;
don't encode every D sub-point as a hard test.

HARNESS — do PHASE 0 once, then loop PHASE 1.

PHASE 0 (INITIALIZER — run only if feature_list.json does not yet exist):
1. Read the spec docs above. Decompose Goals 1-8 into discrete, END-TO-END-TESTABLE
   features in feature_list.json, at the level of contracts and observable behavior (NOT
   every internal D sub-decision). Each feature: {id, goal, category
   (functional|artifact|constraint|trap|repro|packaging), description, steps[],
   passes:false}. Include explicit features for every hard constraint, trap, and
   deliverable, e.g.:
   - ranking step runs CPU-only, no-network, <=5 min wall-clock, <=16 GB RAM, <=5 GB disk
   - precompute.py -> rank.py --candidates candidates.jsonl --out submission.csv is a
     working two-command reproduction
   - validate_submission.py passes on the produced CSV (score non-increasing, exactly 100
     rows, unique ranks/ids, equal-score => candidate_id ascending)
   - all rule-detected honeypots (H1/H2/H3, OR-fired) excluded from the top-100, AND
     top-100 honeypot exposure kept comfortably below the 10% DQ threshold WITH MARGIN
     (the company-age honeypots are unobservable; build for margin, not a guaranteed zero)
   - canonical keyword-stuffer ranked DOWN; plain-language Tier-5 ranked UP; each honeypot
     archetype excluded (the trap/adversarial unit suite — Goal 4 D9 / Goal 5 D6)
   - alignment invariants: 100k rows; candidate_ids match across ALL artifacts; NaN/_present
     contract holds
   - build_manifest.json hash chain + per-goal sub-manifests + runtime_report.json
   - reasoning column: non-empty, unique, NO skills[]-array hallucination, rank-tone
     consistent, <=~240 chars, QUOTE_ALL CSV
   - PACKAGING: a working §10.5 sandbox/demo (HuggingFace Space / Streamlit / Colab / a
     `docker pull`+`docker run` recipe) that ranks a <=100-candidate sample end-to-end in
     budget, AND submission_metadata.yaml at repo root, AND README with the single repro
     command. (Missing sandbox = Stage-1 flag.)
2. Write init.sh: create TWO pinned dependency sets — Competition Runtime = lean,
   CPU-only/offline (e.g. numpy/scipy/pyarrow/pandas/joblib as needed; no GPU/network deps,
   no runtime model inference); precompute+sandbox = + torch/transformers/sentence-
   transformers + any reranker. Then run a fast smoke test (import ranking deps, run the
   trap unit tests, run validate_submission.py on any existing submission.csv).
3. Write claude-progress.txt (done / next / key decisions / risks).
4. git init && initial commit of scaffolding + feature_list.json.
5. Build the VERIFICATION HARNESS EARLY: tests/ with the trap/adversarial unit tests,
   invariant asserts, and a runtime-budget check that measures wall-clock + peak RSS and
   writes runtime_report.json. These tests are the source of truth for "passes" — the
   external skeptic the build is graded against.

PHASE 1 (CODING LOOP — repeat until the STOP CONDITION):
1. Run `pwd`; read claude-progress.txt, `git log --oneline -20`, and feature_list.json.
2. Run init.sh smoke test. If anything is broken, FIX IT before any new work.
3. Pick the SINGLE highest-priority feature with passes:false. Implement only that.
   Build order: Goal 1 ingest -> Goal 2 JD artifact -> Goal 3 representation -> Goal 4
   scoring -> Goal 5 risk -> Goal 6 selection -> Goal 7 reasoning -> Goal 8 runtime/repro,
   then the Goal 4 D9 audit/tuning loop, then packaging.
4. SELF-VERIFY end-to-end against REAL data and the REAL validators — never mark a feature
   passing on unit tests alone. For the full pipeline, actually run precompute -> rank on
   candidates.jsonl, time it, and run validate_submission.py on the real output. Flip
   passes:true ONLY after it genuinely passes. NEVER edit, weaken, or delete a test to make
   it pass (that could hide missing/broken functionality).
5. SEPARATE EVALUATION FROM GENERATION for the SUBJECTIVE gates (you skew positive grading
   your own ranking taste). For the Goal 4 D9 top-k audit and the Stage-4 reasoning quality,
   run a SEPARATE skeptical reviewer (spawn a subagent, or do a dedicated review pass with
   an adversarial prompt) that tries to FALSIFY "this top-100 is good" and "this reasoning is
   specific, honest, and hallucination-free." Write its findings to a file and turn each real
   problem into a new passes:false feature. Objective features stay gated by the machine
   validators above.
6. PROFILE after each major subsystem (ingest, embeddings/representation, BM25, scoring,
   risk, selection, reasoning, full ranking step): measure wall-clock + peak RSS on
   realistic data shapes, find the top bottleneck, optimize it before moving on. Record
   numbers in runtime_report.json and claude-progress.txt. Do NOT defer performance work to
   the end — catch a slow ranking step early, not at minute 12.
7. Commit with a descriptive message; update claude-progress.txt. Leave the repo in a clean,
   mergeable state (no half-built features, no broken imports).

STOP CONDITION: every feature in feature_list.json has passes:true AND
`python rank.py --candidates candidates.jsonl --out submission.csv` produces a CSV that
passes validate_submission.py within <=5 min / <=16 GB / CPU-only / no-network, with the
trap suite green (all rule-detected honeypots excluded; top-100 honeypot exposure
comfortably below 10%; stuffer ranked down; Tier-5 ranked up), the §10.5 sandbox/docker-run
recipe working on a <=100 sample, submission_metadata.yaml present, and the precompute->rank
two-command reproduction documented in README.md. Then write a final summary to
claude-progress.txt and stop.

DECISION RULES (resolve anything the prompt doesn't spell out):
- When in doubt, push computation into PRECOMPUTE, not runtime. Runtime is a lightweight
  artifact loader + scorer.
- If two implementations both satisfy the official competition rules, prefer the SIMPLER
  one — avoid runtime complexity or uncertainty (e.g. never load a transformer at runtime
  to compute embeddings that are already available offline).
- The ranking step's ONLY hard runtime limits are CPU-only / no-network / <=5 min / <=16 GB
  / <=5 GB disk — ANY dependency that respects them is allowed.
```

---

## 4. Quick how-to recap
1. Open this folder in **Claude Code**.
2. Paste the §3 block. It runs PHASE 0 once (scaffold + `feature_list.json` + tests), then loops PHASE 1.
3. When context fills, `/clear` and paste the same block — it re-orients from `claude-progress.txt` + `feature_list.json` + git. Or wire it into the `autonomous-coding` quickstart for hands-off multi-session runs.
4. Periodically let the **separate skeptical-reviewer** pass run over the top-k audit and reasoning before you spend a submission.
5. Stop when every feature is green and `rank.py` yields a validator-passing CSV in budget.

*Sources:*
- [Harness design for long-running application development — Anthropic (Mar 24, 2026)](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- [Effective harnesses for long-running agents — Anthropic (Nov 26, 2025)](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [autonomous-coding quickstart](https://github.com/anthropics/claude-quickstarts/tree/main/autonomous-coding) · [cwc-long-running-agents](https://github.com/anthropics/cwc-long-running-agents)
- Local: `validate_submission.py`, `submission_spec.docx`, `Architecture/contracts.md`, `brainstorm_handoff.md`
