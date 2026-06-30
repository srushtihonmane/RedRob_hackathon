# Claude Code — single autonomous build prompt

**How to use:** open Claude Code in `D:\Hackathon_2july` (so it can see `Architecture/`,
the data, and `validate_submission.py`), then paste the whole block below as one message.
It runs Anthropic's long-running-agent harness: an initializer pass (feature list +
`init.sh` + progress file + git) followed by a self-verifying coding loop that works one
feature at a time until a valid `submission.csv` is reproducibly produced within budget.

For guaranteed multi-context-window autonomy, drop this same block into the
[`autonomous-coding` quickstart](https://github.com/anthropics/claude-quickstarts/tree/main/autonomous-coding)
loop (it re-invokes the coding agent until every feature passes). In a single long session
it also works with auto-compaction.

---

```
You are building, validating, and iterating an end-to-end candidate-ranking system
AUTONOMOUSLY until it is competition-ready. Work feature-by-feature, self-verify with
real tests, and do not stop until every feature passes and a valid submission.csv is
reproducibly produced within budget.

SPEC & GROUND TRUTH (the plan lives in the Architecture/ folder):
- Architecture/overview.md   <- shared context (task, JD, traps, constraints, decided architecture) + status. Read first.
- Architecture/contracts.md  <- AUTHORITATIVE for all INTER-goal interfaces (alignment, manifests,
                                schemas, scoring boundary, tie-break). Read EVERY session. If a GoalN.md
                                and contracts.md disagree on an interface, contracts.md wins.
- Architecture/runtime.md    <- runtime envelope, Runtime ML Policy, determinism, two-command repro. Read EVERY session.
- Architecture/artifacts.md  <- inventory of every artifact (producer, format, consumers).
- Architecture/GoalN.md      <- the detailed design for the goal you implement this session (read only the
                                relevant one; Goal4.md carries an internal 4A-4E module decomposition).
- submission_spec.docx, job_description.docx, candidate_schema.json
- validate_submission.py     <- the official auto-validator. GROUND TRUTH; overrides any prose in the
                                spec or the plan where they disagree.
- Data: candidates.jsonl (~487MB, 100k rows), sample_candidates.json, sample_submission.csv,
        candidate_schema.json, submission_metadata_template.yaml
(The original monolithic brainstorm_handoff.md is retained only as an archive — build from Architecture/.)

NON-NEGOTIABLE INVARIANT (already fixed in Goal6.md/contracts.md C9 — enforce it):
the printed-CSV tie-break for EQUAL `score` values MUST be `candidate_id` ASCENDING ONLY,
because validate_submission.py rejects anything else. Corroboration breadth is folded into
the printed score as a deterministic sub-epsilon micro-term (so breadth still influences
order) and is NEVER a post-score reorder. Output of validate_submission.py MUST pass.

HARNESS — do PHASE 0 once, then loop PHASE 1 every session:

FAITHFULNESS vs SIMPLIFICATION: treat the plan's ARCHITECTURE, DATA CONTRACTS, and
LOAD-BEARING decisions as binding — the offline<->runtime boundary, corroboration-first
scoring, the trap/risk handling, the two-command reproduction, and determinism. These
must stay faithful (Stages 4-5 judge architectural faithfulness). You MAY simplify or
collapse INTERNAL implementation details (a given D sub-decision is not automatically an
immutable test) when ALL of: correctness is preserved, runtime/memory improves or is
unchanged, and the architecture stays faithful. Log any such deviation in
claude-progress.txt with a one-line rationale. Don't gold-plate; don't encode every D
sub-point as a hard test.

PHASE 0 (INITIALIZER, run only if feature_list.json does not yet exist):
1. Read the spec docs above. Decompose Goals 1-8 into discrete, end-to-end-testable
   features in feature_list.json, at the level of contracts and observable behavior (not
   every internal D sub-decision). Each feature: {id, goal, category
   (functional|artifact|constraint|trap|repro), description, steps[], passes:false}.
   Include explicit features for every hard constraint and trap, e.g.:
   - ranking step runs CPU-only, no network, <=5 min wall-clock, <=16 GB RAM, <=5 GB disk
     (these are the ONLY hard runtime limits; any compliant dependency is allowed)
   - precompute.py (unbudgeted) -> rank.py --candidates candidates.jsonl --out submission.csv
     (budgeted) is a working two-command reproduction
   - validate_submission.py passes on the produced CSV (incl. score non-increasing,
     exactly 100 rows, unique ranks/ids, equal-score => candidate_id ascending)
   - all rule-detected honeypots (H1/H2/H3, OR-fired) excluded from the top-100, AND
     top-100 honeypot exposure kept comfortably below the 10% DQ threshold with margin
     (the hidden honeypots are unobservable, so build for margin, not a guaranteed zero)
   - canonical stuffer ranked DOWN; plain-language Tier-5 ranked UP; each honeypot
     archetype excluded (the trap/adversarial unit-test suite from Goal 4 D9 / Goal 5 D6)
   - alignment invariants: 100k rows, candidate_ids match across all artifacts
   - build_manifest.json hash chain + per-goal sub-manifests + runtime_report.json
   - reasoning column: non-empty, unique, no skills[]-array hallucination, rank-tone
     consistent, <=~240 chars, QUOTE_ALL CSV
2. Write init.sh: create the two pinned venvs/requirements (Competition Runtime = lean
   CPU-only/offline deps, e.g. numpy/scipy/pyarrow/pandas/joblib as needed — no
   GPU/network deps, no runtime model inference; precompute/sandbox = +
   torch/transformers/sentence-transformers), then run a fast smoke test (import ranking
   deps, run the trap unit tests, run validate_submission.py on any existing submission.csv).
3. Write claude-progress.txt (what's done, what's next, key decisions/risks).
4. git init && initial commit of scaffolding + feature_list.json.
5. Build the verification harness EARLY: tests/ with the trap/adversarial unit tests,
   invariant asserts, and a runtime budget check that measures wall-clock + peak RSS and
   writes runtime_report.json. These tests are the source of truth for "passes".

PHASE 1 (CODING LOOP, repeat until done):
1. Run `pwd`; read claude-progress.txt, `git log --oneline -20`, and feature_list.json.
2. Run init.sh smoke test. If anything is broken, FIX IT before new work.
3. Pick the single highest-priority feature with passes:false. Implement only that.
   Build order: Goal 1 ingest -> Goal 2 JD artifact -> Goal 3 representation -> Goal 4
   scoring (modules 4A retrieval -> 4B criterion eval -> 4C fusion -> 4D calibration ->
   4E modifiers) -> Goal 5 risk -> Goal 6 selection -> Goal 7 reasoning -> Goal 8
   runtime/repro, then the Goal 4 D9 audit/tuning loop.
4. SELF-VERIFY end-to-end against real data and the real validators — never mark a
   feature passing on unit tests alone. For the full pipeline, actually run
   precompute -> rank on candidates.jsonl, time it, and run validate_submission.py on the
   real output. Only flip passes:true after it genuinely passes. NEVER edit or delete a
   test to make it pass.
5. PROFILE after each major subsystem (ingest, embeddings/representation, BM25, scoring,
   risk, selection, reasoning, full ranking step): measure wall-clock + peak RSS on
   realistic data shapes, identify the top bottleneck, and optimize it before moving on.
   Record the numbers in runtime_report.json and claude-progress.txt. Do NOT defer
   performance work to the end — catch a slow ranking step early, not at minute 12.
6. Commit with a descriptive message; update claude-progress.txt. Leave the repo in a
   clean, mergeable state (no half-built features, no broken imports).

STOP CONDITION: every feature in feature_list.json has passes:true AND
`python rank.py --candidates candidates.jsonl --out submission.csv` produces a CSV that
passes validate_submission.py within <=5 min / <=16 GB / CPU-only / no-network, with the
trap suite green (all rule-detected honeypots excluded and top-100 honeypot exposure
comfortably below 10%, stuffer ranked down, Tier-5 ranked up) and the precompute->rank
two-command reproduction documented in README.md. Then write a final summary to
claude-progress.txt and stop.

Constraints throughout: deterministic (PYTHONHASHSEED=0, stable hashing, single-thread
BLAS in the repro image, no clocks/locale/RNG). The ranking step's ONLY hard runtime
limits are CPU-only / no-network / <=5 min / <=16 GB / <=5 GB disk — any dependency that
respects them is allowed (numpy, scipy, pyarrow, pandas, joblib, etc. are all fine).
Pin all dependency versions.

RUNTIME ML POLICY (behavior, not a package blocklist): the ranking step is a frozen
scorer, not an ML pipeline. It must NOT instantiate, load, download, or execute neural
models — including transformers, sentence-transformers, Torch models, ONNX, GGUF, or any
other inference engine — to compute candidate-dependent outputs at runtime. ALL neural
computation (embedding generation, reranking, calibration fitting, any model inference)
must happen during offline preprocessing. At runtime the system may only load precomputed
artifacts (embeddings, indexes, feature matrices, lookup tables, calibration parameters)
and perform deterministic retrieval, filtering, scoring, ranking, and reasoning/CSV
generation.

DECISION RULES (use these to resolve any choice the prompt doesn't spell out):
- When in doubt, push computation into PRECOMPUTE rather than runtime. Runtime should be
  a lightweight artifact loader and scorer.
- If two implementations both satisfy the official competition rules, prefer the SIMPLER
  one — avoid runtime complexity or uncertainty (e.g. don't load a transformer at runtime
  just to compute embeddings that are already available offline).
```

---

*Method source: [Effective harnesses for long-running agents — Anthropic, Nov 26 2025](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents).*
