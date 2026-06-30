### Goal 6 — Ranking & selection — ✅ FULLY DETAILED & LOCKED (D1–D7)
**SCOPE:** the **decision policy** — the only goal that converts scores into the submission. Consumes Goal 4 fit bundle (`fit_score`, `engagement_modifier`, `risk_modifier`, per-criterion evidence trail) + Goal 5 risk primitives (`honeypot_flag`, `honeypot_reasons`, `consulting_gate_flag`, `consulting_gate_suppressed`, `suppression_evidence`). **Canonical flow: score everyone → mark eligibility → filter eligible → sort → take top 100 → (Goal 7 reasoning) → assemble final dataframe → validate → write CSV.** `final_score = fit_score × engagement_modifier × risk_modifier` is computed for **all** candidates; **selection is restricted to the eligible subset**.

**Data-profiling findings that shaped Goal 6 (all 100k, fresh re-run):**
- **Eligible after hard gates = 90,197** (9,803 ineligible) → ~900× headroom over the 100 needed; **filling the top-100 is never at risk.**
- **All 69 detected honeypots are excluded** (H1=25, H2=23, H3=21, disjoint OR) → detected honeypot rate in top-100 = **0** (≪10% satisfied automatically with headroom). **12 of the 69 carry relevant ML/AI titles** (the dangerous ones) — all excluded. The ~11 undetectable company-age honeypots are backstopped by Goal 4 corroboration-first scoring (no Tier-1/2 evidence → they sink), not by selection logic.
- **Consulting-entire-career = 9,745** before the product-stint override.
- **Zero true full-profile twins** (career-sequence + skillset + signal-envelope identical = 0 across 100k) → diversity/dedup is a genuine no-op.

**D1 — Eligibility & hard-gate enforcement (LOCKED): floor, never delete; derive eligibility only in Goal 6.**
- **Never delete** candidates from the master frame; **preserve all Goal 5 primitive detection outputs verbatim**.
- **Eligibility derived only inside Goal 6:** `hard_gate_flag = honeypot_flag OR (consulting_gate_flag AND NOT consulting_gate_suppressed)` — **not persisted** (a composed decision stays ephemeral, per Goal 5 D6). Select from the eligible subset.
- **`selection_manifest.json`** emits: total candidates, eligible count, excluded count, **exclusions by reason**, and **top excluded candidate scores** (audit artifact defending gate decisions).
- **Asserts:** exactly **100 rows**; **zero `honeypot_flag==true`**; **zero `hard_gate==true`** in the selected set.

**D2 — Output `score` column (LOCKED): emit `final_score`, no rescaling, with a deterministic tie-encoding micro-term.**
- Metrics use rank order vs hidden ground truth; the numeric score only needs to be **non-increasing with rank** (spec §3). Emit **`final_score`** — **no min-max scaling, no Goal-6 recalibration**. Calibration lives entirely in Goal 4 D5 (tail-anchored); Goal 6 presents only. Monotonicity guaranteed by sorting on the emitted key.
- **Tie-encoding (resolves the validator conflict — see D4):** corroboration breadth is folded into the printed score as a **deterministic, strictly-ordered sub-epsilon micro-term** (e.g. `printed_score = final_score − rank·δ`, δ ≪ smallest meaningful score gap), so the full deterministic sort is encoded *in the score itself* and printed scores are effectively unique. This keeps breadth influential **without** a post-score reordering that could violate `candidate_id` ordering. The micro-term preserves non-increasing-with-rank and is audit-documented; any residual exact-tie case (identical `final_score` **and** breadth) is broken by `candidate_id` ascending only.

**D3 — Honeypot ≪10% (LOCKED): automatic via hard exclusion, no quota juggling.**
- All detected honeypots excluded at D1 → detected rate = 0; **no constraint/quota selection logic**. Undetectable company-age honeypots backstopped architecturally (Goal 4 D7). The D1 assertion enforces zero honeypot in top-100. Diagnostic `stuffer_suspect_flag` survivors logged for audit only.

**D4 — Tie-breaking (LOCKED, REVISED): candidate_id ascending on equal printed score — validator-conformant.**
- **Why revised:** the provided `validate_submission.py` (lines 136–144) **rejects** any two rows with equal `score` that are not in **`candidate_id`-ascending** order. This is **stricter than the prose spec §3** ("a secondary signal from your model, *or* candidate_id ascending") and, being the Stage-1 auto-validator, is **ground truth**. A corroboration-breadth-first tie-break (prior D4) could place a higher `candidate_id` before a lower one at an equal printed score → auto-reject. Exact ties are most likely in the **filler tail** (ranks ~60–100), where modifiers collapse to neutral.
- **Resolution:** the deterministic **sort key** is `final_score` desc → corroboration breadth desc → `candidate_id` asc, but **breadth is expressed through the printed-score micro-term (D2)**, not as a post-score reordering. Therefore the **only** tie-break acting on an exactly-equal *printed* `score` is **`candidate_id` ascending** — exactly what the validator enforces. **`fit_score` is still never a tie-break.**
- Net effect: breadth still influences order (via the score), and the emitted CSV always satisfies `validate_submission.py`.

**D5 — Diversity / twins (LOCKED): no diversity constraint, no dedup.**
- Profiling shows **zero true full-profile duplicates** → no deduping required and none applied. NDCG/MAP don't reward diversity; collapsing templated near-duplicates would be harmful on this data. No twin safeguard shipped; the zero-twins finding is documented for Stage-5 defense.

**D6 — Filling exactly 100 (LOCKED): always fill, with honest filler.**
- Spec requires **exactly 100**; truly-relevant candidates are far fewer. Fill ranks past the relevant tier with the **best-available eligibles by `final_score`**. Goal 7 reasoning marks fillers honestly (≥1 limitation acknowledged) — matches the spec's rank-100 example and the Stage-4 rank-consistency check.

**D7 — Output contract & validation (LOCKED): validate the EXACT final dataframe, by invariants, with CSV round-trip.**
Pipeline: select → generate reasoning (Goal 7) → **assemble final dataframe → validate → write CSV**. Run the provided `validate_submission.py` **PLUS**:
1. **Final-artifact validation** — validate the exact dataframe that will be written (not an intermediate).
2. **Finite-score check** — all scores finite (no NaN/inf).
3. **Rank-order consistency** — recompute ordering with the production sort logic; assert emitted ranks match exactly.
4. **Eligibility consistency** — for every selected candidate assert `not honeypot_flag` and `not hard_gate_flag`, validated **directly against Goal 5 primitive flags** (not merely "absent from an exclusion list").
5. **Top-100 correctness** — `lowest_selected_score ≥ highest_unselected_score` (with tie-break consideration).
6. **Candidate alignment** — every selected id exists in `candidate_ids.npy`, Goal 1 parquet, and the Goal 3 feature matrix.
7. **Duplicate-score tie-break audit** — for any exactly-equal printed `score`, verify ordering is **`candidate_id` ascending** (per revised D4 — matches `validate_submission.py` lines 136–144; corroboration breadth is already encoded in the score via the D2 micro-term, `fit_score` intentionally excluded).
8. **Reasoning checks** — non-empty; uniqueness; **reasoning-rank consistency** (top candidates contain positive evidence; bottom fillers contain ≥1 limitation — catches reasoning inversion).
9. **CSV round-trip** — write a temp CSV, reload, re-run all spec checks, then emit the final file.
10. **Highest-scoring excluded candidate** recorded (id, score, exclusion reason) for gate-decision defense.
11. **Selection provenance** — every selected candidate must have a **fit bundle + evidence trail + reasoning**; none enters the top-100 without explainability artifacts.
Manifest discipline mirrors Goals 1–5: `selection_manifest.json` (config hash, counts, exclusions-by-reason, top-excluded, git commit, timestamp). **Config-agnostic across the 3-submission discipline** (Goal 4 D9) — Goal 6 logic is frozen; only Goal 4 weights vary between {safe, tuned, final}.

**STATUS: Goal 6 fully detailed & locked (D1–D7).** Output artifacts: `submission.csv` (`candidate_id, rank, score, reasoning`), `selection_manifest.json`. **Not yet built.**
**[Goal 7 addendum]:** Goal 6 **consumes `fit_tier` from Goal 4** (does not recompute it) and passes it into the Goal 7 contract; **`filler_flag ≡ fit_tier==Filler`** — the D6 "relevant-tier cutoff" is exactly the Filler boundary, so no separate filler computation is introduced.
**Goal 6 → carry-overs:** (1) Goal 7 reasoning is consumed at assembly time (validation #8, #11). (2) Goal 4 must surface a **corroboration-breadth count** in the fit bundle — Goal 6 folds it into the printed-score micro-term (D2), not a post-score tie-break (revised D4). (3) Goal 6 logic is frozen across all 3 submissions.
