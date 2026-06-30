### Goal 1 — Data ingestion — ✅ DECIDED (detailing in progress)
**SCOPE (rescoped, tight):** Goal 1 is **pure ingestion** — parse the 100k JSONL **once** and persist a faithful, fast-loading copy so nothing downstream re-parses the 487 MB JSON. **No** filtering, classification, derived/interpretive features, scoring decisions, hot/cold column split, or embeddings. *Embeddings (`.npy`, float32 L2-normalized, brute-force cosine, no-FAISS), the lexical/BM25 doc, derived features, and the numeric feature matrix all moved to **Goal 3** (candidate representation).* (Rejected for ingestion: load-all-as-dicts — RAM risk.)

**D1 — Storage layout (LOCKED): single denormalized nested Parquet table, one row per candidate.**
- Nested arrays (`career_history`, `education`, `skills`, `certifications`, `languages`) stored as native Arrow/Parquet **`list<struct>`**; `skill_assessment_scores` stored as native **`map<string, value>`** (not dynamic columns, not JSON).
- This table is the **single source of truth** and must support **lossless reconstruction** of the original record **without joins or JSON parsing**.
- Optional **projection tables** (`skills_flat.parquet`, `career_history_flat.parquet`, …) may later be derived for cross-candidate analytics — **read-optimized views only, never authoritative**.
- Priorities: fast candidate-level loading, schema fidelity, minimal downstream reconstruction overhead, with a retained path to future analytical workloads.
- *Rejected:* normalized star schema (joins, no payoff — downstream access is row-wise per candidate); flat-scalars-plus-JSON-strings (re-incurs JSON parsing we're killing).

**D2 — Schema fidelity & type handling (LOCKED):**
- **Dates** → native `date32` (consistent ISO `YYYY-MM-DD`, lossless round-trip); `end_date` nullable preserved as true null; education `start_year`/`end_year` → `int16`; parse failures fall to the D5 quarantine path rather than being guessed.
- **`-1` sentinels** (`github_activity_score`, `offer_acceptance_rate`) → **preserved verbatim**, never converted to null (the "−1 means missing" semantics belong to Goal 3).
- **Optional/missing arrays** (`certifications`, `languages`, `education`) → **normalize absent → empty `list`** (schema `minItems:0` treats absent ≡ empty; the one documented normalization).
- **Low-cardinality enums** (`company_size`, `proficiency`, `tier`, `preferred_work_mode`, `industry`) → **dictionary-encoded** (transparent storage win, identical round-trip).
- **By type:** `years_of_experience`→`float32`, `duration_months`→`int16`, counts→`int32`, rates→`float32`, booleans→native `bool`, `skill_assessment_scores`→`map<string,float32>`, free text→`string`.

**D3 — Streaming write mechanics (LOCKED):**
- **Loop:** stream JSONL line-by-line → buffer parsed records → flush each buffer as one Arrow `RecordBatch` via an open `pyarrow.parquet.ParquetWriter` (only one batch resident at a time). *Rejected:* build one giant Arrow Table then write once — RAM spike with nested `list<struct>`.
- **Batch / row-group size:** **~5,000–10,000 candidates per batch** (~25–50 MB resident; ~10–20 row groups total).
- **Compression:** **Zstd (~level 3)** — better ratio than Snappy, decode cost irrelevant vs budget, compresses bulky text well.
- **Schema:** **explicit pinned Arrow schema** (per D2) constructed for every batch — never infer from data (avoids cross-batch type drift, e.g. all-null `end_date` or empty `certifications` in the first batch; more defensible at Stage 3).

**D4 — Canonical ordering & ID contract (LOCKED):**
- **Ordering:** **preserve source order** — row *i* = the *i*-th line of `candidates.jsonl` (deterministic, no sort pass, keeps the pure-streaming write). *Rejected:* sort-by-`candidate_id` (adds a full materialization/sort, no payoff — no id-locality relied on).
- **ID contract:** `candidate_id` is a column in the canonical table in row order = the authoritative index; all Goal 3 artifacts are built by iterating this table in order (alignment guaranteed by construction). Also emit a standalone **`candidate_ids`** key (`.npy` int32 of the 7-digit part / small parquet) for cheap alignment checks without loading the nested table.
- **Verification:** carry a `row_index` column; every downstream artifact derives from the same `candidate_ids` array; startup **assertion** that lengths match and ids line up (silent misalignment → loud failure).

**D5 — Integrity & reproducibility (LOCKED):**
- **Completeness assertions (hard, abort on failure):** exactly **100,000 rows**; `candidate_id` unique and matching `^CAND_[0-9]{7}$`; required fields present. No silent partial artifacts.
- **Malformed-record policy:** **quarantine, don't drop, don't crash.** Failing lines (with line number + error) → `quarantine.jsonl` + count in manifest. Policy gate: 0 quarantined → proceed; >0 → surface loudly, require human decision before artifact is valid. Expected 0 for this released synthetic pool.
- **Determinism:** explicit pinned Arrow schema (D3) + fixed source order (D4) + pinned lib versions (`pyarrow`, Python) in `requirements.txt` + a fixed reference-date constant for downstream date math; document the single regenerate command (mirrors Stage-3 reproduction).
- **Manifest:** emit `ingest_manifest.json` — row count, quarantine count, source filename + SHA-256, output SHA-256, schema fingerprint, lib versions, timestamp, git commit (Stage 3/4 artifact-authenticity evidence).

**STATUS: Goal 1 fully detailed & locked (D1–D5).** Output artifacts: `candidates.parquet` (canonical nested table), `candidate_ids` (alignment key), `ingest_manifest.json`, optional `quarantine.jsonl`. Next: Goal 2 (JD understanding).
**[Goal 8 addendum — LOCKED, verified non-breaking]:** the per-record **parse/normalize logic is factored as a reusable pure function** `parse_record(raw)→canonical struct`, **distinct from** the offline streaming Parquet-writer (D3), so the **Sandbox Runtime** (Goal 8 D8) can parse a ≤100-candidate sample straight into memory without the Parquet path. The builder boundary is the *record parser*, not the writer. Pool-level integrity (exactly-100k assertion, quarantine policy, ingest manifest — D5) stays in the **offline driver only** (it is meaningless on an arbitrary sample). No conflict with D1–D5: the streaming write still composes the same `parse_record` over the full file.
