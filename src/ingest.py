"""ingest.py — Goal 1: pure data ingestion.

Parse the 100k-line JSONL ONCE into a faithful, fast-loading nested Parquet table so nothing
downstream re-parses the 487 MB JSON. NO derivation/representation here (that is Goal 3).

Two halves, deliberately separated (contracts.md C11 / Goal 1 D3 + Goal 8 addendum):
  * ``parse_record(raw) -> canonical struct`` — a PURE, pool-independent function (also the
    Sandbox-runtime parse path). One documented normalization: absent optional arrays -> [].
  * ``ingest_file(...)`` — the offline streaming driver (ParquetWriter, integrity asserts,
    quarantine-not-drop, candidate_ids.npy, ingest_manifest.json). Pool-level only.

Storage (Goal 1 D1/D2): single denormalized nested table, one row/candidate, SOURCE ORDER.
Nested arrays as ``list<struct>``; ``skill_assessment_scores`` as ``map<string,float32>``;
dates as ``date32`` (``end_date`` nullable); ``-1`` sentinels preserved verbatim.

Deviation log (faithful simplification): low-cardinality enums are stored as plain UTF-8
strings rather than dictionary-encoded (Goal 1 D2). Round-trip is identical; this avoids
cross-batch dictionary-unification pitfalls inside nested structs. Storage cost is trivial
vs the 5 GB budget. Logged in claude-progress.txt.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .common import (CANDIDATE_ID_RE, candidate_id_to_int, parse_date, sha256_file,
                     write_json)

SCHEMA_VERSION = 1
DEFAULT_BATCH = 8000  # ~5-10k/candidate batch (Goal 1 D3)

# --------------------------------------------------------------------------- #
# Pinned Arrow schema (explicit; never inferred — Goal 1 D3)                   #
# --------------------------------------------------------------------------- #
_STR = pa.string()
_F32 = pa.float32()
_I16 = pa.int16()
_I32 = pa.int32()
_BOOL = pa.bool_()
_DATE = pa.date32()

_PROFILE = pa.struct([
    ("anonymized_name", _STR), ("headline", _STR), ("summary", _STR), ("location", _STR),
    ("country", _STR), ("years_of_experience", _F32), ("current_title", _STR),
    ("current_company", _STR), ("current_company_size", _STR), ("current_industry", _STR),
])
_ROLE = pa.struct([
    ("company", _STR), ("title", _STR), ("start_date", _DATE), ("end_date", _DATE),
    ("duration_months", _I16), ("is_current", _BOOL), ("industry", _STR),
    ("company_size", _STR), ("description", _STR),
])
_EDU = pa.struct([
    ("institution", _STR), ("degree", _STR), ("field_of_study", _STR),
    ("start_year", _I16), ("end_year", _I16), ("grade", _STR), ("tier", _STR),
])
_SKILL = pa.struct([
    ("name", _STR), ("proficiency", _STR), ("endorsements", _I32), ("duration_months", _I16),
])
_CERT = pa.struct([("name", _STR), ("issuer", _STR), ("year", _I16)])
_LANG = pa.struct([("language", _STR), ("proficiency", _STR)])
_SALARY = pa.struct([("min", _F32), ("max", _F32)])
_ASSESS = pa.map_(_STR, _F32)
_SIGNALS = pa.struct([
    ("profile_completeness_score", _F32), ("signup_date", _DATE), ("last_active_date", _DATE),
    ("open_to_work_flag", _BOOL), ("profile_views_received_30d", _I32),
    ("applications_submitted_30d", _I32), ("recruiter_response_rate", _F32),
    ("avg_response_time_hours", _F32), ("skill_assessment_scores", _ASSESS),
    ("connection_count", _I32), ("endorsements_received", _I32), ("notice_period_days", _I16),
    ("expected_salary_range_inr_lpa", _SALARY), ("preferred_work_mode", _STR),
    ("willing_to_relocate", _BOOL), ("github_activity_score", _F32),
    ("search_appearance_30d", _I32), ("saved_by_recruiters_30d", _I32),
    ("interview_completion_rate", _F32), ("offer_acceptance_rate", _F32),
    ("verified_email", _BOOL), ("verified_phone", _BOOL), ("linkedin_connected", _BOOL),
])

ARROW_SCHEMA = pa.schema([
    ("candidate_id", _STR), ("row_index", _I32), ("profile", _PROFILE),
    ("career_history", pa.list_(_ROLE)), ("education", pa.list_(_EDU)),
    ("skills", pa.list_(_SKILL)), ("certifications", pa.list_(_CERT)),
    ("languages", pa.list_(_LANG)), ("redrob_signals", _SIGNALS),
])


# --------------------------------------------------------------------------- #
# Pure parse (the Sandbox-runtime parse path)                                  #
# --------------------------------------------------------------------------- #
def _req(d: dict, k: str) -> Any:
    if k not in d:
        raise ValueError(f"missing required field {k!r}")
    return d[k]


def _f(x: Any) -> float | None:
    return None if x is None else float(x)


def _i(x: Any) -> int | None:
    return None if x is None else int(x)


def _parse_profile(p: dict) -> dict:
    return {
        "anonymized_name": str(_req(p, "anonymized_name")),
        "headline": str(_req(p, "headline")), "summary": str(_req(p, "summary")),
        "location": str(_req(p, "location")), "country": str(_req(p, "country")),
        "years_of_experience": _f(_req(p, "years_of_experience")),
        "current_title": str(_req(p, "current_title")),
        "current_company": str(_req(p, "current_company")),
        "current_company_size": str(_req(p, "current_company_size")),
        "current_industry": str(_req(p, "current_industry")),
    }


def _parse_role(r: dict) -> dict:
    sd = parse_date(_req(r, "start_date"))
    if sd is None:
        raise ValueError(f"unparseable start_date: {r.get('start_date')!r}")
    return {
        "company": str(_req(r, "company")), "title": str(_req(r, "title")),
        "start_date": sd, "end_date": parse_date(r.get("end_date")),
        "duration_months": _i(_req(r, "duration_months")), "is_current": bool(_req(r, "is_current")),
        "industry": str(_req(r, "industry")), "company_size": str(_req(r, "company_size")),
        "description": str(_req(r, "description")),
    }


def _parse_edu(e: dict) -> dict:
    return {
        "institution": str(_req(e, "institution")), "degree": str(_req(e, "degree")),
        "field_of_study": str(_req(e, "field_of_study")),
        "start_year": _i(e.get("start_year")), "end_year": _i(e.get("end_year")),
        "grade": None if e.get("grade") is None else str(e["grade"]),
        "tier": None if e.get("tier") is None else str(e["tier"]),
    }


def _parse_skill(s: dict) -> dict:
    return {
        "name": str(_req(s, "name")), "proficiency": str(_req(s, "proficiency")),
        "endorsements": _i(_req(s, "endorsements")),
        "duration_months": _i(s.get("duration_months")),
    }


def _parse_cert(c: dict) -> dict:
    return {"name": str(_req(c, "name")), "issuer": str(_req(c, "issuer")),
            "year": _i(_req(c, "year"))}


def _parse_lang(l: dict) -> dict:
    return {"language": str(_req(l, "language")), "proficiency": str(_req(l, "proficiency"))}


def _parse_signals(s: dict) -> dict:
    assess = _req(s, "skill_assessment_scores") or {}
    sal = _req(s, "expected_salary_range_inr_lpa") or {}
    return {
        "profile_completeness_score": _f(_req(s, "profile_completeness_score")),
        "signup_date": parse_date(_req(s, "signup_date")),
        "last_active_date": parse_date(_req(s, "last_active_date")),
        "open_to_work_flag": bool(_req(s, "open_to_work_flag")),
        "profile_views_received_30d": _i(_req(s, "profile_views_received_30d")),
        "applications_submitted_30d": _i(_req(s, "applications_submitted_30d")),
        "recruiter_response_rate": _f(_req(s, "recruiter_response_rate")),
        "avg_response_time_hours": _f(_req(s, "avg_response_time_hours")),
        "skill_assessment_scores": {str(k): float(v) for k, v in assess.items()},
        "connection_count": _i(_req(s, "connection_count")),
        "endorsements_received": _i(_req(s, "endorsements_received")),
        "notice_period_days": _i(_req(s, "notice_period_days")),
        "expected_salary_range_inr_lpa": {"min": _f(sal.get("min")), "max": _f(sal.get("max"))},
        "preferred_work_mode": str(_req(s, "preferred_work_mode")),
        "willing_to_relocate": bool(_req(s, "willing_to_relocate")),
        "github_activity_score": _f(_req(s, "github_activity_score")),
        "search_appearance_30d": _i(_req(s, "search_appearance_30d")),
        "saved_by_recruiters_30d": _i(_req(s, "saved_by_recruiters_30d")),
        "interview_completion_rate": _f(_req(s, "interview_completion_rate")),
        "offer_acceptance_rate": _f(_req(s, "offer_acceptance_rate")),
        "verified_email": bool(_req(s, "verified_email")),
        "verified_phone": bool(_req(s, "verified_phone")),
        "linkedin_connected": bool(_req(s, "linkedin_connected")),
    }


def parse_record(raw: dict) -> dict:
    """Pure normalization of one raw candidate dict -> canonical struct (dates as date objects,
    absent optional arrays -> []). Raises ValueError on malformed/missing-required (the caller
    decides quarantine vs abort). Pool-independent — safe for the Sandbox runtime."""
    cid = str(_req(raw, "candidate_id"))
    if not CANDIDATE_ID_RE.match(cid):
        raise ValueError(f"bad candidate_id: {cid!r}")
    career = [_parse_role(x) for x in _req(raw, "career_history")]
    if not career:
        raise ValueError("career_history must have >=1 role")
    return {
        "candidate_id": cid,
        "profile": _parse_profile(_req(raw, "profile")),
        "career_history": career,
        "education": [_parse_edu(x) for x in (raw.get("education") or [])],
        "skills": [_parse_skill(x) for x in (raw.get("skills") or [])],
        "certifications": [_parse_cert(x) for x in (raw.get("certifications") or [])],
        "languages": [_parse_lang(x) for x in (raw.get("languages") or [])],
        "redrob_signals": _parse_signals(_req(raw, "redrob_signals")),
    }


# --------------------------------------------------------------------------- #
# Offline streaming driver (pool-level)                                        #
# --------------------------------------------------------------------------- #
def _batch_to_arrow(records: list[dict], start_index: int) -> pa.RecordBatch:
    n = len(records)
    arrays = [
        pa.array([r["candidate_id"] for r in records], _STR),
        pa.array(list(range(start_index, start_index + n)), _I32),
        pa.array([r["profile"] for r in records], _PROFILE),
        pa.array([r["career_history"] for r in records], pa.list_(_ROLE)),
        pa.array([r["education"] for r in records], pa.list_(_EDU)),
        pa.array([r["skills"] for r in records], pa.list_(_SKILL)),
        pa.array([r["certifications"] for r in records], pa.list_(_CERT)),
        pa.array([r["languages"] for r in records], pa.list_(_LANG)),
        pa.array([_signals_for_arrow(r["redrob_signals"]) for r in records], _SIGNALS),
    ]
    return pa.RecordBatch.from_arrays(arrays, schema=ARROW_SCHEMA)


def _signals_for_arrow(sig: dict) -> dict:
    """skill_assessment_scores must be list-of-(k,v) for the Arrow map type."""
    out = dict(sig)
    out["skill_assessment_scores"] = list(sig["skill_assessment_scores"].items())
    return out


def iter_jsonl(path: str | os.PathLike) -> Iterator[tuple[int, str]]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if line.strip():
                yield lineno, line


def iter_parquet_records(path: str | os.PathLike, batch_size: int = 4000):
    """Stream canonical records from candidates.parquet one row-group batch at a time
    (memory-frugal — never materializes the whole 100k table). Yields dicts in row order."""
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            row.pop("row_index", None)
            sig = row["redrob_signals"]
            sig["skill_assessment_scores"] = {k: float(v) for k, v in (sig["skill_assessment_scores"] or [])}
            yield row


def read_parquet_records(path: str | os.PathLike) -> list[dict]:
    """Reload candidates.parquet into canonical structs matching ``parse_record`` output
    (map -> dict, ``row_index`` dropped). Used by round-trip checks and the Sandbox path."""
    table = pq.read_table(path)
    out = []
    for row in table.to_pylist():
        row.pop("row_index", None)
        sig = row["redrob_signals"]
        assess = sig["skill_assessment_scores"]
        # pyarrow map -> list of (k, v) tuples; normalize back to a dict.
        sig["skill_assessment_scores"] = {k: float(v) for k, v in (assess or [])}
        out.append(row)
    return out


def ingest_file(candidates_path: str | os.PathLike, out_dir: str | os.PathLike,
                *, expected_n: int | None = None, batch_size: int = DEFAULT_BATCH,
                git_commit: str | None = None) -> dict:
    """Stream candidates.jsonl -> candidates.parquet (+ candidate_ids.npy + ingest_manifest.json).
    Quarantine malformed lines (don't drop, don't crash). Returns the manifest dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "candidates.parquet"
    ids_path = out_dir / "candidate_ids.npy"
    quarantine_path = out_dir / "quarantine.jsonl"
    manifest_path = out_dir / "ingest_manifest.json"

    ids: list[int] = []
    seen: set[str] = set()
    quarantined: list[dict] = []
    buf: list[dict] = []
    written = 0

    writer = pq.ParquetWriter(parquet_path, ARROW_SCHEMA, compression="zstd",
                              compression_level=3)
    try:
        def flush():
            nonlocal written, buf
            if not buf:
                return
            writer.write_batch(_batch_to_arrow(buf, written))
            written += len(buf)
            buf = []

        for lineno, line in iter_jsonl(candidates_path):
            try:
                rec = parse_record(json.loads(line))
            except Exception as e:  # malformed -> quarantine, keep going
                quarantined.append({"line": lineno, "error": str(e)})
                continue
            cid = rec["candidate_id"]
            if cid in seen:
                quarantined.append({"line": lineno, "error": f"duplicate candidate_id {cid}"})
                continue
            seen.add(cid)
            ids.append(candidate_id_to_int(cid))
            buf.append(rec)
            if len(buf) >= batch_size:
                flush()
        flush()
    finally:
        writer.close()

    np.save(ids_path, np.asarray(ids, dtype=np.int32))
    if quarantined:
        with open(quarantine_path, "w", encoding="utf-8", newline="\n") as f:
            for q in quarantined:
                f.write(json.dumps(q) + "\n")

    # Completeness assertions (Goal 1 D5). expected_n=None on arbitrary samples.
    if expected_n is not None and written != expected_n:
        raise AssertionError(f"expected {expected_n} rows, wrote {written}")
    if written == 0:
        raise AssertionError("no rows written")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "rows_written": written,
        "quarantined": len(quarantined),
        "quarantine_examples": quarantined[:20],
        "source_path": str(candidates_path),
        "source_sha256": sha256_file(candidates_path),
        "parquet_sha256": sha256_file(parquet_path),
        "candidate_ids_sha256": sha256_file(ids_path),
        "schema_fingerprint": str(ARROW_SCHEMA),
        "lib_versions": {"pyarrow": pa.__version__, "numpy": np.__version__},
        "git_commit": git_commit,
    }
    write_json(manifest_path, manifest)
    return manifest
