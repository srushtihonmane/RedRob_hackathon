"""common.py — cross-cutting primitives shared by every goal.

Determinism, stable hashing, candidate_id <-> int conversion, file/JSON IO, and frozen
date math. Deliberately dependency-light (stdlib + numpy) so it is importable from both the
lean Competition Runtime and the heavier precompute/sandbox runtime.

DETERMINISM CONTRACT (runtime.md R3 / Goal 8 D6):
- Never use the builtin salted ``hash()`` for any ordering or bucketing. Use ``stable_hash``.
- BLAS is pinned single-thread in the canonical reproduction path; call
  ``configure_determinism()`` at the very top of an entrypoint *before* importing numpy.
- No clocks/locale/RNG influence candidate-dependent outputs. ``REF_DATE`` (a frozen pool
  statistic) replaces ``today()`` for all date math.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import zlib
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# Identity                                                                     #
# --------------------------------------------------------------------------- #
CANDIDATE_ID_RE = re.compile(r"^CAND_([0-9]{7})$")


def candidate_id_to_int(cid: str) -> int:
    """'CAND_0000001' -> 1. Raises ValueError on a malformed id."""
    m = CANDIDATE_ID_RE.match(cid)
    if not m:
        raise ValueError(f"malformed candidate_id: {cid!r}")
    return int(m.group(1))


def int_to_candidate_id(n: int) -> str:
    """1 -> 'CAND_0000001'. Inverse of candidate_id_to_int."""
    if not (0 <= n <= 9_999_999):
        raise ValueError(f"candidate int out of 7-digit range: {n}")
    return f"CAND_{n:07d}"


def is_valid_candidate_id(cid: str) -> bool:
    return bool(CANDIDATE_ID_RE.match(cid))


# --------------------------------------------------------------------------- #
# Stable hashing (NEVER the builtin hash())                                    #
# --------------------------------------------------------------------------- #
def stable_hash(s: str | bytes) -> int:
    """Deterministic, unsigned 32-bit hash via zlib.crc32. Stable across processes,
    platforms, and Python invocations (unlike the salted builtin ``hash``)."""
    if isinstance(s, str):
        s = s.encode("utf-8")
    return zlib.crc32(s) & 0xFFFFFFFF


def stable_unit_hash(s: str | bytes) -> float:
    """Stable hash mapped deterministically into [0, 1) — for sub-epsilon tiebreak jitter."""
    return stable_hash(s) / 0x1_0000_0000


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Determinism environment                                                      #
# --------------------------------------------------------------------------- #
_BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def configure_determinism(single_thread_blas: bool = True) -> None:
    """Set deterministic env. Call FIRST in an entrypoint, before importing numpy/scipy.

    - PYTHONHASHSEED=0 (belt-and-suspenders; our logic never relies on builtin hash()).
    - Single-thread BLAS removes residual float-reduction jitter in the repro image.
    Setting BLAS vars only takes effect if numpy has not yet imported its backend.
    """
    os.environ.setdefault("PYTHONHASHSEED", "0")
    if single_thread_blas:
        for var in _BLAS_THREAD_VARS:
            os.environ.setdefault(var, "1")


def blas_thread_settings() -> dict[str, str | None]:
    """Snapshot of thread env for the runtime_report / manifest provenance."""
    return {var: os.environ.get(var) for var in _BLAS_THREAD_VARS}


# --------------------------------------------------------------------------- #
# Frozen date math (REF_DATE replaces today())                                 #
# --------------------------------------------------------------------------- #
def parse_date(s: str | None) -> _dt.date | None:
    """Parse an ISO 'YYYY-MM-DD'. Returns None for null/empty/unparseable (caller decides)."""
    if not s:
        return None
    try:
        return _dt.date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


def days_between(ref: _dt.date, d: _dt.date | None) -> float | None:
    """Whole days from d to ref (ref - d). None if d is None. Frozen-ref date math."""
    if d is None:
        return None
    return float((ref - d).days)


def ref_date_from_dates(dates: Iterable[_dt.date]) -> _dt.date:
    """REF_DATE = max(last_active_date in pool) + 1 day (contracts.md C1 / Goal 3 D5)."""
    mx = None
    for d in dates:
        if d is not None and (mx is None or d > mx):
            mx = d
    if mx is None:
        raise ValueError("ref_date_from_dates: no valid dates provided")
    return mx + _dt.timedelta(days=1)


# --------------------------------------------------------------------------- #
# Deterministic JSON IO                                                        #
# --------------------------------------------------------------------------- #
def write_json(path: str | os.PathLike, obj: Any, *, sort_keys: bool = True) -> None:
    """UTF-8, sorted keys, trailing newline — byte-stable across runs. Atomic write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=sort_keys, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
