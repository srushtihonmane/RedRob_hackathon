#!/usr/bin/env bash
# init.sh — environment setup + fast smoke test for the Redrob ranker.
# Two pinned dependency sets:
#   - Competition Runtime (lean, CPU-only, offline): requirements.runtime.txt
#   - Precompute + Sandbox (heavier, embedding deps):  requirements.precompute.txt
# This script verifies the runtime deps import, runs the test harness, and runs the
# official validator on any existing submission CSV. It is idempotent.
set -uo pipefail

PY="${PYTHON:-python}"
echo "== Python =="
"$PY" --version || { echo "FATAL: python not found"; exit 1; }

echo "== Install/verify Competition Runtime deps (lean) =="
# Idempotent; comment out if running fully offline with deps already present.
"$PY" -m pip install -q -r requirements.runtime.txt || echo "WARN: pip install skipped/failed (offline?) — continuing to import check"

echo "== Import check (runtime deps must import) =="
"$PY" - <<'PY'
import importlib, sys
mods = ["numpy", "scipy", "pyarrow", "pandas", "joblib"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append(f"{m}: {e}")
if bad:
    print("FATAL: missing runtime deps:\n  " + "\n  ".join(bad)); sys.exit(1)
print("runtime deps OK:", ", ".join(mods))
PY
[ $? -ne 0 ] && exit 1

echo "== Trap/invariant unit tests =="
if [ -d tests ] && ls tests/test_*.py >/dev/null 2>&1; then
  PYTHONHASHSEED=0 "$PY" -m pytest -q tests/ || { echo "FAIL: tests"; exit 1; }
else
  echo "(no tests yet — will be added during the build)"
fi

echo "== Official validator smoke (validate_submission.py) =="
CSV=""
[ -f submission.csv ] && CSV="submission.csv"
[ -z "$CSV" ] && [ -f sample_submission.csv ] && CSV="sample_submission.csv"
if [ -n "$CSV" ]; then
  "$PY" validate_submission.py "$CSV" || echo "NOTE: validator reported issues on $CSV (expected for placeholder/sample)"
else
  echo "(no CSV to validate yet)"
fi

echo "== init.sh smoke test complete =="
