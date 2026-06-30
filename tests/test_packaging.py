"""Goal 8 packaging (F8.9) + sandbox (F8.8) tests."""
import csv
import os
import subprocess
import sys

import pytest


def test_F8_9_packaging_files_present():
    for f in ("README.md", "requirements.runtime.txt", "requirements.precompute.txt",
              "submission_metadata.yaml", "Dockerfile.runtime", "precompute.py", "rank.py",
              "sandbox_rank.py"):
        assert os.path.exists(f), f"missing deliverable {f}"
    readme = open("README.md", encoding="utf-8").read()
    assert "python rank.py --candidates ./candidates.jsonl --out ./submission.csv" in readme
    assert "precompute.py" in readme
    meta = open("submission_metadata.yaml", encoding="utf-8").read()
    assert "reproduce_command" in meta and "has_network_during_ranking: false" in meta


def test_F8_9_runtime_requirements_have_no_neural_deps():
    req = open("requirements.runtime.txt", encoding="utf-8").read().lower()
    for banned in ("torch", "transformers", "sentence-transformers", "fastembed", "onnxruntime"):
        assert banned not in req, f"runtime deps must not include {banned}"


@pytest.mark.skipif(not os.path.exists("data/sample/artifacts/normalization_stats.json"),
                    reason="sample artifacts not built")
def test_F8_8_sandbox_runs_on_small_sample(tmp_path):
    # 25-candidate sandbox input (embeds on the fly; applies frozen pool stats)
    src = "data/sample/sample.jsonl"
    sample = tmp_path / "s25.jsonl"
    with open(src, encoding="utf-8") as f, open(sample, "w", encoding="utf-8", newline="\n") as g:
        for i, line in enumerate(f):
            if i >= 25:
                break
            g.write(line)
    out = tmp_path / "ranked.csv"
    env = dict(os.environ, EMBED_THREADS="8", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    r = subprocess.run([sys.executable, "sandbox_rank.py", "--sample", str(sample),
                        "--artifacts", "data/sample/artifacts", "--out", str(out)],
                       capture_output=True, text=True, env=env, timeout=280)
    assert r.returncode == 0, r.stderr[-500:]
    assert "FROZEN REF_DATE" in r.stdout         # proves shipped frozen stats are applied
    rows = list(csv.reader(open(out, encoding="utf-8")))
    assert rows[0] == ["candidate_id", "rank", "score", "reasoning"]
    assert 1 <= len(rows) - 1 <= 25
