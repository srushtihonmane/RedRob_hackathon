"""runtime_io.py — Goal 8: two-phase load, artifact<->pool binding, runtime_report, manifests.

R5 two-phase load: read the SMALL artifacts first (manifests, candidate_ids), run hash +
alignment + binding checks, and only then load the large matrices. A broken/misaligned/stale
artifact fails loudly in ~100 ms before hundreds of MB load.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .common import read_json, sha256_file, write_json, blas_thread_settings


class RuntimeReport:
    """Per-stage wall-clock + peak RSS tracker -> runtime_report.json (Goal 8 D7)."""
    def __init__(self):
        self.stages: list[dict] = []
        self._t0 = time.perf_counter()
        try:
            import psutil
            self._proc = psutil.Process()
        except Exception:
            self._proc = None
        self.peak_rss_mb = self._rss()

    def _rss(self) -> float:
        if self._proc is None:
            return 0.0
        rss = self._proc.memory_info().rss / 1e6
        self.peak_rss_mb = max(getattr(self, "peak_rss_mb", 0.0), rss)
        return rss

    def stage(self, name: str):
        return _StageTimer(self, name)

    def finalize(self, out_path: str, extra: dict | None = None) -> dict:
        rep = {
            "total_wall_s": round(time.perf_counter() - self._t0, 3),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "stages": self.stages,
            "thread_settings": blas_thread_settings(),
        }
        if extra:
            rep.update(extra)
        write_json(out_path, rep)
        return rep


class _StageTimer:
    def __init__(self, report: RuntimeReport, name: str):
        self.r = report; self.name = name

    def __enter__(self):
        self._t = time.perf_counter(); return self

    def __exit__(self, *exc):
        self.r._rss()
        self.r.stages.append({"stage": self.name,
                              "wall_s": round(time.perf_counter() - self._t, 3),
                              "rss_mb": round(self.r.peak_rss_mb, 1)})
        return False


def validate_and_bind(art_dir: str, candidates_path: str | None = None) -> dict:
    """Phase-1 validation (cheap): manifests present, alignment lengths, and the artifact<->pool
    source-hash binding (satisfies §10.3 'cover exactly the released pool'). Raises on failure."""
    art = Path(art_dir)
    fm = read_json(art / "feature_manifest.json")
    ids = np.load(art / "candidate_ids.npy")
    if len(ids) != fm["n_rows"]:
        raise AssertionError(f"candidate_ids ({len(ids)}) != feature_manifest n_rows ({fm['n_rows']})")
    # artifact<->pool binding
    if candidates_path is not None:
        ing = read_json(art / "ingest_manifest.json")
        src = sha256_file(candidates_path)
        if src != ing.get("source_sha256"):
            raise AssertionError("artifact<->pool binding failed: candidates file hash does not "
                                 "match ingest_manifest.source_sha256 (stale/wrong artifacts)")
    return {"n_rows": fm["n_rows"], "F": fm["F"]}


def write_build_manifest(art_dir: str, candidates_path: str, out_path: str,
                         submission_sha256: str | None = None, git_commit: str | None = None,
                         lib_versions: dict | None = None) -> dict:
    """build_manifest.json — the SOLE root referencing every sub-manifest + report (Goal 8 D9)."""
    art = Path(art_dir)
    def h(p):
        return sha256_file(art / p) if (art / p).exists() else None
    manifest = {
        "schema_version": 1,
        "candidates_sha256": sha256_file(candidates_path) if Path(candidates_path).exists() else None,
        "sub_manifests": {
            "ingest_manifest": str(art / "ingest_manifest.json"),
            "feature_manifest": str(art / "feature_manifest.json"),
            "repr_manifest": str(art / "repr_manifest.json"),
            "risk_manifest": str(art / "risk_manifest.json"),
            "jd_manifest": "jd/jd_manifest.json",
        },
        "artifact_hashes": {p: h(p) for p in
                            ("features.npy", "embeddings_identity.npy", "embeddings_evidence.npy",
                             "candidate_ids.npy", "normalization_stats.json")},
        "runtime_report": str(art / "runtime_report.json"),
        "submission_sha256": submission_sha256,
        "git_commit": git_commit,
        "lib_versions": lib_versions or {},
        "thread_settings": blas_thread_settings(),
    }
    write_json(out_path, manifest)
    return manifest
