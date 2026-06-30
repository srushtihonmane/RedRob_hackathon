"""jd_build.py — Goal 2: validate jd_query.json and generate the JD vectors + manifest.

Goal 2 declares WHAT to look for (typed criteria + evidence bindings + concept terms +
narratives). This module: (1) build-time-validates jd_query.json against candidate_schema.json
(every evidence_sources path must resolve; unique ids; valid enums) — no LLM/network at build;
(2) embeds the ideal narrative -> jd_ideal.npy and each anti-profile narrative -> jd_antiprofile.npy
(row order == antiprofile_labels), using the locked model + query prefix; (3) emits jd_manifest.json.

The .npy vectors are regenerable from the canonical narratives + the pinned model.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import read_json, sha256_file, write_json

CRITERION_TYPES = {"hard_gate", "strong_negative", "must_have", "nice_to_have", "context_modifier"}
IMPORTANCES = {"critical", "high", "medium", "low"}
TARGET_KINDS = {"presence", "range", "gate", "modifier"}
MATCH_MODES = {"lexical", "semantic", "structured", "signal"}
POLARITIES = {"+", "-"}


# --------------------------------------------------------------------------- #
# Evidence-path resolution against candidate_schema.json                       #
# --------------------------------------------------------------------------- #
def _schema_props(node: dict) -> dict | None:
    """Return the .properties dict for an object schema node (descending arrays via .items)."""
    if node.get("type") == "array" or "items" in node:
        node = node.get("items", {})
    return node.get("properties")


def evidence_path_exists(schema: dict, path: str) -> bool:
    """Resolve a dotted path with optional '[]' array markers against the JSON schema.
    e.g. 'career_history[].description', 'redrob_signals.skill_assessment_scores'."""
    props = schema.get("properties")
    if props is None:
        return False
    segs = path.split(".")
    for i, seg in enumerate(segs):
        key = seg[:-2] if seg.endswith("[]") else seg
        if props is None or key not in props:
            return False
        node = props[key]
        if i == len(segs) - 1:
            return True
        props = _schema_props(node)
    return True


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def validate_jd_query(jd: dict, schema: dict) -> list[str]:
    errs: list[str] = []
    for k in ("model", "criteria", "lexical_concept_groups", "ideal_narrative",
              "antiprofiles", "antiprofile_labels"):
        if k not in jd:
            errs.append(f"missing top-level key: {k}")
    if errs:
        return errs

    m = jd["model"]
    for k in ("name", "dim", "normalize", "query_prefix", "passage_prefix"):
        if k not in m:
            errs.append(f"model missing key: {k}")

    seen_ids: set[str] = set()
    for c in jd["criteria"]:
        cid = c.get("id", "<missing>")
        if cid in seen_ids:
            errs.append(f"duplicate criterion id: {cid}")
        seen_ids.add(cid)
        if c.get("criterion_type") not in CRITERION_TYPES:
            errs.append(f"{cid}: bad criterion_type {c.get('criterion_type')!r}")
        if c.get("importance") not in IMPORTANCES:
            errs.append(f"{cid}: bad importance {c.get('importance')!r}")
        if c.get("target_kind") not in TARGET_KINDS:
            errs.append(f"{cid}: bad target_kind {c.get('target_kind')!r}")
        if c.get("polarity") not in POLARITIES:
            errs.append(f"{cid}: bad polarity {c.get('polarity')!r}")
        modes = c.get("match_modes", [])
        if not modes or any(mm not in MATCH_MODES for mm in modes):
            errs.append(f"{cid}: bad match_modes {modes!r}")
        if not isinstance(c.get("corroboration", {}).get("min_sources"), int):
            errs.append(f"{cid}: corroboration.min_sources must be int")
        srcs = c.get("evidence_sources", [])
        if not srcs:
            errs.append(f"{cid}: evidence_sources is empty")
        for path in srcs:
            if not evidence_path_exists(schema, path):
                errs.append(f"{cid}: evidence_sources path not in candidate_schema: {path!r}")
        if not c.get("description"):
            errs.append(f"{cid}: missing description (jd_label)")

    labels = jd["antiprofile_labels"]
    ap_labels = [a["label"] for a in jd["antiprofiles"]]
    if labels != ap_labels:
        errs.append(f"antiprofile_labels {labels} != antiprofiles order {ap_labels}")
    if len(set(labels)) != len(labels):
        errs.append("antiprofile_labels not unique")
    return errs


# --------------------------------------------------------------------------- #
# Vector generation + manifest                                                 #
# --------------------------------------------------------------------------- #
def build_vectors(jd: dict, out_dir: Path, jd_query_path: str) -> tuple[Path, Path]:
    from . import embed  # precompute-only dependency
    ideal_texts = [jd["ideal_narrative"]] + list(jd.get("ideal_narrative_paraphrases", []))
    ideal_mat = embed.embed_queries(ideal_texts, jd_query_path=jd_query_path)
    ideal = ideal_mat.mean(axis=0, keepdims=True)
    ideal = ideal / np.maximum(np.linalg.norm(ideal, axis=1, keepdims=True), 1e-12)
    ideal = ideal.astype(np.float32)

    anti_texts = [a["narrative"] for a in jd["antiprofiles"]]  # already in label order
    anti = embed.embed_queries(anti_texts, jd_query_path=jd_query_path).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    ideal_path = out_dir / "jd_ideal.npy"
    anti_path = out_dir / "jd_antiprofile.npy"
    np.save(ideal_path, ideal)
    np.save(anti_path, anti)
    return ideal_path, anti_path


def build(jd_query_path: str = "jd/jd_query.json", out_dir: str = "jd",
          schema_path: str = "candidate_schema.json", git_commit: str | None = None) -> dict:
    jd = read_json(jd_query_path)
    schema = read_json(schema_path)
    errs = validate_jd_query(jd, schema)
    if errs:
        raise AssertionError("jd_query.json validation failed:\n  " + "\n  ".join(errs))

    out = Path(out_dir)
    ideal_path, anti_path = build_vectors(jd, out, jd_query_path)
    anti = np.load(anti_path)

    import fastembed  # for version provenance
    manifest = {
        "schema_version": jd.get("schema_version", 1),
        "model": jd["model"],
        "embedding_backend": {"library": "fastembed", "version": fastembed.__version__,
                              "note": "bge-small-en-v1.5 int8-quantized ONNX build"},
        "antiprofile_labels": jd["antiprofile_labels"],
        "n_criteria": len(jd["criteria"]),
        "dim": int(np.load(ideal_path).shape[1]),
        "n_antiprofiles": int(anti.shape[0]),
        "jd_query_sha256": sha256_file(jd_query_path),
        "jd_ideal_sha256": sha256_file(ideal_path),
        "jd_antiprofile_sha256": sha256_file(anti_path),
        "validation_status": "passed",
        "git_commit": git_commit,
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_json(out / "jd_manifest.json", manifest)
    return manifest
