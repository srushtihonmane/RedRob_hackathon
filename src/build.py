"""build.py — Goal 3 driver: dense embeddings + embedding-derived features + the feature
matrix writer. Composes the model-dependent half on top of represent.derive_structured.

Two dense vectors per candidate (Goal 3 D2):
  * identity = current_title + headline + summary (what they present as).
  * evidence = all role titles + query-aware top-relevance role descriptions (what they did).
role_relevance[i] = cosine(role_emb_i, jd_ideal); used to pick which descriptions go into the
evidence doc and to derive G9 role-relevance features. Embedding is PRECOMPUTE-ONLY.

The same ``assemble_features`` is used by the batch driver (full pool) and by the per-candidate
sandbox ``builder`` — identical raw features given the frozen inputs (contracts.md C11).
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Iterable

import numpy as np

from . import bm25 as bm25mod
from . import represent
from . import snippets as snippetsmod
from .common import parse_date, read_json, sha256_file, write_json
from .represent import FrozenInputs, FeatureRow

EVIDENCE_CHAR_BUDGET = 1800
EMBED_CHUNK = 200


# --------------------------------------------------------------------------- #
# Text composition                                                             #
# --------------------------------------------------------------------------- #
def identity_doc(rec: dict) -> str:
    p = rec["profile"]
    return f"{p.get('current_title','')}. {p.get('headline','')}. {p.get('summary','')}".strip()


def role_doc(role: dict) -> str:
    return f"{role.get('title','')}. {role.get('description','')}".strip()


def _concept_hit_count(text: str, all_keywords: list[str]) -> int:
    t = (text or "").lower()
    return sum(1 for k in all_keywords if k in t)


def _recent_start(role: dict):
    sd = role.get("start_date")
    return represent.parse_date(sd) if isinstance(sd, str) else sd


def evidence_doc(rec: dict, all_keywords: list[str]) -> str:
    """Evidence doc = all role titles + descriptions ordered query-aware by lexical concept-hit
    count (then recency), truncated to the token budget. (Replaces the per-role-embedding
    relevance ordering — a faithful, ~3x-cheaper approximation: for the ~95% of candidates whose
    descriptions fit 512 tokens the doc is identical; only the overflow tail reorders.)"""
    roles = rec["career_history"]
    titles = [r.get("title", "") for r in roles]
    order = sorted(range(len(roles)),
                   key=lambda i: (-_concept_hit_count(f"{titles[i]} {roles[i].get('description','')}", all_keywords),
                                  -(_recent_start(roles[i]) or represent._dt.date.min).toordinal()))
    parts = [" . ".join(t for t in titles if t)]
    budget = EVIDENCE_CHAR_BUDGET
    for i in order:
        desc = roles[i].get("description", "")
        if not desc:
            continue
        parts.append(desc[:budget])
        budget -= len(desc)
        if budget <= 0:
            break
    return " . ".join(p for p in parts if p).strip()


# --------------------------------------------------------------------------- #
# Embedding-derived features (model-dependent half)                            #
# --------------------------------------------------------------------------- #
def _recent_role_index(roles: list[dict]) -> int:
    best, bi = None, 0
    for i, r in enumerate(roles):
        sd = r.get("start_date")
        if isinstance(sd, str):
            sd = represent.parse_date(sd)
        if sd is not None and (best is None or sd > best):
            best, bi = sd, i
    return bi


def add_embedding_features(row: FeatureRow, rec: dict, identity_vec: np.ndarray,
                           evidence_vec: np.ndarray, fz: FrozenInputs) -> None:
    ideal = fz.jd_ideal[0]
    ev_ideal = float(evidence_vec @ ideal)
    row.add("identity_cos_ideal", float(identity_vec @ ideal))
    row.add("evidence_cos_ideal", ev_ideal)
    label_cos: dict[str, float] = {}
    for i, label in enumerate(fz.antiprofile_labels):
        anti = fz.jd_antiprofile[i]
        row.add(f"identity_cos_{label}", float(identity_vec @ anti))
        ec = float(evidence_vec @ anti)
        row.add(f"evidence_cos_{label}", ec)
        label_cos[label] = ec
    # G9 role-relevance features (unused by scoring) -> overall evidence relevance proxy,
    # since per-role embeddings were dropped for throughput. Columns retained for stability.
    row.add("max_role_relevance", ev_ideal)
    row.add("mean_role_relevance", ev_ideal)
    row.add("recent_role_relevance", ev_ideal)
    row.add("cv_speech_vs_nlp_lean", float(label_cos.get("cv_speech_robotics", 0.0) - ev_ideal))
    row.add("identity_evidence_divergence", float(1.0 - (identity_vec @ evidence_vec)))


def assemble_features(rec: dict, identity_vec: np.ndarray, evidence_vec: np.ndarray,
                      fz: FrozenInputs) -> FeatureRow:
    row = represent.derive_structured(rec, fz)
    add_embedding_features(row, rec, identity_vec, evidence_vec, fz)
    return row


# --------------------------------------------------------------------------- #
# Batched embedding of a chunk of candidates                                   #
# --------------------------------------------------------------------------- #
def _all_keywords(fz: FrozenInputs) -> list[str]:
    kws: list[str] = []
    for cdef in fz.concepts.values():
        kws += cdef["keywords"]
    return kws


def embed_chunk(records: list[dict], fz: FrozenInputs, jd_query_path: str):
    """Return (identity_vecs[n,384], evidence_vecs[n,384]). One embed pass over identity +
    evidence docs (2 per candidate; per-role embeddings dropped for throughput)."""
    from . import embed
    n = len(records)
    kws = _all_keywords(fz)
    id_docs = [identity_doc(r) for r in records]
    ev_docs = [evidence_doc(r, kws) for r in records]
    vecs = embed.embed_passages(id_docs + ev_docs, jd_query_path=jd_query_path)
    return vecs[:n], vecs[n:]


def builder(candidate: dict, fz: FrozenInputs, jd_query_path: str = "jd/jd_query.json") -> dict:
    """Per-candidate Sandbox builder: identical raw features for one candidate from frozen
    inputs. Returns {features: FeatureRow, identity_vec, evidence_vec}."""
    iv, ev = embed_chunk([candidate], fz, jd_query_path)
    row = assemble_features(candidate, iv[0], ev[0], fz)
    return {"features": row, "identity_vec": iv[0], "evidence_vec": ev[0]}


# --------------------------------------------------------------------------- #
# Driver: feature matrix + embeddings (sample or full pool)                    #
# --------------------------------------------------------------------------- #
def build_features_and_embeddings(records: Iterable[dict], fz: FrozenInputs, out_dir: str,
                                  jd_query_path: str = "jd/jd_query.json",
                                  chunk: int = EMBED_CHUNK):
    """Stream records -> features.npy, embeddings_identity.npy, embeddings_evidence.npy, and
    the ordered feature-name list. Returns (feature_names, n, role_relevances_by_row)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    feature_names: list[str] | None = None
    feat_rows: list[list[float]] = []
    id_list: list[np.ndarray] = []
    ev_list: list[np.ndarray] = []

    buf: list[dict] = []

    def flush():
        nonlocal feature_names
        if not buf:
            return
        iv, ev = embed_chunk(buf, fz, jd_query_path)
        for i, rec in enumerate(buf):
            row = assemble_features(rec, iv[i], ev[i], fz)
            if feature_names is None:
                feature_names = list(row.values.keys())
            elif list(row.values.keys()) != feature_names:
                raise AssertionError(f"feature key mismatch at {rec['candidate_id']}")
            feat_rows.append([row.values[k] for k in feature_names])
            id_list.append(iv[i])
            ev_list.append(ev[i])
        buf.clear()

    for rec in records:
        buf.append(rec)
        if len(buf) >= chunk:
            flush()
    flush()

    feats = np.asarray(feat_rows, dtype=np.float32)
    np.save(out / "features.npy", feats)
    np.save(out / "embeddings_identity.npy", np.asarray(id_list, dtype=np.float32))
    np.save(out / "embeddings_evidence.npy", np.asarray(ev_list, dtype=np.float32))
    return feature_names, feats.shape[0], None


# --------------------------------------------------------------------------- #
# Feature grouping / manifest / normalization                                  #
# --------------------------------------------------------------------------- #
_GROUP_RULES = [
    ("present_flag", lambda n: n.endswith("_present")),
    ("G1_experience", lambda n: n in {"yoe", "summed_tenure_yrs", "n_roles"} or n.endswith("_role_tenure_mo")),
    ("G2_jobhop", lambda n: n in {"count_stints_under_18mo", "frac_short_stints", "switches_per_year", "mean_tenure_at_exit"}),
    ("G3_company", lambda n: n in {"any_product_stint", "frac_career_services", "is_consulting_entire_career", "product_tenure_yrs", "max_company_size_at_product", "largest_employer_fraction"}),
    ("G4_applied_ml", lambda n: n in {"kw_ml_role_tenure_yrs", "ml_role_count"}),
    ("G6_education", lambda n: n in {"highest_tier_ordinal", "has_tier1", "highest_degree_level", "has_phd", "field_relevance", "latest_grad_year"}),
    ("G7_consistency", lambda n: n in {"yoe_minus_summed_tenure", "expert_skill_zero_duration_count", "skill_duration_exceeds_career_flag", "timeline_overlap_months", "gradyear_vs_yoe_gap"}),
    ("G8_location", lambda n: n in {"is_noida_pune", "is_tier1_indian_city", "is_india"}),
    ("signals", lambda n: n.startswith("sig_")),
    ("corroboration", lambda n: n.startswith("assess_") or n.startswith("skillmeta_") or n.startswith("title_hit_")),
    ("G5_G9_G10_embedding", lambda n: n.startswith("identity_cos_") or n.startswith("evidence_cos_") or n.endswith("_role_relevance") or n in {"cv_speech_vs_nlp_lean", "identity_evidence_divergence"}),
]


def feature_group(name: str) -> str:
    for group, rule in _GROUP_RULES:
        if rule(name):
            return group
    return "other"


def compute_ref_date(records) -> _dt.date:
    """REF_DATE = max(last_active_date in pool) + 1 day (contracts C1)."""
    from .common import ref_date_from_dates
    dates = []
    for rec in records:
        la = rec["redrob_signals"].get("last_active_date")
        la = parse_date(la) if isinstance(la, str) else la
        if la is not None:
            dates.append(la)
    return ref_date_from_dates(dates)


def _normalization_stats(feats: np.ndarray, names: list[str], nullable: set[str]) -> dict:
    stats = {}
    pct = [1, 5, 25, 50, 75, 95, 99]
    for j, nm in enumerate(names):
        col = feats[:, j]
        finite = col[~np.isnan(col)]
        d = {"count": int(finite.size), "missing": int(np.isnan(col).sum())}
        if finite.size:
            d["min"] = float(finite.min()); d["max"] = float(finite.max())
            d["mean"] = float(finite.mean()); d["std"] = float(finite.std())
            for p in pct:
                d[f"p{p}"] = float(np.percentile(finite, p))
        stats[nm] = d
    return stats


def build_representation(parquet_path: str, candidate_ids_path: str, out_dir: str,
                         ref_date: _dt.date | None = None, jd_query_path: str = "jd/jd_query.json",
                         git_commit: str | None = None) -> dict:
    """Goal 3 capstone: build ALL representation artifacts from Goal 1's parquet, aligned by
    row order. Returns the repr_manifest dict."""
    from .ingest import read_parquet_records
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = read_parquet_records(parquet_path)
    if ref_date is None:
        ref_date = compute_ref_date(records)
    fz = FrozenInputs.load(ref_date, jd_query_path=jd_query_path)
    if fz.jd_ideal is None:
        raise AssertionError("JD vectors not found — run jd_build.build first")

    names, n, _rr = build_features_and_embeddings(records, fz, out_dir, jd_query_path)
    feats = np.load(out / "features.npy")

    # NaN/_present nullable set: any feature with a companion _present flag.
    nullable = {nm for nm in names if nm + "_present" in names}

    # features.parquet (named cols, native nulls for inspection)
    cols = {nm: pa.array([None if np.isnan(v) else float(v) for v in feats[:, j]], pa.float32())
            for j, nm in enumerate(names)}
    cols = {"candidate_id": pa.array([r["candidate_id"] for r in records], pa.string()), **cols}
    pq.write_table(pa.table(cols), out / "features.parquet", compression="zstd")

    # normalization_stats.json
    write_json(out / "normalization_stats.json", _normalization_stats(feats, names, nullable))

    # feature_manifest.json
    feature_manifest = {
        "schema_version": 1, "F": len(names), "n_rows": n,
        "ref_date": ref_date.isoformat(),
        "antiprofile_label_order": fz.antiprofile_labels,
        "model": fz.model,
        "company_table_sha256": sha256_file("reference/company_table.json"),
        "concept_registry_sha256": sha256_file("reference/concept_registry.json"),
        "columns": [{"index": j, "name": nm, "group": feature_group(nm),
                     "nullable": nm in nullable, "is_present_flag": nm.endswith("_present")}
                    for j, nm in enumerate(names)],
    }
    write_json(out / "feature_manifest.json", feature_manifest)

    # BM25 + snippets
    bm = bm25mod.build_index(records, out_dir, jd_query_path)
    sn = snippetsmod.build_sidecar(records, fz, out_dir)

    # Alignment assertion (loud fail) — row i lines up everywhere.
    ids = np.load(candidate_ids_path)
    from .common import candidate_id_to_int
    assert feats.shape[0] == n == len(ids), "feature/ids length mismatch"
    assert np.load(out / "embeddings_identity.npy").shape[0] == n
    assert np.load(out / "embeddings_evidence.npy").shape[0] == n
    assert sn["n_rows"] == n
    for i, rec in enumerate(records):
        if candidate_id_to_int(rec["candidate_id"]) != int(ids[i]):
            raise AssertionError(f"alignment mismatch at row {i}")

    repr_manifest = {
        "schema_version": 1, "n_rows": n, "F": len(names), "ref_date": ref_date.isoformat(),
        "features_sha256": sha256_file(out / "features.npy"),
        "embeddings_identity_sha256": sha256_file(out / "embeddings_identity.npy"),
        "embeddings_evidence_sha256": sha256_file(out / "embeddings_evidence.npy"),
        "feature_manifest_sha256": sha256_file(out / "feature_manifest.json"),
        "normalization_stats_sha256": sha256_file(out / "normalization_stats.json"),
        "bm25": bm, "snippets": sn,
        "company_table_sha256": sha256_file("reference/company_table.json"),
        "concept_registry_sha256": sha256_file("reference/concept_registry.json"),
        "model": fz.model, "git_commit": git_commit,
    }
    write_json(out / "repr_manifest.json", repr_manifest)
    return repr_manifest
