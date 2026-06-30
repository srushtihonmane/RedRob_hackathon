"""Goal 3 artifact integration tests — validate the representation artifacts built on the
sample pool (data/sample/artifacts). Skips cleanly if not built yet."""
import json
import math
import os

import numpy as np
import pytest

from src.common import read_json, candidate_id_to_int

ART = "data/sample/artifacts"
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ART, "features.npy")),
    reason="sample representation not built (run build.build_representation)")


def test_feature_matrix_matches_manifest():
    feats = np.load(os.path.join(ART, "features.npy"))
    fm = read_json(os.path.join(ART, "feature_manifest.json"))
    assert feats.dtype == np.float32
    assert feats.shape[1] == fm["F"] == len(fm["columns"])
    assert feats.shape[0] == fm["n_rows"]
    groups = {c["group"] for c in fm["columns"]}
    for g in ("G1_experience", "G3_company", "G7_consistency", "signals",
              "corroboration", "G5_G9_G10_embedding"):
        assert g in groups, f"missing feature group {g}"


def test_nan_present_invariant_full_matrix():
    feats = np.load(os.path.join(ART, "features.npy"))
    fm = read_json(os.path.join(ART, "feature_manifest.json"))
    name_to_idx = {c["name"]: c["index"] for c in fm["columns"]}
    for c in fm["columns"]:
        if c["nullable"]:
            val = feats[:, c["index"]]
            present = feats[:, name_to_idx[c["name"] + "_present"]]
            # _present==0  <=>  value is NaN
            assert np.all(np.isnan(val) == (present == 0.0)), f"NaN/_present mismatch on {c['name']}"


def test_embeddings_shape_and_norm():
    n = np.load(os.path.join(ART, "features.npy")).shape[0]
    for f in ("embeddings_identity.npy", "embeddings_evidence.npy"):
        e = np.load(os.path.join(ART, f))
        assert e.shape == (n, 384) and e.dtype == np.float32
        assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-3)


def test_alignment_candidate_ids():
    ids = np.load(os.path.join(ART, "candidate_ids.npy"))
    feats = np.load(os.path.join(ART, "features.npy"))
    assert len(ids) == feats.shape[0]
    rm = read_json(os.path.join(ART, "repr_manifest.json"))
    assert rm["n_rows"] == feats.shape[0]


def test_bm25_skills_excluded_and_phrases():
    vocab = json.load(open(os.path.join(ART, "bm25_index", "vocab.json"), encoding="utf-8"))
    # phrase-joined tokens exist
    assert any("_" in t for t in vocab)
    # the index loads and scores
    import scipy.sparse as sp
    W = sp.load_npz(os.path.join(ART, "bm25_index", "matrix.npz"))
    assert W.shape[0] == np.load(os.path.join(ART, "features.npy")).shape[0]


def test_snippets_row_addressable_and_stable_ids():
    from src.snippets import SnippetReader
    r = SnippetReader(ART)
    n = np.load(os.path.join(ART, "features.npy")).shape[0]
    assert len(r) == n
    # random access a few rows without a full load
    for i in (0, n // 2, n - 1):
        row = r.row(i)
        assert "candidate_id" in row and "snippets" in row
        for s in row["snippets"]:
            assert s["evidence_id"].startswith(row["candidate_id"])
            assert s["source_tier"] in ("verified", "demonstrated", "declared")
    r.close()


def test_normalization_stats_percentiles():
    ns = read_json(os.path.join(ART, "normalization_stats.json"))
    yoe = ns["yoe"]
    for k in ("count", "missing", "min", "max", "mean", "std", "p1", "p50", "p99"):
        assert k in yoe
