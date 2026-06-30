"""Goal 2 vector tests (require the downloaded model + generated jd vectors).
Skips cleanly if the artifacts haven't been built yet."""
import os

import numpy as np
import pytest

from src.common import read_json

pytestmark = pytest.mark.skipif(
    not (os.path.exists("jd/jd_ideal.npy") and os.path.exists("jd/jd_antiprofile.npy")),
    reason="JD vectors not built yet (run src.jd_build.build)")

JD = read_json("jd/jd_query.json")


def test_ideal_vector_shape_and_norm():
    ideal = np.load("jd/jd_ideal.npy")
    assert ideal.shape == (1, JD["model"]["dim"])
    assert ideal.dtype == np.float32
    assert abs(float(np.linalg.norm(ideal[0])) - 1.0) < 1e-4


def test_antiprofile_matrix_shape_norm_and_order():
    anti = np.load("jd/jd_antiprofile.npy")
    labels = JD["antiprofile_labels"]
    assert anti.shape == (len(labels), JD["model"]["dim"])
    assert np.allclose(np.linalg.norm(anti, axis=1), 1.0, atol=1e-4)

    # Row order == declared label order: re-embed each narrative and match its row.
    from src import embed
    narrs = [a["narrative"] for a in JD["antiprofiles"]]
    re = embed.embed_queries(narrs)
    for i in range(len(labels)):
        assert float(re[i] @ anti[i]) > 0.999, f"row {i} ({labels[i]}) order mismatch"


def test_jd_manifest_fields():
    m = read_json("jd/jd_manifest.json")
    for k in ("model", "jd_query_sha256", "jd_ideal_sha256", "jd_antiprofile_sha256",
              "validation_status", "n_criteria", "dim", "n_antiprofiles"):
        assert k in m and m[k] is not None
    assert m["validation_status"] == "passed"
    assert m["dim"] == JD["model"]["dim"]
