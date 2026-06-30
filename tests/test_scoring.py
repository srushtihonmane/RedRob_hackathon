"""Goal 4 scoring tests + trap suite, run against the sample-scored output (real data).
Skips cleanly if the sample artifacts are not built."""
import os

import numpy as np
import pytest

ART = "data/sample/artifacts"
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ART, "features.npy")),
                                reason="sample artifacts not built")

from src import scoring   # noqa: E402


@pytest.fixture(scope="module")
def scored():
    b = scoring.load_bundle(ART)
    res = scoring.score_pool(b)
    id2row = {int(b.candidate_ids[i]): i for i in range(len(b.candidate_ids))}
    final = res["fit_score"] * res["engagement_modifier"] * res["risk_modifier"]
    order = np.argsort(-final)
    rank = {int(b.candidate_ids[order[r]]): r + 1 for r in range(len(order))}
    return {"b": b, "res": res, "final": final, "id2row": id2row, "rank": rank}


def _row(s, cid):  # cid int e.g. 9000002
    return s["id2row"][cid]


def test_retrieval_recall_includes_planted(scored):
    b = scored["b"]
    sl = set(scoring.retrieve_shortlist(b).tolist())
    for cid in (9000002, 9000009, 9000003):   # tier5, active_strong, honeypot
        assert _row(scored, cid) in sl


def test_calibration_irrelevant_bulk_near_zero(scored):
    fit = scored["res"]["fit_score"]
    assert np.median(fit) < 0.15      # NOT ~0.5 (tail-anchor property)
    assert fit.max() > 0.7


def test_fusion_high_dense_alone_cannot_inflate(scored):
    # stuffer: dense/lexical/assessment all weak -> fit near floor despite "active"
    fit = scored["res"]["fit_score"]
    assert fit[_row(scored, 9000001)] < 0.2


def test_modifiers_bounded(scored):
    eng = scored["res"]["engagement_modifier"]; rm = scored["res"]["risk_modifier"]
    assert eng.min() >= 0.7 - 1e-6 and eng.max() <= 1.1 + 1e-6
    assert rm.min() >= 0.5 - 1e-6 and rm.max() <= 1.0 + 1e-6


def test_fit_bundle_fields(scored):
    res = scored["res"]
    for k in ("fit_score", "fit_tier", "engagement_modifier", "risk_modifier",
              "corroboration_breadth", "crit_sats"):
        assert k in res


# ---- trap suite -----------------------------------------------------------
def test_FT2_tier5_ranked_up(scored):
    assert scored["rank"][9000002] <= 100        # plain-language Tier-5 in top-100
    assert scored["res"]["fit_tier"][_row(scored, 9000002)] in ("Elite", "Strong")


def test_FT1_stuffer_ranked_down(scored):
    assert scored["rank"][9000001] > 100         # keyword-stuffer out of top-100
    # low-fit x high-engagement stays low (the HR-Manager trap)
    final = scored["final"]
    assert final[_row(scored, 9000001)] < 0.2


def test_FT4_active_beats_inactive(scored):
    final = scored["final"]
    a, i = _row(scored, 9000009), _row(scored, 9000010)
    assert scored["res"]["fit_score"][a] == scored["res"]["fit_score"][i]   # identical fit
    assert final[a] > final[i]                                              # engagement separates


def test_cv_speech_penalized(scored):
    rm = scored["res"]["risk_modifier"]
    assert rm[_row(scored, 9000008)] < 0.8       # archetype resemblance penalized
