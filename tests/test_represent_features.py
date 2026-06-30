"""Goal 3 (structured, model-independent) feature tests + NaN/_present invariant."""
import datetime as dt
import math

import pytest

from src import represent
from tests import fixtures

REF = dt.date(2024, 6, 1)


@pytest.fixture(scope="module")
def fz():
    return represent.FrozenInputs.load(REF)


def _vals(rec, fz):
    return represent.derive_structured(rec, fz)


def test_nan_present_invariant_holds(fz):
    for name, rec in fixtures.all_fixtures().items():
        from src.ingest import parse_record
        row = _vals(parse_record(rec), fz)
        for nb in row.nullable:
            present = row.values[nb + "_present"]
            val = row.values[nb]
            if present == 0.0:
                assert math.isnan(val), f"{name}/{nb}: _present==0 but value not NaN"
            else:
                assert not math.isnan(val), f"{name}/{nb}: _present==1 but value NaN"


def test_honeypot_consistency_features(fz):
    from src.ingest import parse_record
    h1 = _vals(parse_record(fixtures.honeypot_h1()), fz).values
    assert h1["yoe_minus_summed_tenure"] > 2.0
    h2 = _vals(parse_record(fixtures.honeypot_h2()), fz).values
    assert h2["yoe_minus_summed_tenure"] < -2.0   # summed >> yoe
    h3 = _vals(parse_record(fixtures.honeypot_h3()), fz).values
    assert h3["expert_skill_zero_duration_count"] >= 2.0


def test_company_type_features(fz):
    from src.ingest import parse_record
    only = _vals(parse_record(fixtures.consulting_only()), fz).values
    assert only["is_consulting_entire_career"] == 1.0
    assert only["any_product_stint"] == 0.0
    assert only["frac_career_services"] == 1.0

    wp = _vals(parse_record(fixtures.consulting_with_product()), fz).values
    assert wp["is_consulting_entire_career"] == 0.0
    assert wp["any_product_stint"] == 1.0
    assert wp["product_tenure_yrs"] > 0.0


def test_tier5_has_product_and_ml_evidence(fz):
    from src.ingest import parse_record
    t = _vals(parse_record(fixtures.plain_language_tier5()), fz).values
    assert t["product_tenure_yrs"] > 0.0
    assert t["kw_ml_role_tenure_yrs"] > 0.0
    # Tier-1 corroboration via assessment present (Python + Information Retrieval)
    assert t["assess_strong_python_max_present"] == 1.0
    assert t["assess_ranking_eval_max_present"] == 1.0   # Information Retrieval


def test_stuffer_lacks_corroboration(fz):
    from src.ingest import parse_record
    s = _vals(parse_record(fixtures.canonical_stuffer()), fz).values
    # HR Manager: no assessments at all -> all assess_*_present == 0
    assert s["assess_retrieval_embeddings_max_present"] == 0.0
    assert s["assess_ranking_eval_max_present"] == 0.0
    # title is HR, not a ranking/retrieval role
    assert s["title_hit_ranking_eval"] == 0.0


def test_cv_speech_evidence_pattern(fz):
    from src.ingest import parse_record
    c = _vals(parse_record(fixtures.cv_speech_primary()), fz).values
    assert c["assess_cv_speech_max_present"] == 1.0
    assert c["assess_retrieval_embeddings_max_present"] == 0.0


def test_determinism(fz):
    from src.ingest import parse_record
    rec = parse_record(fixtures.plain_language_tier5())
    a = _vals(rec, fz).values
    b = _vals(rec, fz).values
    assert a == b
