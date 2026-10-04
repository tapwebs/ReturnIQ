"""Honest model tests: trains, calibrates, round-trips; non-inferiority is reported, not tuned."""

import hashlib

import numpy as np
import pytest
from returniq_contracts import Stage

from returniq_engine.assess import get_view
from returniq_engine.ml import split_indices

BANNED = ("fraud", "fake customer")


@pytest.mark.parametrize("which", ["s2", "s1"])
def test_model_metrics_recorded_and_non_inferior(ds42, model_s1, model_s2, which):
    m = model_s2 if which == "s2" else model_s1
    met = m.info.metrics
    print(
        f"{m.info.stage.value}: ML AUC={met.auc:.4f} rules AUC={met.rules_auc:.4f} "
        f"P@12={met.precision_at_12:.3f} R@12={met.recall_at_12:.3f} "
        f"captured={met.preventable_captured_pct}"
    )
    assert met.auc is not None and met.rules_auc is not None
    assert met.auc >= met.rules_auc - 0.02  # non-inferiority, reported not tuned
    assert m.info.split == "TIME_BASED" and m.info.calibration is not None
    assert m.info.band_table and sum(b.pct_orders for b in m.info.band_table) == pytest.approx(
        100.0
    )


def test_time_split_boundaries(ds42):
    v = get_view(ds42, Stage.POST_DELIVERY)
    tr, va, te = split_indices(v.as_of, len(v.order_ids))
    ts = np.array([t.timestamp() for t in v.as_of])
    assert ts[tr].max() <= ts[va].min() and ts[va].max() <= ts[te].min()
    n = len(ts)
    assert len(tr) == int(n * 0.7) and len(te) == n - int(n * 0.8)


def test_artifact_round_trip_predictions_equal(eng, ds42, model_s2):
    loaded = eng.load_model(model_s2.artifact)
    x = get_view(ds42, Stage.POST_DELIVERY).feature_frame
    assert np.allclose(model_s2.predict_proba(x), loaded.predict_proba(x))
    assert loaded.info.id == model_s2.info.id
    assert isinstance(model_s2.artifact, bytes)


def test_retrain_is_deterministic(eng, ds42, model_s2):
    again = eng.train(ds42, Stage.POST_DELIVERY, None)
    assert (
        hashlib.sha256(again.artifact).hexdigest() == hashlib.sha256(model_s2.artifact).hexdigest()
    )


def test_calibrated_output_in_range_and_monotone_with_raw(ds42, model_s2):
    x = get_view(ds42, Stage.POST_DELIVERY).feature_frame
    p = model_s2.predict_proba(x)
    assert p.min() >= 0 and p.max() <= 1
    raw = model_s2.booster.predict(x[model_s2.feature_names])
    order = np.argsort(raw, kind="stable")
    assert np.all(np.diff(p[order]) >= -1e-9)


def test_model_blend_marks_scoring_method_and_signals(eng, ds42, settings, model_s2):
    batch = eng.assess_batch(ds42, Stage.POST_DELIVERY, settings, model_s2)
    assert all(a.scoring_method.value == "RULES_AND_MODEL" for a in batch)
    assert all(a.model_version == model_s2.info.id for a in batch)
    with_ml = [a for a in batch if any(s.source.value == "ML" for s in a.signals)]
    assert with_ml
    for a in batch:
        assert not any(
            b in (a.reason + " ".join(s.label for s in a.signals)).lower() for b in BANNED
        )


def test_stage_mismatch_model_is_ignored(eng, ds42, settings, model_s2):
    a = eng.assess(
        ds42,
        "ord_scn_01",
        Stage.PRE_DISPATCH,
        __import__("datetime").datetime(2027, 1, 1, tzinfo=__import__("datetime").UTC),
        settings,
        model_s2,
    )
    assert a.scoring_method.value == "RULES" and a.model_version is None


def test_train_without_labels_or_outcomes_fails_clearly(eng, ds42):
    from returniq_engine import Dataset

    bare = Dataset(ds42.master, ds42.provenance, labels=None)
    with pytest.raises(ValueError, match="no training labels"):
        eng.train(bare, Stage.POST_DELIVERY, None)
