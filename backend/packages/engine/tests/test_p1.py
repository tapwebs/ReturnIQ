"""P1: IsolationForest anomaly, outcomes as extra labels, multi-seed evaluation."""

import numpy as np
import pytest
from returniq_contracts import Outcome, OutcomeResult, Stage

from returniq_engine import Dataset, report
from returniq_engine.assess import get_view
from returniq_engine.ml import _labels

from .conftest import FAR_FUTURE


def test_anomaly_component_percentile_and_weights(eng, ds42, settings):
    s = settings.model_copy(update={"anomaly_enabled": True})
    batch = eng.assess_batch(ds42, Stage.PRE_DISPATCH, s, None)
    for a in batch:
        assert a.components.anomaly is not None and 0 <= a.components.anomaly <= 100
        assert a.weights.rule == pytest.approx(2 / 3, abs=1e-3)
        assert a.weights.anomaly == pytest.approx(1 / 3, abs=1e-3)
        assert 0 <= a.risk_score <= 100
    again = eng.assess_batch(ds42, Stage.PRE_DISPATCH, s, None)
    assert [a.components.anomaly for a in batch] == [a.components.anomaly for a in again]


def test_anomaly_off_by_default(eng, ds42, settings):
    a = eng.assess(ds42, "ord_scn_01", Stage.PRE_DISPATCH, FAR_FUTURE, settings, None)
    assert a.components.anomaly is None and a.weights.rule == 1.0


def test_all_three_components_use_default_weights(eng, ds42, settings, model_s2):
    s = settings.model_copy(update={"anomaly_enabled": True})
    a = eng.assess(ds42, "ord_scn_04", Stage.POST_DELIVERY, FAR_FUTURE, s, model_s2)
    assert (a.weights.rule, a.weights.ml, a.weights.anomaly) == pytest.approx((0.4, 0.4, 0.2))


def test_outcomes_become_labels_and_can_train_without_hidden_labels(eng, ds42):
    bare = Dataset(ds42.master, ds42.provenance, labels=None)
    v = get_view(bare, Stage.POST_DELIVERY)
    lab = ds42.labels.set_index("order_id")["is_preventable_return"]
    outs = [
        Outcome(
            action_id=f"act_{i}",
            order_id=o,
            outcome=OutcomeResult.SUCCESS if lab[o] else OutcomeResult.OVERRIDDEN,
        )
        for i, o in enumerate(v.order_ids)
    ]
    y = _labels(bare, Stage.POST_DELIVERY, outs)
    assert not np.isnan(y).any() and y.sum() == lab.loc[v.order_ids].sum()
    model = eng.train(bare, Stage.POST_DELIVERY, outs)
    assert model.info.metrics.auc is not None


def test_outcome_overrides_hidden_label(ds42):
    v = get_view(ds42, Stage.POST_DELIVERY)
    oid = next(
        o for o in v.order_ids if ds42.labels.set_index("order_id").loc[o, "is_preventable_return"]
    )
    out = [Outcome(action_id="act_1", order_id=oid, outcome=OutcomeResult.OVERRIDDEN)]
    assert _labels(ds42, Stage.POST_DELIVERY, out)[v.index_of[oid]] == 0.0


def test_multi_seed_evaluation_runs():
    rows = report.evaluate_seeds(seeds=(41, 42), n=2500)
    assert len(rows) == 4 and all(r["ml_auc"] is not None for r in rows)
