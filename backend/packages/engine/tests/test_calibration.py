import numpy as np
import pytest
from returniq_contracts import RiskLevel, Stage, Thresholds, Weights

from returniq_engine.calibrate import band, band_table, blend, fit_isotonic, largest_remainder

W = {"rule": 0.4, "ml": 0.4, "anomaly": 0.2}


def test_rules_only_weight_is_one():
    score, used = blend({"rule": 63.0, "ml": None, "anomaly": None}, W)
    assert used == {"rule": 1.0} and score == 63


def test_rules_plus_ml_renormalised_half_half():
    score, used = blend({"rule": 40.0, "ml": 80.0}, W)
    assert used == pytest.approx({"rule": 0.5, "ml": 0.5}) and score == 60


def test_all_three_default_weights():
    score, used = blend({"rule": 50.0, "ml": 50.0, "anomaly": 100.0}, W)
    assert used == pytest.approx(W) and score == 60


def test_blend_formula_property():
    rng = np.random.default_rng(0)
    for _ in range(200):
        c = {k: float(rng.uniform(0, 100)) for k in W}
        w = {k: float(rng.uniform(0.1, 1)) for k in W}
        score, used = blend(c, w)
        assert sum(used.values()) == pytest.approx(1.0)
        assert score == int(round(sum(w[k] * c[k] for k in W) / sum(w.values())))
        assert 0 <= score <= 100


@pytest.mark.parametrize(
    ("score", "level"),
    [
        (39, "LOW"),
        (40, "MEDIUM"),
        (69, "MEDIUM"),
        (70, "HIGH"),
        (100, "HIGH"),
        (None, "INSUFFICIENT_DATA"),
    ],
)
def test_band_edges(score, level):
    assert band(score, Thresholds()) == RiskLevel(level)


def test_largest_remainder_sums_exactly():
    for counts in ([1, 1, 1], [333, 333, 334, 0], [7, 0, 0, 0], [5, 3, 1, 1]):
        assert sum(largest_remainder(counts)) == 1000


@pytest.mark.parametrize("stage", [Stage.PRE_DISPATCH, Stage.POST_DELIVERY])
def test_components_in_range_and_band_table_sums_to_100(
    eng, ds42, settings, model_s1, model_s2, stage
):
    model = model_s1 if stage == Stage.PRE_DISPATCH else model_s2
    for m in (None, model):
        batch = eng.assess_batch(ds42, stage, settings, m)
        for a in batch:
            for v in (a.components.rule, a.components.ml, a.components.anomaly, a.risk_score):
                assert v is None or 0 <= v <= 100
            if m is None:
                assert a.components.ml is None
                if a.risk_score is not None:
                    assert a.weights.rule == 1.0
            elif a.risk_score is not None:
                assert a.weights.rule == pytest.approx(0.5) and a.weights.ml == pytest.approx(0.5)
        scores = np.array([np.nan if a.risk_score is None else a.risk_score for a in batch])
        table = band_table(scores, np.zeros(len(scores), dtype=bool), settings.thresholds)
        assert sum(r.pct_orders for r in table) == pytest.approx(100.0)
        assert sum(r.orders for r in table) == len(batch)


def test_isotonic_is_monotone():
    rng = np.random.default_rng(3)
    x = rng.random(300)
    y = (rng.random(300) < x).astype(float)
    iso = fit_isotonic(x, y)
    grid = iso.predict(np.linspace(0, 1, 50))
    assert np.all(np.diff(grid) >= -1e-12) and grid.min() >= 0 and grid.max() <= 1


def test_weights_model_defaults():
    w = Weights()
    assert (w.rule, w.ml, w.anomaly) == (0.4, 0.4, 0.2)
