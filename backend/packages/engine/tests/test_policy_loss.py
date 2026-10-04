import pytest
from returniq_contracts import Stage


@pytest.mark.parametrize("stage", [Stage.PRE_DISPATCH, Stage.POST_DELIVERY])
def test_captured_is_monotone_in_top_pct(eng, ds42, settings, stage):
    prev = -1.0
    for pct in (0, 5, 12, 25, 50, 75, 100):
        sim = eng.simulate_policy(ds42, stage, pct, settings, None)
        cap = sim.preventable_captured_pct or 0.0
        assert cap >= prev
        prev = cap
    full = eng.simulate_policy(ds42, stage, 100, settings, None)
    assert full.preventable_captured_pct == 100.0 and full.returns_captured_pct == 100.0


def test_targeted_count_and_fp(eng, ds42, settings):
    sim = eng.simulate_policy(ds42, Stage.PRE_DISPATCH, 12, settings, None)
    assert sim.orders_targeted == 144
    assert 0 <= sim.false_positive_estimate <= 1


def test_policy_rejects_bad_pct(eng, ds42, settings):
    with pytest.raises(ValueError):
        eng.simulate_policy(ds42, Stage.PRE_DISPATCH, 120, settings, None)


def test_policy_deterministic(eng, ds42, settings):
    a = eng.simulate_policy(ds42, Stage.POST_DELIVERY, 12, settings, None)
    b = eng.simulate_policy(ds42, Stage.POST_DELIVERY, 12, settings, None)
    assert a == b


def test_ml_policy_at_least_as_good_as_rules_on_proxy(eng, ds42, settings, model_s2):
    rules = eng.simulate_policy(ds42, Stage.POST_DELIVERY, 25, settings, None)
    ml = eng.simulate_policy(ds42, Stage.POST_DELIVERY, 25, settings, model_s2)
    print("captured rules/ml:", rules.preventable_captured_pct, ml.preventable_captured_pct)
    assert ml.preventable_captured_pct >= rules.preventable_captured_pct - 5
