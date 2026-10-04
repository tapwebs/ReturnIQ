import numpy as np
import pandas as pd
import pytest

from returniq_engine.rules import POINTS_TABLE, SIGNAL_TEXT, compute_rules

BANNED = ("fraud", "fake customer")


def _frame(**over):
    base = dict(
        is_cod=0.0,
        amount_paise=100000.0,
        value_ratio=1.0,
        cat_code=0.0,
        prior_orders=0.0,
        prior_delivered=0.0,
        prior_rto=0.0,
        prior_failed=0.0,
        prior_returns=0.0,
        prior_elig_returns=0.0,
        prior_return_rate=np.nan,
        prior_rto_rate=np.nan,
        recent_returns_30d=0.0,
        prior_fast_share=np.nan,
        prior_flagged=0.0,
        cust_median_ttr=np.nan,
        pin_rto_lift=1.0,
        pin_ret_lift=1.0,
        sku_ret_lift=1.0,
        cat_ret_lift=1.0,
        ttr_h=np.nan,
        ttr_known=0.0,
        ttr_ratio=np.nan,
        ttr_z=np.nan,
        seller_median_ttr=48.0,
        is_fast_ttr=0.0,
        delivery_attempts=1.0,
        reason_eligible=1.0,
        same_cat_returns=0.0,
    )
    base.update(over)
    return pd.DataFrame({k: [v] for k, v in base.items()})


def test_cod_with_prior_rto_points():
    one = compute_rules(_frame(is_cod=1.0, prior_rto=1.0), 1, 7)
    two = compute_rules(_frame(is_cod=1.0, prior_rto=2.0), 1, 7)
    assert one["COD_WITH_PRIOR_RTO"][0] == POINTS_TABLE["COD_WITH_PRIOR_RTO"]["one_prior"]
    assert two["COD_WITH_PRIOR_RTO"][0] == POINTS_TABLE["COD_WITH_PRIOR_RTO"]["two_or_more_prior"]
    prepaid = compute_rules(_frame(is_cod=0.0, prior_rto=2.0), 1, 7)
    assert prepaid["COD_WITH_PRIOR_RTO"][0] == 0


def test_fast_return_is_relative_to_seller_median():
    f = dict(ttr_h=3.0, ttr_known=1.0)
    fast = compute_rules(_frame(ttr_ratio=3.0 / 48.0, **f), 2, 7)
    slow_seller = compute_rules(_frame(ttr_ratio=3.0 / 4.0, **f), 2, 7)  # seller median 4 h
    assert fast["TTR_ABNORMALLY_FAST"][0] == 20
    assert slow_seller["TTR_ABNORMALLY_FAST"][0] == 0


def test_policy_window_violation():
    late = compute_rules(_frame(ttr_h=200.0, ttr_known=1.0, ttr_ratio=4.0), 2, 7)
    ok = compute_rules(_frame(ttr_h=100.0, ttr_known=1.0, ttr_ratio=2.0), 2, 7)
    assert late["POLICY_WINDOW_VIOLATION"][0] == 15 and ok["POLICY_WINDOW_VIOLATION"][0] == 0


def test_total_capped_at_100():
    f = _frame(
        is_cod=1.0,
        prior_orders=5.0,
        prior_rto=4.0,
        prior_rto_rate=0.8,
        prior_failed=4.0,
        prior_elig_returns=4.0,
        prior_return_rate=0.8,
        recent_returns_30d=3.0,
        prior_fast_share=1.0,
        prior_flagged=3.0,
        pin_rto_lift=3.0,
        sku_ret_lift=3.0,
        cat_ret_lift=2.0,
        value_ratio=3.0,
    )
    assert compute_rules(f, 1, 7)["total"][0] == 100.0


@pytest.mark.parametrize("reason_eligible", [0.0])
def test_fault_reason_zeroes_behaviour_points(reason_eligible):
    f = _frame(
        prior_orders=5.0,
        prior_elig_returns=4.0,
        prior_return_rate=0.8,
        recent_returns_30d=3.0,
        prior_fast_share=1.0,
        ttr_h=2.0,
        ttr_known=1.0,
        ttr_ratio=0.05,
        reason_eligible=reason_eligible,
    )
    out = compute_rules(f, 2, 7)
    assert out["total"][0] == 0 and out["SELLER_OR_CARRIER_FAULT_REASON"][0] == 1


def test_points_table_sanity():
    assert all(isinstance(v, dict) for v in POINTS_TABLE.values())


def test_signal_text_has_no_banned_words():
    row = dict.fromkeys(_frame().columns, 1.0) | {"_stage": 1}
    for label, evidence in SIGNAL_TEXT.values():
        text = (label(row) + evidence(row)).lower()
        assert not any(b in text for b in BANNED)
