"""Vectorised features equal a slow brute-force reference on sampled rows, at both stages."""

import numpy as np
import pandas as pd
import pytest
from returniq_contracts import Stage

from returniq_engine.features import (
    MIN_BASELINE_ORDERS,
    SMOOTH_K_PINCODE,
    _lift,
)


def _reference(m: pd.DataFrame, i: int, T: pd.Timestamp) -> dict:
    o = m.drop(index=i)
    same = o[o["customer_id"] == m.loc[i, "customer_id"]]
    elig = same[same["reason_eligible"] & (same["ret_ts"] < T)]
    pin = m.loc[i, "pincode"]
    out = {
        "prior_orders": int((same["order_ts"] < T).sum()),
        "prior_delivered": int((same["delivered_ts"] < T).sum()),
        "prior_failed": int((same["fail_known_ts"] < T).sum()),
        "prior_rto": int(((same["fail_known_ts"] < T) & (same["delivery_status"] == "RTO")).sum()),
        "prior_returns": int((same["ret_ts"] < T).sum()),
        "prior_elig_returns": len(elig),
        "recent_returns_30d": int((elig["ret_ts"] >= T - pd.Timedelta(days=30)).sum()),
        "same_cat_returns": int((elig["category"] == m.loc[i, "category"]).sum()),
    }
    ttr = elig["ttr_h"].dropna()
    out["cust_median_ttr"] = float(ttr.median()) if len(ttr) else np.nan
    g_n = float((o["order_ts"] < T).sum())
    g_h = float((o["reason_eligible"] & (o["ret_ts"] < T)).sum())
    if isinstance(pin, str):
        kp = o[o["pincode"] == pin]
        kn = float((kp["order_ts"] < T).sum())
        kh = float((kp["reason_eligible"] & (kp["ret_ts"] < T)).sum())
        rate = g_h / g_n if g_n else 0.0
        lift = (SMOOTH_K_PINCODE + kh) / (SMOOTH_K_PINCODE + kn * rate)
        out["pin_ret_lift"] = 1.0 if g_n < MIN_BASELINE_ORDERS else lift
    return out


@pytest.mark.parametrize("stage", [Stage.PRE_DISPATCH, Stage.POST_DELIVERY])
def test_vectorised_equals_bruteforce(ds42, stage):
    p = ds42.prepare()
    m = ds42.master
    frame = p.s1 if stage == Stage.PRE_DISPATCH else p.s2
    rng = np.random.default_rng(11)
    rows = rng.choice(len(frame), size=60, replace=False)
    for r in rows:
        i = int(r) if stage == Stage.PRE_DISPATCH else int(p.s2_pos[r])
        T = m.loc[i, "order_ts"] if stage == Stage.PRE_DISPATCH else m.loc[i, "ret_ts"]
        ref = _reference(m, i, T)
        for k, v in ref.items():
            if k not in frame.columns:
                continue
            got = frame.iloc[r][k]
            if isinstance(v, float) and np.isnan(v):
                assert np.isnan(got), (k, i)
            else:
                assert got == pytest.approx(v), (k, m.loc[i, "order_id"], got, v)


def test_lift_smoothing_small_pincodes_stay_near_one():
    tiny = _lift(np.array([2.0]), np.array([2.0]), np.array([1000.0]), np.array([100.0]), 5.0)
    big = _lift(np.array([400.0]), np.array([120.0]), np.array([1000.0]), np.array([100.0]), 5.0)
    assert 1.0 < tiny[0] < 1.7  # 2/2 returns on a tiny pincode does not spike
    assert big[0] == pytest.approx((5 + 120) / (5 + 40))  # large pincode approaches true ratio


def test_lift_neutral_without_baseline():
    v = _lift(np.array([3.0]), np.array([3.0]), np.array([10.0]), np.array([5.0]), 5.0)
    assert v[0] == 1.0


def test_missing_pincode_lift_is_nan_not_one(ds42):
    p = ds42.prepare()
    missing = ds42.master["pincode"].isna().to_numpy()
    assert missing.any()
    assert p.s1.loc[missing, "pin_rto_lift"].isna().all()


def test_prepare_is_cached(ds42):
    assert ds42.prepare() is ds42.prepare()


def test_no_row_wise_apply_in_hot_path():
    import inspect

    from returniq_engine import features

    src = inspect.getsource(features)
    assert ".apply(" not in src and "iterrows" not in src and "itertuples" not in src
