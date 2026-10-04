"""Metric definitions (0.3): RTO vs returns, PRR, preventable, TTR, loss."""

import numpy as np
import pandas as pd
import pytest
from returniq_contracts import CostDefaults, Filters, Stage, TtrBucket

from returniq_engine import Dataset
from returniq_engine import loss as loss_mod
from returniq_engine.analytics import flagged_mask, ttr_bucket, ttr_bucket_counts
from returniq_engine.prepare import derive_columns
from returniq_engine.rules import FAULT_CODE

from .conftest import FAR_FUTURE


@pytest.fixture(scope="module")
def scn(ds42):
    """The 18 fixed scenario orders only: small enough to verify by hand."""
    m = ds42.master[ds42.master["customer_id"].str.startswith("cus_scn")].reset_index(drop=True)
    return Dataset(m, ds42.provenance)


def test_rto_and_returns_are_separate_metrics(eng, scn, settings):
    a = eng.analytics(scn, Filters(), settings)
    assert (a.rto.count, a.rto.denominator) == (2, 18)
    assert a.rto.rate == pytest.approx(11.11, abs=0.01)
    assert (a.returns.requested, a.returns.accepted, a.returns.denominator) == (7, 6, 16)
    assert a.returns.rate == pytest.approx(37.5)
    fields = set(a.model_dump())
    assert not any("rto_and_return" in f or "combined" in f for f in fields)


def test_prr_formula_flagged_and_eligible_over_delivered(eng, scn, settings):
    a = eng.analytics(scn, Filters(), settings)
    assert a.flagged_returns == 2  # ord_scn_04_h4 and ord_scn_04
    assert a.estimated_prr == pytest.approx(2 / 16 * 100)
    assert a.confirmed_prr is None  # unknown stays unknown


def test_prr_on_full_dataset_matches_formula(eng, ds42, settings):
    a = eng.analytics(ds42, Filters(), settings)
    assert a.estimated_prr == pytest.approx(round(a.flagged_returns / a.delivered * 100, 2))
    assert a.rto.denominator != a.returns.denominator


@pytest.mark.parametrize("reason", ["DEFECTIVE", "DAMAGED_IN_TRANSIT", "WRONG_ITEM"])
def test_seller_fault_reasons_never_preventable(eng, ds42, settings, reason):
    i = ds42.order_pos["ord_scn_04"]
    m = ds42.master.copy()
    m.loc[i, "return_reason"] = reason
    ds = Dataset(derive_columns(m), ds42.provenance)
    a = eng.assess(ds, "ord_scn_04", Stage.POST_DELIVERY, FAR_FUTURE, settings, None)
    assert FAULT_CODE in {s.code for s in a.signals}
    assert a.risk_score is not None and a.risk_score <= 15 and a.risk_level.value == "LOW"
    assert not flagged_mask(ds, settings)[i]


def test_missing_delivered_at_gives_none_ttr(ds42):
    row = ds42.master.iloc[ds42.order_pos["ord_scn_06"]]
    assert pd.isna(row["ttr_h"])
    assert ttr_bucket(None) == TtrBucket.UNKNOWN
    counts = {b.bucket: b.count for b in ttr_bucket_counts(pd.Series([np.nan, 2.0]))}
    assert counts[TtrBucket.UNKNOWN] == 1 and counts[TtrBucket.LT_6H] == 1


def test_negative_ttr_is_not_a_real_ttr(ds42):
    m = ds42.master.copy()
    i = ds42.order_pos["ord_scn_04"]
    m.loc[i, "ret_ts"] = m.loc[i, "delivered_ts"] - pd.Timedelta(hours=1)
    assert pd.isna(derive_columns(m).loc[i, "ttr_h"])


@pytest.mark.parametrize(
    ("hours", "bucket"),
    [
        (5.99, "LT_6H"),
        (6, "H6_24"),
        (23.99, "H6_24"),
        (24, "D1_3"),
        (72, "D3_7"),
        (168, "D3_7"),
        (168.01, "GT_7D"),
    ],
)
def test_ttr_bucket_boundaries(hours, bucket):
    assert ttr_bucket(hours).value == bucket


def _one_return(amount_inr: float, qc: str) -> pd.DataFrame:
    base = pd.Timestamp("2026-03-01", tz="UTC")
    rows = pd.DataFrame(
        {
            "return_status": ["ACCEPTED"],
            "qc_result": [qc],
            "amount_paise": [int(amount_inr * 100)],
            "fwd_paise": [np.nan],
            "pack_paise": [np.nan],
            "rev_paise": [np.nan],
            "handling_paise": [np.nan],
            "writeoff_paise": [np.nan],
            "recovery_paise": [np.nan],
            "delivery_status": ["DELIVERED"],
            "ret_ts": [base],
        }
    )
    return rows


def test_loss_excludes_order_value_and_matches_hand_calculation():
    cost = CostDefaults()
    small = loss_mod.summarize(_one_return(1000, "UNUSED"), cost)
    big = loss_mod.summarize(_one_return(100000, "UNUSED"), cost)
    assert small.observed_paise == big.observed_paise == 6000 + 8000 + 1500 + 2500
    assert "writeoff" not in small.assumed_fields  # no write-off applied, nothing assumed
    used = loss_mod.summarize(_one_return(1000, "USED"), cost)
    assert used.observed_paise == 18000 + 10000  # + 10% write-off of Rs 1,000
    assert {"forward_ship", "reverse_ship", "packaging", "handling"} <= set(used.assumed_fields)


def test_provided_cost_fields_are_used_not_assumed():
    df = _one_return(1000, "UNUSED")
    for c, v in (
        ("fwd_paise", 5000),
        ("rev_paise", 7000),
        ("pack_paise", 1000),
        ("handling_paise", 2000),
        ("recovery_paise", 500),
    ):
        df[c] = float(v)
    s = loss_mod.summarize(df, CostDefaults())
    assert s.observed_paise == 5000 + 7000 + 1000 + 2000 - 500
    assert s.assumed_fields == []


def test_net_avoidable_can_be_negative_and_is_scenario(eng, ds42, settings):
    est = eng.estimate_loss(ds42, Filters(), settings)
    assert est.net_avoidable.low <= est.net_avoidable.base <= est.net_avoidable.high
    assert "not realised savings" in est.note
    costly = settings.model_copy(deep=True)
    costly.cost_defaults_paise.intervention_cost_paise = 10_000_000
    assert eng.estimate_loss(ds42, Filters(), costly).net_avoidable.base < 0


def test_timeline_cohorts_and_waterfall(eng, ds42, settings):
    t = eng.timeline(ds42, Filters(), settings)
    assert {c.cohort for c in t.cohort_compare} == {"FLAGGED", "NOT_FLAGGED"}
    assert (
        sum(b.count for b in t.ttr_buckets)
        == eng.analytics(ds42, Filters(), settings).returns.requested
    )
    total = sum(s.amount_paise for s in t.loss_waterfall[:-1])
    assert total == t.loss_waterfall[-1].amount_paise
