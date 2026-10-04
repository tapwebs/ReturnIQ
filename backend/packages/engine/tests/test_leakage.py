"""Leakage guard: stage feature tables and point-in-time mutation tests."""

import numpy as np
import pandas as pd
import pytest
from returniq_contracts import Stage

from returniq_engine import Dataset
from returniq_engine.features import (
    FORBIDDEN_FEATURES,
    STAGE1_FEATURES,
    STAGE2_FEATURES,
    STAGE2_ONLY_FEATURES,
)
from returniq_engine.prepare import derive_columns

from .conftest import FAR_FUTURE


def test_stage1_table_has_no_stage2_or_forbidden_columns(ds42):
    p = ds42.prepare()
    assert tuple(p.s1.columns) == STAGE1_FEATURES
    assert not set(p.s1.columns) & set(STAGE2_ONLY_FEATURES)
    assert not set(p.s1.columns) & set(FORBIDDEN_FEATURES)


def test_stage2_table_has_no_qc_or_refund_columns(ds42):
    p = ds42.prepare()
    assert tuple(p.s2.columns) == STAGE2_FEATURES
    assert not set(p.s2.columns) & set(FORBIDDEN_FEATURES)
    assert not any("qc" in c or "refund" in c or "accepted" in c for c in p.s2.columns)


def _assess(ds, oid, stage, eng, settings):
    a = eng.assess(ds, oid, stage, FAR_FUTURE, settings, None)
    return a.model_dump(exclude={"latency_ms"})


def _mutate_future(master: pd.DataFrame, target: int, as_of: pd.Timestamp, rng) -> pd.DataFrame:
    """Scramble every event after ``as_of`` (other orders and the target's own later events)."""
    m = master.copy()
    others = np.arange(len(m)) != target
    future_order = others & (m["order_ts"] > as_of).to_numpy()
    m.loc[future_order, "payment_mode"] = "COD"
    m.loc[future_order, "amount_paise"] = m.loc[future_order, "amount_paise"] * 7
    m.loc[future_order, "delivery_status"] = "RTO"
    m.loc[future_order, "return_reason"] = "CHANGED_MIND"
    m.loc[future_order, "ret_ts"] = m.loc[future_order, "order_ts"] + pd.Timedelta(hours=2)
    m.loc[future_order, "delivered_ts"] = pd.NaT
    late_ret = others & (m["ret_ts"] > as_of).to_numpy() & ~future_order
    m.loc[late_ret, "return_reason"] = "DEFECTIVE"
    m.loc[late_ret, "qc_result"] = "USED"
    m.loc[late_ret, "ret_ts"] = m.loc[late_ret, "ret_ts"] + pd.Timedelta(days=30)
    late_del = others & (m["delivered_ts"] > as_of).to_numpy() & ~future_order
    m.loc[late_del, "delivered_ts"] = m.loc[late_del, "delivered_ts"] + pd.Timedelta(days=20)
    late_fail = others & (m["fail_known_ts"] > as_of).to_numpy() & ~future_order
    m.loc[late_fail, "delivery_status"] = "DELIVERED"
    for col, val in (
        ("qc_result", "DAMAGED_BY_CUSTOMER"),
        ("refund_paise", 999.0),
        ("ret_acc_ts", pd.Timestamp("2030-01-01", tz="UTC")),
        ("return_status", "REJECTED"),
        ("rev_paise", 1.0),
        ("handling_paise", 1.0),
        ("writeoff_paise", 1.0),
        ("recovery_paise", 1.0),
    ):
        m.loc[m.index[target], col] = val
    return derive_columns(m)


def _sample_rows(ds, stage, k=25):
    p = ds.prepare()
    rng = np.random.default_rng(5)
    pos = np.arange(len(ds.master)) if stage == Stage.PRE_DISPATCH else p.s2_pos
    pick = list(rng.choice(pos, size=k, replace=False))
    scn = [ds.order_pos[o] for o in ("ord_scn_04", "ord_scn_07", "ord_scn_02")]
    return [int(i) for i in pick + [i for i in scn if i in set(pos)]]


@pytest.mark.parametrize("stage", [Stage.PRE_DISPATCH, Stage.POST_DELIVERY])
def test_mutating_events_after_as_of_leaves_assessment_unchanged(ds42, eng, settings, stage):
    rng = np.random.default_rng(1)
    for i in _sample_rows(ds42, stage):
        row = ds42.master.iloc[i]
        as_of = row["order_ts"] if stage == Stage.PRE_DISPATCH else row["ret_ts"]
        base = _assess(ds42, row["order_id"], stage, eng, settings)
        mutated = Dataset(_mutate_future(ds42.master, i, as_of, rng), ds42.provenance)
        after = _assess(mutated, row["order_id"], stage, eng, settings)
        assert base == after, f"{row['order_id']} changed after future-event mutation"


def test_stage1_ignores_the_orders_own_later_events(ds42, eng, settings):
    i = ds42.order_pos["ord_scn_04"]
    row = ds42.master.iloc[i]
    m = ds42.master.copy()
    m.loc[i, ["return_reason", "qc_result"]] = ["DEFECTIVE", "USED"]
    m.loc[i, "ret_ts"] = row["ret_ts"] + pd.Timedelta(days=9)
    m = derive_columns(m)
    a = _assess(ds42, "ord_scn_04", Stage.PRE_DISPATCH, eng, settings)
    b = _assess(Dataset(m, ds42.provenance), "ord_scn_04", Stage.PRE_DISPATCH, eng, settings)
    assert a == b


def test_tied_timestamps_are_excluded_strictly_before(ds42):
    """An event exactly at as_of is not history."""
    from returniq_engine.features import EventTable

    t = EventTable(np.array([0, 0, 0]), np.array([10.0, 20.0, 20.0]), {"n": np.ones(3)})
    assert t.total("n", np.array([0]), np.array([20.0]))[0] == 1
    assert t.total("n", np.array([0]), np.array([21.0]))[0] == 3
    assert t.total("n", np.array([0]), np.array([10.0]))[0] == 0
