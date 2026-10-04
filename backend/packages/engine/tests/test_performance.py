"""Measured performance (printed with -s): assess_batch on 25,000 orders < 5 s."""

import time
from datetime import UTC, datetime

import numpy as np
import pytest
from returniq_contracts import Stage

from returniq_engine import Dataset, ReturnIQEngine, default_settings, generate_dataset

pytestmark = pytest.mark.perf


@pytest.fixture(scope="module")
def big():
    t = time.perf_counter()
    ds = Dataset.from_generated(generate_dataset(25000, 7))
    t_build = time.perf_counter() - t
    t = time.perf_counter()
    ds.prepare()
    return ds, t_build, time.perf_counter() - t


def test_assess_batch_25k_under_5s(big):
    ds, t_build, t_prep = big
    eng, s = ReturnIQEngine(), default_settings()
    t = time.perf_counter()
    out = eng.assess_batch(ds, Stage.PRE_DISPATCH, s, None)
    elapsed = time.perf_counter() - t
    print(f"\nPERF build={t_build:.2f}s prepare={t_prep:.2f}s assess_batch(25000)={elapsed:.2f}s")
    assert len(out) == 25000
    assert elapsed < 5.0


def test_assess_p50_p95_over_1000_calls(big):
    ds, _, _ = big
    eng, s = ReturnIQEngine(), default_settings()
    ids = ds.master["order_id"].to_numpy()
    pick = np.random.default_rng(0).choice(ids, size=1000)
    now = datetime(2027, 1, 1, tzinfo=UTC)
    eng.assess(ds, str(pick[0]), Stage.PRE_DISPATCH, now, s, None)  # warm the stage view
    lat = []
    for oid in pick:
        t = time.perf_counter()
        eng.assess(ds, str(oid), Stage.PRE_DISPATCH, now, s, None)
        lat.append((time.perf_counter() - t) * 1000)
    p50, p95 = np.percentile(lat, [50, 95])
    print(f"\nPERF assess() p50={p50:.3f} ms p95={p95:.3f} ms over 1000 calls")
    assert p95 < 50
