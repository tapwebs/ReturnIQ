"""Vectorised point-in-time features (Return DNA).

Every event lives in a sorted table keyed by ``group_code * SPAN + seconds``. A query for
"events strictly before ``as_of``" is two ``searchsorted`` calls plus a cumulative-sum
difference, so there is no per-order filtering anywhere. ``side="left"`` makes ties
(events at exactly ``as_of``) excluded, i.e. strictly-before semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("2026-01-01", tz="UTC")
OFFSET = 10_000_000  # keeps (seconds + OFFSET) positive for windowed queries
SPAN = 1_000_000_000
FAST_TTR_H = 6.0
RECENT_DAYS = 30
RTO_KNOWN_AFTER_H = (
    7 * 24
)  # CSV has no RTO timestamp: outcome is treated as known 7 d after dispatch
SMOOTH_K_PINCODE = 5.0
SMOOTH_K_SKU = 5.0
SMOOTH_K_CATEGORY = 10.0
MIN_BASELINE_ORDERS = 50
MIN_CATEGORY_MEDIAN_ORDERS = 20
MIN_SELLER_TTR_POINTS = 10
DEFAULT_SELLER_MEDIAN_TTR_H = 48.0
CATEGORY_CODES = {"FASHION": 0, "BEAUTY": 1, "ELECTRONICS": 2, "LIFESTYLE": 3, "OTHER": 4}
UNKNOWN_CATEGORY_CODE = 5

STAGE1_FEATURES: tuple[str, ...] = (
    "is_cod",
    "amount_paise",
    "value_ratio",
    "cat_code",
    "prior_orders",
    "prior_delivered",
    "prior_rto",
    "prior_failed",
    "prior_returns",
    "prior_elig_returns",
    "prior_return_rate",
    "prior_rto_rate",
    "recent_returns_30d",
    "prior_fast_share",
    "prior_flagged",
    "cust_median_ttr",
    "pin_rto_lift",
    "pin_ret_lift",
    "sku_ret_lift",
    "cat_ret_lift",
)
STAGE2_ONLY_FEATURES: tuple[str, ...] = (
    "ttr_h",
    "ttr_known",
    "ttr_ratio",
    "ttr_z",
    "seller_median_ttr",
    "is_fast_ttr",
    "delivery_attempts",
    "reason_eligible",
    "same_cat_returns",
)
STAGE2_FEATURES: tuple[str, ...] = (
    tuple(f for f in STAGE1_FEATURES if f != "prior_flagged") + STAGE2_ONLY_FEATURES
)
FORBIDDEN_FEATURES: tuple[str, ...] = (
    "qc_result",
    "return_accepted_at",
    "ret_acc_ts",
    "refund_amount_inr",
    "refund_paise",
    "reverse_shipping_inr",
    "handling_inr",
    "writeoff_inr",
    "cost_recovery_inr",
    "return_status",
    "is_preventable_return",
    "is_return",
    "is_rto",
    "cohort",
)


def seconds(ts: pd.Series) -> np.ndarray:
    """UTC timestamps to float seconds since EPOCH (NaN for missing)."""
    return (ts - EPOCH).dt.total_seconds().to_numpy(dtype="float64")


class EventTable:
    """Sorted (group, time) events with cumulative weights for strictly-before queries."""

    def __init__(self, codes: np.ndarray, sec: np.ndarray, weights: dict[str, np.ndarray]):
        key = codes.astype("int64") * SPAN + np.rint(sec + OFFSET).astype("int64")
        order = np.argsort(key, kind="stable")
        self.key = key[order]
        self.cum = {
            k: np.concatenate([[0.0], np.cumsum(v[order].astype("float64"))])
            for k, v in weights.items()
        }

    def _bounds(self, q_codes: np.ndarray, q_sec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        base = q_codes.astype("int64") * SPAN
        qs = np.rint(np.nan_to_num(q_sec, nan=0.0) + OFFSET).astype("int64")
        lo = np.searchsorted(self.key, base, side="left")
        hi = np.searchsorted(self.key, base + qs, side="left")
        return lo, np.maximum(hi, lo)

    def total(self, name: str, q_codes: np.ndarray, q_sec: np.ndarray) -> np.ndarray:
        lo, hi = self._bounds(q_codes, q_sec)
        return self.cum[name][hi] - self.cum[name][lo]


class ExpandingMedian:
    """Per-group expanding median of event values, queried strictly before a time."""

    def __init__(self, codes: np.ndarray, sec: np.ndarray, values: np.ndarray):
        key = codes.astype("int64") * SPAN + np.rint(sec + OFFSET).astype("int64")
        order = np.argsort(key, kind="stable")
        self.key = key[order]
        sorted_codes = codes[order]
        self.med = (
            pd.Series(values[order].astype("float64"))
            .groupby(sorted_codes)
            .expanding()
            .median()
            .to_numpy()
        )

    def at(self, q_codes: np.ndarray, q_sec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        base = q_codes.astype("int64") * SPAN
        qs = np.rint(np.nan_to_num(q_sec, nan=0.0) + OFFSET).astype("int64")
        lo = np.searchsorted(self.key, base, side="left")
        hi = np.maximum(np.searchsorted(self.key, base + qs, side="left"), lo)
        cnt = hi - lo
        val = np.full(len(q_codes), np.nan)
        ok = cnt > 0
        if len(self.med):
            val[ok] = self.med[hi[ok] - 1]
        return val, cnt.astype("float64")


@dataclass
class Prepared:
    """Everything computed once by ``Dataset.prepare()``."""

    master: pd.DataFrame
    s1: pd.DataFrame
    s2: pd.DataFrame
    s2_pos: np.ndarray  # master row positions that have a return request (aligned to s2)
    s2_of_master: np.ndarray  # master position -> s2 row or -1
    rules1: pd.DataFrame
    rules2: pd.DataFrame
    flagged_s2: np.ndarray  # bool per s2 row (rules-only, default MEDIUM threshold)


def _codes(values: pd.Series) -> np.ndarray:
    return pd.factorize(values.astype("object"), use_na_sentinel=True)[0].astype("int64")


def _lift(
    key_n: np.ndarray,
    key_hits: np.ndarray,
    glob_n: np.ndarray,
    glob_hits: np.ndarray,
    k: float,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    rate = np.divide(glob_hits, glob_n, out=np.zeros_like(glob_hits), where=glob_n > 0)
    lift = (k + key_hits) / (k + np.maximum(key_n, 0.0) * rate)
    lift = np.where(glob_n < MIN_BASELINE_ORDERS, 1.0, lift)
    if valid is not None:
        lift = np.where(valid, lift, np.nan)
    return lift


class _Keys:
    """Group codes and raw event arrays derived once from the master frame."""

    def __init__(self, m: pd.DataFrame):
        self.n = len(m)
        self.cust = _codes(m["customer_id"])
        cat = m["category"].map(CATEGORY_CODES).fillna(UNKNOWN_CATEGORY_CODE).astype("int64")
        self.cat = cat.to_numpy()
        self.pin_valid = m["pincode"].notna().to_numpy()
        self.pin = _codes(m["pincode"])
        self.sku = _codes(m["sku_id"])
        self.cust_cat = self.cust * 8 + self.cat
        self.order_sec = seconds(m["order_ts"])
        self.deliv_sec = seconds(m["delivered_ts"])
        self.fail_sec = seconds(m["fail_known_ts"])
        self.ret_sec = seconds(m["ret_ts"])
        self.has_ret = ~np.isnan(self.ret_sec)
        self.is_rto = (m["delivery_status"] == "RTO").to_numpy()
        self.elig = m["reason_eligible"].to_numpy(dtype=bool) & self.has_ret
        self.ttr = m["ttr_h"].to_numpy(dtype="float64")
        self.fast = self.elig & (self.ttr < FAST_TTR_H)
        self.amount = m["amount_paise"].to_numpy(dtype="float64")
        self.zero = np.zeros(self.n, dtype="int64")


class _Tables:
    """All event tables for one feature pass (flag weights differ per stage)."""

    def __init__(self, k: _Keys, flagged: np.ndarray):
        ones = np.ones(k.n)
        has_deliv = ~np.isnan(k.deliv_sec)
        has_fail = ~np.isnan(k.fail_sec)
        r = k.has_ret
        self.order_c = EventTable(k.cust, k.order_sec, {"n": ones})
        self.deliv_c = EventTable(k.cust[has_deliv], k.deliv_sec[has_deliv], {"n": ones[has_deliv]})
        self.fail_c = EventTable(
            k.cust[has_fail],
            k.fail_sec[has_fail],
            {"n": ones[has_fail], "rto": k.is_rto[has_fail].astype(float)},
        )
        self.ret_c = EventTable(
            k.cust[r],
            k.ret_sec[r],
            {
                "n": ones[r],
                "elig": k.elig[r].astype(float),
                "fast": k.fast[r].astype(float),
                "flag": flagged[r].astype(float),
            },
        )
        self.ret_cc = EventTable(k.cust_cat[r], k.ret_sec[r], {"elig": k.elig[r].astype(float)})
        ttr_ok = r & k.elig & ~np.isnan(k.ttr)
        self.ttr_c = ExpandingMedian(k.cust[ttr_ok], k.ret_sec[ttr_ok], k.ttr[ttr_ok])
        seller_ok = r & ~np.isnan(k.ttr)
        logt = np.log(np.maximum(k.ttr[seller_ok], 0.1))
        self.ttr_seller = ExpandingMedian(k.zero[seller_ok], k.ret_sec[seller_ok], k.ttr[seller_ok])
        self.ttr_log = EventTable(
            k.zero[seller_ok],
            k.ret_sec[seller_ok],
            {"n": np.ones(len(logt)), "s1": logt, "s2": logt**2},
        )
        self.cat_amount = ExpandingMedian(k.cat, k.order_sec, k.amount)
        self.glob_order = EventTable(k.zero, k.order_sec, {"n": ones})
        self.glob_ret = EventTable(k.zero[r], k.ret_sec[r], {"elig": k.elig[r].astype(float)})
        self.glob_rto = EventTable(
            k.zero[has_fail], k.fail_sec[has_fail], {"rto": k.is_rto[has_fail].astype(float)}
        )
        self.keyed: dict[str, tuple[EventTable, EventTable, EventTable | None]] = {}
        for name, codes, valid in (
            ("pin", k.pin, k.pin_valid),
            ("sku", k.sku, None),
            ("cat", k.cat, None),
        ):
            ok = np.ones(k.n, bool) if valid is None else valid
            orders = EventTable(codes[ok], k.order_sec[ok], {"n": ones[ok]})
            rets_ok = ok & r
            rets = EventTable(
                codes[rets_ok], k.ret_sec[rets_ok], {"elig": k.elig[rets_ok].astype(float)}
            )
            rto_ok = ok & has_fail
            rtos = (
                EventTable(
                    codes[rto_ok], k.fail_sec[rto_ok], {"rto": k.is_rto[rto_ok].astype(float)}
                )
                if name == "pin"
                else None
            )
            self.keyed[name] = (orders, rets, rtos)


def _history(
    k: _Keys, t: _Tables, q_sec: np.ndarray, q_mask: np.ndarray, include_flag: bool
) -> dict[str, np.ndarray]:
    """Customer / key history strictly before ``q_sec`` for the rows selected by ``q_mask``."""
    idx = np.flatnonzero(q_mask)
    q = q_sec[idx]
    cust, cc = k.cust[idx], k.cust_cat[idx]
    own_order = (k.order_sec[idx] < q).astype(float)
    own_deliv = (k.deliv_sec[idx] < q).astype(float)  # NaN compares False
    own_fail = (k.fail_sec[idx] < q).astype(float)
    out: dict[str, np.ndarray] = {}
    out["prior_orders"] = t.order_c.total("n", cust, q) - own_order
    out["prior_delivered"] = t.deliv_c.total("n", cust, q) - own_deliv
    out["prior_failed"] = t.fail_c.total("n", cust, q) - own_fail
    out["prior_rto"] = t.fail_c.total("rto", cust, q) - own_fail * k.is_rto[idx]
    out["prior_returns"] = t.ret_c.total("n", cust, q)
    out["prior_elig_returns"] = t.ret_c.total("elig", cust, q)
    fast = t.ret_c.total("fast", cust, q)
    recent = out["prior_elig_returns"] - t.ret_c.total(
        "elig", cust, np.maximum(q - RECENT_DAYS * 86400, -OFFSET + 1)
    )
    out["recent_returns_30d"] = recent
    po = out["prior_orders"]
    out["prior_return_rate"] = np.divide(
        out["prior_elig_returns"], po, out=np.full(len(idx), np.nan), where=po > 0
    )
    out["prior_rto_rate"] = np.divide(
        out["prior_rto"], po, out=np.full(len(idx), np.nan), where=po > 0
    )
    pe = out["prior_elig_returns"]
    out["prior_fast_share"] = np.divide(fast, pe, out=np.full(len(idx), np.nan), where=pe > 0)
    out["prior_flagged"] = t.ret_c.total("flag", cust, q) if include_flag else np.zeros(len(idx))
    out["same_cat_returns"] = t.ret_cc.total("elig", cc, q)
    med, _ = t.ttr_c.at(cust, q)
    out["cust_median_ttr"] = med
    g_n = t.glob_order.total("n", k.zero[idx], q) - own_order
    g_ret = t.glob_ret.total("elig", k.zero[idx], q)
    g_rto = t.glob_rto.total("rto", k.zero[idx], q) - own_fail * k.is_rto[idx]
    for name, kk, col in (
        ("pin", SMOOTH_K_PINCODE, "pin_ret_lift"),
        ("sku", SMOOTH_K_SKU, "sku_ret_lift"),
        ("cat", SMOOTH_K_CATEGORY, "cat_ret_lift"),
    ):
        orders, rets, _ = t.keyed[name]
        codes = {"pin": k.pin, "sku": k.sku, "cat": k.cat}[name][idx]
        kn = orders.total("n", codes, q) - own_order
        kh = rets.total("elig", codes, q)
        valid = k.pin_valid[idx] if name == "pin" else None
        out[col] = _lift(kn, kh, g_n, g_ret, kk, valid)
    orders, _, rtos = t.keyed["pin"]
    assert rtos is not None
    kn = orders.total("n", k.pin[idx], q) - own_order
    kr = rtos.total("rto", k.pin[idx], q) - own_fail * k.is_rto[idx]
    out["pin_rto_lift"] = _lift(kn, kr, g_n, g_rto, SMOOTH_K_PINCODE, k.pin_valid[idx])
    # seller TTR baseline
    smed, scnt = t.ttr_seller.at(k.zero[idx], q)
    out["seller_median_ttr"] = np.where(
        scnt >= MIN_SELLER_TTR_POINTS, smed, DEFAULT_SELLER_MEDIAN_TTR_H
    )
    n_t = t.ttr_log.total("n", k.zero[idx], q)
    s1 = t.ttr_log.total("s1", k.zero[idx], q)
    s2 = t.ttr_log.total("s2", k.zero[idx], q)
    mean = np.divide(s1, n_t, out=np.zeros(len(idx)), where=n_t > 0)
    var = np.divide(s2, n_t, out=np.zeros(len(idx)), where=n_t > 0) - mean**2
    out["_ttr_log_mean"] = np.where(n_t >= MIN_SELLER_TTR_POINTS, mean, np.nan)
    out["_ttr_log_sd"] = np.where(
        n_t >= MIN_SELLER_TTR_POINTS, np.sqrt(np.maximum(var, 1e-6)), np.nan
    )
    return out


def _order_level(m: pd.DataFrame, k: _Keys, t: _Tables, idx: np.ndarray) -> dict[str, np.ndarray]:
    """Current-order attributes (allowed at both stages); medians use orders before order_ts."""
    med, cnt = t.cat_amount.at(k.cat[idx], k.order_sec[idx])
    ratio = np.where(
        cnt >= MIN_CATEGORY_MEDIAN_ORDERS, k.amount[idx] / np.where(med > 0, med, np.nan), 1.0
    )
    return {
        "is_cod": m["payment_mode"].isin(["COD", "PARTIAL_COD"]).to_numpy()[idx].astype(float),
        "amount_paise": k.amount[idx],
        "value_ratio": np.nan_to_num(ratio, nan=1.0),
        "cat_code": k.cat[idx].astype(float),
    }


def build_stage_frames(
    m: pd.DataFrame, rules_fn, flag_threshold: int, window_days: int
) -> Prepared:
    """Compute Stage 2 features/rules first (needed for flags), then Stage 1 features/rules."""
    k = _Keys(m)
    n = k.n
    # ---- Stage 2 (as_of = return_requested_at) ----
    t2 = _Tables(k, np.zeros(n, dtype=bool))
    s2_pos = np.flatnonzero(k.has_ret)
    h2 = _history(k, t2, k.ret_sec, k.has_ret, include_flag=False)
    f2 = _order_level(m, k, t2, s2_pos)
    ttr = k.ttr[s2_pos]
    f2.update({key: v for key, v in h2.items() if not key.startswith("_")})
    ttr_known = ~np.isnan(ttr)
    smed = f2["seller_median_ttr"]
    f2["ttr_h"] = ttr
    f2["ttr_known"] = ttr_known.astype(float)
    f2["ttr_ratio"] = np.where(ttr_known, ttr / smed, np.nan)
    sd, mu = h2["_ttr_log_sd"], h2["_ttr_log_mean"]
    f2["ttr_z"] = np.where(ttr_known, (np.log(np.maximum(ttr, 0.1)) - mu) / sd, np.nan)
    f2["is_fast_ttr"] = (ttr_known & (ttr < FAST_TTR_H)).astype(float)
    f2["delivery_attempts"] = m["delivery_attempts"].to_numpy(dtype="float64")[s2_pos]
    f2["reason_eligible"] = k.elig[s2_pos].astype(float)
    s2 = pd.DataFrame({c: f2[c] for c in STAGE2_FEATURES})
    rules2 = rules_fn(s2, 2, window_days)
    flagged_s2 = (rules2["total"].to_numpy() >= flag_threshold) & k.elig[s2_pos]
    # ---- Stage 1 (as_of = order_date) ----
    flag_w = np.zeros(n, dtype=bool)
    flag_w[s2_pos] = flagged_s2
    t1 = _Tables(k, flag_w)
    all_rows = np.arange(n)
    h1 = _history(k, t1, k.order_sec, np.ones(n, bool), include_flag=True)
    f1 = _order_level(m, k, t1, all_rows)
    f1.update({key: v for key, v in h1.items() if not key.startswith("_")})
    s1 = pd.DataFrame({c: f1[c] for c in STAGE1_FEATURES})
    rules1 = rules_fn(s1, 1, window_days)
    s2_of_master = np.full(n, -1, dtype="int64")
    s2_of_master[s2_pos] = np.arange(len(s2_pos))
    return Prepared(m, s1, s2, s2_pos, s2_of_master, rules1, rules2, flagged_s2)
