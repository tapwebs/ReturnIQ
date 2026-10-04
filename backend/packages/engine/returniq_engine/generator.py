"""Deterministic synthetic data generator (GENERATOR SPEC v1, frozen once P0 tests pass).

All randomness comes from seeded numpy Generators spawned per concern. Times are generated as
IST wall-clock and serialised with a ``+05:30`` offset. Nothing reads the wall clock.
Hidden ``labels`` are for evaluation only and are never written to CSV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GENERATOR_SPEC_VERSION = "gen-1.0.0"
ANCHOR = pd.Timestamp("2026-01-01 00:00:00")
SPAN_DAYS = 180
SNAPSHOT = ANCHOR + pd.Timedelta(days=186)
TZ_SUFFIX = "+05:30"
CATEGORIES = ("FASHION", "BEAUTY", "ELECTRONICS", "LIFESTYLE")
CAT_P = np.array([0.40, 0.20, 0.20, 0.20])
CAT_MEDIAN_INR = {"FASHION": 900.0, "BEAUTY": 500.0, "ELECTRONICS": 2500.0, "LIFESTYLE": 700.0}
P_RETURN_BASE = {"FASHION": 0.14, "BEAUTY": 0.05, "ELECTRONICS": 0.06, "LIFESTYLE": 0.05}
HIGH_RETURN_SKUS = ("sku_FAS_003", "sku_FAS_011", "sku_BEA_005", "sku_LIF_008")
COHORTS = ("GENUINE", "WARDROBER", "RISKY_PIN", "LOW_INTENT_COD")
COHORT_P = np.array([0.74, 0.06, 0.10, 0.10])
P_COD = {"GENUINE": 0.50, "WARDROBER": 0.40, "RISKY_PIN": 0.80, "LOW_INTENT_COD": 0.90}
P_RTO_COD = {"GENUINE": 0.07, "WARDROBER": 0.05, "RISKY_PIN": 0.32, "LOW_INTENT_COD": 0.45}
P_RTO_PREPAID = {"GENUINE": 0.015, "WARDROBER": 0.015, "RISKY_PIN": 0.05, "LOW_INTENT_COD": 0.08}
NORMAL_PIN_INDEX = 20


def _pincode_pool() -> list[str]:
    rng = np.random.default_rng(7)  # fixed, seed-independent part of the spec
    pins = rng.choice(np.arange(110001, 800000), size=120, replace=False)
    return [str(p) for p in pins]


PINCODES = _pincode_pool()
RISKY_PINCODES = tuple(PINCODES[:8])
NORMAL_PINCODES = tuple(PINCODES[8:])


@dataclass
class GeneratedDataset:
    """CSV-form frames (strings for times, rupees as floats) plus hidden labels."""

    orders: pd.DataFrame
    shipments: pd.DataFrame
    returns: pd.DataFrame
    labels: pd.DataFrame
    seed: int = 0
    n_orders: int = 0


def _iso(ts: pd.Series) -> pd.Series:
    """IST wall-clock Timestamps to ISO strings with offset; NaT stays missing."""
    return ts.dt.strftime("%Y-%m-%dT%H:%M:%S") + TZ_SUFFIX


def _hours(h: np.ndarray | float) -> pd.Timedelta | pd.TimedeltaIndex:
    return pd.to_timedelta(h, unit="h")


def _scenario_rows() -> tuple[list[dict], list[dict], list[dict]]:
    """Fixed, seed-independent scenario orders (A.6). Returns (orders, shipments, returns)."""
    orders: list[dict] = []
    ships: list[dict] = []
    rets: list[dict] = []
    normal = NORMAL_PINCODES[NORMAL_PIN_INDEX]

    def order(
        oid: str,
        cus: str,
        day: int,
        cat: str,
        pay: str,
        amt: float,
        pin: str,
        status: str = "DELIVERED",
        ttr_h: float | None = None,
        reason: str | None = None,
        delivered_missing: bool = False,
        qc: str | None = None,
        attempts: int = 1,
    ) -> None:
        odt = ANCHOR + pd.Timedelta(days=day, hours=10)
        shipped = odt + pd.Timedelta(hours=6)
        delivered = shipped + pd.Timedelta(hours=96) if status == "DELIVERED" else None
        orders.append(
            {
                "order_id": oid,
                "customer_id": cus,
                "order_ts": odt,
                "sku_id": f"sku_scn_{cat[:3]}",
                "category": cat,
                "quantity": 1,
                "payment_mode": pay,
                "order_amount_inr": amt,
                "pincode": pin,
            }
        )
        ships.append(
            {
                "order_id": oid,
                "shipped_ts": shipped,
                "delivered_ts": None if delivered_missing else delivered,
                "delivery_status": status,
                "delivery_attempts": attempts,
            }
        )
        if ttr_h is not None and delivered is not None and reason is not None:
            req = delivered + pd.Timedelta(hours=ttr_h)
            rets.append(
                {
                    "order_id": oid,
                    "req_ts": req,
                    "acc_ts": req + pd.Timedelta(hours=12),
                    "return_reason": reason,
                    "return_status": "ACCEPTED",
                    "qc_result": qc,
                    "refund_amount_inr": None,
                }
            )

    # 1 normal prepaid order
    order("ord_scn_01_h1", "cus_scn_01", 100, "BEAUTY", "PREPAID", 599.0, normal)
    order("ord_scn_01_h2", "cus_scn_01", 115, "BEAUTY", "PREPAID", 549.0, normal)
    order("ord_scn_01", "cus_scn_01", 130, "BEAUTY", "PREPAID", 649.0, normal)
    # 2 COD order from a customer with two prior RTOs
    order(
        "ord_scn_02_h1",
        "cus_scn_02",
        90,
        "FASHION",
        "COD",
        1299.0,
        RISKY_PINCODES[0],
        status="RTO",
        attempts=3,
    )
    order(
        "ord_scn_02_h2",
        "cus_scn_02",
        105,
        "FASHION",
        "COD",
        1399.0,
        RISKY_PINCODES[0],
        status="RTO",
        attempts=3,
    )
    order("ord_scn_02", "cus_scn_02", 125, "FASHION", "COD", 1499.0, RISKY_PINCODES[0])
    # 3 legitimate fast defect return
    for i, day in enumerate((80, 95, 110), start=1):
        order(f"ord_scn_03_h{i}", "cus_scn_03", day, "LIFESTYLE", "PREPAID", 799.0, normal)
    order(
        "ord_scn_03",
        "cus_scn_03",
        120,
        "LIFESTYLE",
        "PREPAID",
        899.0,
        normal,
        ttr_h=3.0,
        reason="DEFECTIVE",
        qc="UNKNOWN",
    )
    # 4 repeat fast returner (4 of 5 returned, median 3.5 h, same category)
    order("ord_scn_04_h1", "cus_scn_04", 60, "FASHION", "COD", 1199.0, normal)
    order(
        "ord_scn_04_h2",
        "cus_scn_04",
        70,
        "FASHION",
        "COD",
        1199.0,
        normal,
        ttr_h=3.0,
        reason="CHANGED_MIND",
        qc="USED",
    )
    order(
        "ord_scn_04_h3",
        "cus_scn_04",
        80,
        "FASHION",
        "COD",
        1199.0,
        normal,
        ttr_h=3.5,
        reason="SIZE_FIT",
        qc="USED",
    )
    order(
        "ord_scn_04_h4",
        "cus_scn_04",
        90,
        "FASHION",
        "COD",
        1199.0,
        normal,
        ttr_h=3.5,
        reason="CHANGED_MIND",
        qc="USED",
    )
    order(
        "ord_scn_04",
        "cus_scn_04",
        100,
        "FASHION",
        "COD",
        1199.0,
        normal,
        ttr_h=4.5,
        reason="CHANGED_MIND",
        qc="USED",
    )
    # 5 late return at 6 days, first-time customer
    order(
        "ord_scn_05",
        "cus_scn_05",
        110,
        "FASHION",
        "COD",
        2499.0,
        normal,
        ttr_h=144.0,
        reason="CHANGED_MIND",
        qc="UNUSED",
    )
    # 6 return with a missing delivery timestamp
    odt = ANCHOR + pd.Timedelta(days=112, hours=10)
    order(
        "ord_scn_06",
        "cus_scn_06",
        112,
        "ELECTRONICS",
        "PREPAID",
        1499.0,
        normal,
        delivered_missing=True,
    )
    rets.append(
        {
            "order_id": "ord_scn_06",
            "req_ts": odt + pd.Timedelta(hours=6 + 96 + 30),
            "acc_ts": None,
            "return_reason": "NOT_AS_DESCRIBED",
            "return_status": "REQUESTED",
            "qc_result": None,
            "refund_amount_inr": None,
        }
    )
    # 7 the scenario-4 customer places a new COD order after the earlier flagged returns
    order("ord_scn_07", "cus_scn_04", 115, "FASHION", "COD", 1299.0, normal)
    return orders, ships, rets


def _draw_reasons(
    rng: np.random.Generator, behavioural: np.ndarray, category: np.ndarray
) -> np.ndarray:
    n = len(behavioural)
    names = np.array(
        [
            "SIZE_FIT",
            "CHANGED_MIND",
            "NOT_AS_DESCRIBED",
            "DEFECTIVE",
            "DAMAGED_IN_TRANSIT",
            "WRONG_ITEM",
            "OTHER",
        ]
    )
    u = rng.random(n)
    out = np.empty(n, dtype=object)
    beh_p = np.cumsum([0.30, 0.50, 0.20])
    beh_idx = np.searchsorted(beh_p, u, side="right").clip(0, 2)
    fash = category == "FASHION"
    gen_fash = np.cumsum(np.array([0.35, 0.20, 0.15, 0.15, 0.07, 0.05, 0.03]))
    gen_other = np.cumsum(np.array([0.10, 0.20, 0.20, 0.25, 0.12, 0.08, 0.05]))
    idx_f = np.searchsorted(gen_fash, u, side="right").clip(0, 6)
    idx_o = np.searchsorted(gen_other, u, side="right").clip(0, 6)
    gen_idx = np.where(fash, idx_f, idx_o)
    out[:] = names[np.where(behavioural, beh_idx, gen_idx)]
    return out


def generate_dataset(n: int, seed: int) -> GeneratedDataset:
    """Generate ``n`` orders (incl. 18 fixed scenario orders) deterministically from ``seed``."""
    scn_orders, scn_ships, scn_rets = _scenario_rows()
    n_rand = max(n - len(scn_orders), 1)
    ss = np.random.SeedSequence(seed).spawn(4)
    rc, ro, rs, rr = (np.random.default_rng(s) for s in ss)

    # ---- customers -------------------------------------------------------------------
    n_cust = int(np.ceil(n_rand / 2.6)) + 8
    cohort_idx = rc.choice(4, size=n_cust, p=COHORT_P)
    n_orders_c = np.select(
        [cohort_idx == 0, cohort_idx == 1, cohort_idx == 2, cohort_idx == 3],
        [
            1 + rc.poisson(1.2, n_cust),
            4 + rc.poisson(2.0, n_cust),
            1 + rc.poisson(1.5, n_cust),
            2 + rc.poisson(2.0, n_cust),
        ],
    )
    cust_of = np.repeat(np.arange(n_cust), n_orders_c)[:n_rand]
    cohort_c = np.array(COHORTS)[cohort_idx]
    risky_pick = rc.integers(0, len(RISKY_PINCODES), n_cust)
    normal_pick = rc.integers(0, len(NORMAL_PINCODES), n_cust)
    stray = rc.random(n_cust) < 0.03
    pin_c = np.where(
        (cohort_idx == 2) | stray,
        np.array(RISKY_PINCODES)[risky_pick],
        np.array(NORMAL_PINCODES)[normal_pick],
    )
    home_cat = np.where(cohort_idx == 1, 0, rc.choice(4, size=n_cust, p=CAT_P))

    # ---- orders ----------------------------------------------------------------------
    cust = cust_of
    coh = cohort_c[cust]
    cat_i = np.where(
        ro.random(n_rand) < np.where(cohort_idx[cust] == 1, 0.92, 0.75),
        home_cat[cust],
        ro.choice(4, size=n_rand, p=CAT_P),
    )
    category = np.array(CATEGORIES)[cat_i]
    sku_num = np.minimum(ro.geometric(0.12, n_rand), 25)
    sku = np.array(
        [f"sku_{c[:3]}_{k:03d}" for c, k in zip(category, sku_num, strict=True)], dtype=object
    )
    use_hr = ro.random(n_rand) < 0.07
    hr_pick = np.array(HIGH_RETURN_SKUS, dtype=object)[ro.integers(0, 4, n_rand)]
    sku = np.where(use_hr, hr_pick, sku)
    category = np.where(
        use_hr,
        np.array(
            [
                {"FAS": "FASHION", "BEA": "BEAUTY", "LIF": "LIFESTYLE", "ELE": "ELECTRONICS"}[
                    s[4:7]
                ]
                for s in sku
            ]
        ),
        category,
    )
    is_hr = np.isin(sku, HIGH_RETURN_SKUS)
    cod_p = np.array([P_COD[c] for c in coh])
    pay = np.where(ro.random(n_rand) < cod_p, "COD", "PREPAID")
    pay = np.where((pay == "COD") & (ro.random(n_rand) < 0.03), "PARTIAL_COD", pay)
    med = np.array([CAT_MEDIAN_INR[c] for c in category])
    amount = np.round(med * ro.lognormal(0.0, 0.45, n_rand), 0)
    qty = np.where(ro.random(n_rand) < 0.15, 2, 1)
    amount = amount * qty
    day_off = ro.uniform(0, SPAN_DAYS, n_rand)
    order_ts = ANCHOR + pd.to_timedelta(np.round(day_off * 86400), unit="s")
    pin = pin_c[cust].astype(object)
    pin = np.where(ro.random(n_rand) < 0.01, None, pin)

    df = (
        pd.DataFrame(
            {
                "customer_idx": cust,
                "cohort": coh,
                "order_ts": order_ts,
                "sku_id": sku,
                "category": category,
                "quantity": qty,
                "payment_mode": pay,
                "order_amount_inr": amount,
                "pincode": pin,
                "is_hr": is_hr,
            }
        )
        .sort_values(["order_ts", "customer_idx"], kind="stable")
        .reset_index(drop=True)
    )
    df["order_id"] = [f"ord_{i + 1:06d}" for i in range(len(df))]
    df["customer_id"] = [f"cus_{c + 1:04d}" for c in df["customer_idx"]]

    # ---- shipments -------------------------------------------------------------------
    m = len(df)
    is_cod = df["payment_mode"].isin(["COD", "PARTIAL_COD"]).to_numpy()
    p_rto = np.where(is_cod, df["cohort"].map(P_RTO_COD), df["cohort"].map(P_RTO_PREPAID))
    u = rs.random(m)
    status = np.full(m, "DELIVERED", dtype=object)
    status[u < p_rto] = "RTO"
    status[(u >= p_rto) & (u < p_rto + 0.004)] = "LOST"
    status[(u >= p_rto + 0.004) & (u < p_rto + 0.016)] = "NDR"
    shipped = df["order_ts"] + _hours(rs.uniform(2, 30, m))
    delivered = shipped + _hours(rs.uniform(48, 144, m))
    in_transit = (status == "DELIVERED") & (delivered > SNAPSHOT)
    status[in_transit] = "IN_TRANSIT"
    attempts = np.where(
        status == "DELIVERED",
        1 + (rs.random(m) < 0.15) + (rs.random(m) < 0.03),
        np.where(
            status == "RTO",
            rs.integers(2, 4, m),
            np.where(status == "NDR", 3, np.where(status == "LOST", 1, 1)),
        ),
    )
    true_delivered = delivered.where(status == "DELIVERED")

    # ---- returns ---------------------------------------------------------------------
    delivered_ok = status == "DELIVERED"
    cat_arr = df["category"].to_numpy()
    cohort_arr = df["cohort"].to_numpy()
    p_base = np.array([P_RETURN_BASE[c] for c in cat_arr])
    p_extra = np.zeros(m)
    ward = cohort_arr == "WARDROBER"
    p_extra = np.where(ward & (cat_arr == "FASHION"), 0.65, p_extra)
    p_extra = np.where(ward & (cat_arr != "FASHION"), 0.15, p_extra)
    p_extra = np.where(df["is_hr"].to_numpy(), np.maximum(p_extra, 0.40), p_extra)
    p_extra = np.where(cohort_arr == "RISKY_PIN", np.maximum(p_extra, 0.12), p_extra)
    p_extra = np.where(cohort_arr == "LOW_INTENT_COD", np.maximum(p_extra, 0.15), p_extra)
    genuine = rr.random(m) < p_base
    behavioural = rr.random(m) < p_extra
    returned = delivered_ok & (genuine | behavioural)
    reason = _draw_reasons(rr, behavioural, cat_arr)
    defect = np.isin(reason, ["DEFECTIVE", "DAMAGED_IN_TRANSIT", "WRONG_ITEM"])
    ttr = np.exp(np.log(60.0) + 0.8 * rr.standard_normal(m))
    legit_fast = rr.random(m) < 0.04
    ttr = np.where(
        legit_fast & ~behavioural, np.exp(np.log(4.0) + 0.5 * rr.standard_normal(m)), ttr
    )
    ttr = np.where(defect, np.exp(np.log(12.0) + 1.0 * rr.standard_normal(m)), ttr)
    ttr = np.where(behavioural & ward, np.exp(np.log(4.0) + 0.5 * rr.standard_normal(m)), ttr)
    ttr = np.where(behavioural & ~ward, np.exp(np.log(48.0) + 0.9 * rr.standard_normal(m)), ttr)
    late = rr.random(m)
    ttr = np.where(~defect & ~(behavioural & ward) & (late < 0.05), rr.uniform(120, 168, m), ttr)
    ttr = np.where(~defect & ~(behavioural & ward) & (late > 0.98), rr.uniform(170, 220, m), ttr)
    ttr = np.clip(ttr, 0.5, 400.0)
    req = true_delivered + _hours(ttr)
    returned &= (req <= SNAPSHOT).to_numpy()
    acc = req + _hours(rr.uniform(6, 48, m))
    rstatus = np.where(
        ttr > 168,
        np.where(rr.random(m) < 0.8, "REJECTED", "ACCEPTED"),
        np.where(rr.random(m) < 0.7, "REFUNDED", "ACCEPTED"),
    )
    rstatus = np.where((acc > SNAPSHOT).to_numpy() & (rstatus != "REJECTED"), "REQUESTED", rstatus)
    qu = rr.random(m)
    qc = np.where(
        behavioural & ward,
        np.where(qu < 0.5, "USED", np.where(qu < 0.9, "UNUSED", "UNKNOWN")),
        np.where(
            defect,
            np.where(qu < 0.5, "UNKNOWN", "UNUSED"),
            np.where(
                qu < 0.75,
                "UNUSED",
                np.where(qu < 0.85, "USED", np.where(qu < 0.9, "DAMAGED_BY_CUSTOMER", "UNKNOWN")),
            ),
        ),
    )
    accepted_like = np.isin(rstatus, ["ACCEPTED", "REFUNDED"])

    orders = df.assign(order_amount_inr=df["order_amount_inr"])[
        [
            "order_id",
            "customer_id",
            "order_ts",
            "sku_id",
            "category",
            "quantity",
            "payment_mode",
            "order_amount_inr",
            "pincode",
        ]
    ].copy()
    ships = pd.DataFrame(
        {
            "order_id": df["order_id"],
            "shipped_ts": shipped,
            "delivered_ts": true_delivered,
            "delivery_status": status,
            "delivery_attempts": attempts,
        }
    )
    miss = (status == "DELIVERED") & (rs.random(m) < 0.02)
    # missing delivered_at is applied after returns are computed (true delivery drove TTR)
    ships.loc[miss, "delivered_ts"] = pd.NaT
    rets = pd.DataFrame(
        {
            "order_id": df["order_id"],
            "req_ts": req,
            "acc_ts": acc.where(accepted_like),
            "return_reason": reason,
            "return_status": rstatus,
            "qc_result": np.where(accepted_like, qc, None),
            "refund_amount_inr": np.where(rstatus == "REFUNDED", df["order_amount_inr"], np.nan),
        }
    )[returned].reset_index(drop=True)
    labels = pd.DataFrame(
        {
            "order_id": df["order_id"],
            "customer_id": df["customer_id"],
            "cohort": df["cohort"],
            "is_rto": status == "RTO",
            "is_return": returned,
            "is_preventable_return": returned & behavioural & ~defect,
        }
    )

    # ---- scenario block (fixed) ------------------------------------------------------
    so = pd.DataFrame(scn_orders)
    sh = pd.DataFrame(scn_ships)
    sr = pd.DataFrame(scn_rets)
    orders = pd.concat([orders, so], ignore_index=True)
    ships = pd.concat([ships, sh], ignore_index=True)
    rets = pd.concat([rets, sr], ignore_index=True)
    sl = so[["order_id", "customer_id"]].assign(
        cohort="SCENARIO", is_rto=False, is_return=False, is_preventable_return=False
    )
    sl["is_rto"] = sl["order_id"].isin(["ord_scn_02_h1", "ord_scn_02_h2"])
    sl["is_return"] = sl["order_id"].isin(sr["order_id"])
    sl["is_preventable_return"] = sl["order_id"].isin(
        ["ord_scn_04_h2", "ord_scn_04_h3", "ord_scn_04_h4", "ord_scn_04", "ord_scn_05"]
    )
    labels = pd.concat([labels, sl], ignore_index=True)

    orders = orders.sort_values(["order_ts", "order_id"], kind="stable").reset_index(drop=True)
    ships = ships.set_index("order_id").loc[orders["order_id"]].reset_index()
    rets = rets.sort_values(["req_ts", "order_id"], kind="stable").reset_index(drop=True)
    rets.insert(0, "return_id", [f"ret_{i + 1:06d}" for i in range(len(rets))])
    labels = labels.set_index("order_id").loc[orders["order_id"]].reset_index()

    out_orders = pd.DataFrame(
        {
            "order_id": orders["order_id"],
            "customer_id": orders["customer_id"],
            "order_date": _iso(orders["order_ts"]),
            "sku_id": orders["sku_id"],
            "category": orders["category"],
            "quantity": orders["quantity"].astype(int),
            "payment_mode": orders["payment_mode"],
            "order_amount_inr": orders["order_amount_inr"].astype(float).round(2),
            "pincode": orders["pincode"],
            "customer_contact": None,
        }
    )
    out_ships = pd.DataFrame(
        {
            "order_id": ships["order_id"],
            "shipped_at": _iso(pd.to_datetime(ships["shipped_ts"])),
            "delivered_at": _iso(pd.to_datetime(ships["delivered_ts"])),
            "delivery_status": ships["delivery_status"],
            "delivery_attempts": ships["delivery_attempts"].astype(int),
            "forward_shipping_inr": np.nan,
            "packaging_inr": np.nan,
        }
    )
    out_rets = pd.DataFrame(
        {
            "return_id": rets["return_id"],
            "order_id": rets["order_id"],
            "return_requested_at": _iso(pd.to_datetime(rets["req_ts"])),
            "return_accepted_at": _iso(pd.to_datetime(rets["acc_ts"])),
            "return_reason": rets["return_reason"],
            "return_status": rets["return_status"],
            "qc_result": rets["qc_result"],
            "refund_amount_inr": rets["refund_amount_inr"].astype(float),
            "reverse_shipping_inr": np.nan,
            "handling_inr": np.nan,
            "writeoff_inr": np.nan,
            "cost_recovery_inr": np.nan,
        }
    )
    return GeneratedDataset(out_orders, out_ships, out_rets, labels, seed=seed, n_orders=n)


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Canonical CSV bytes: fixed column order, LF endings, fixed float format."""
    return frame.to_csv(index=False, lineterminator="\n", na_rep="", float_format="%.2f").encode(
        "utf-8"
    )
