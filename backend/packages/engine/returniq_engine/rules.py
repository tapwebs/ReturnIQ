"""Deterministic rules ``rules-1.0.0``: the published points table and signal text.

Each rule is a vectorised mask times its points. The rule component is ``min(sum, 100)``.
Return-related history counts only reason-eligible returns (seller/carrier-fault reasons
never add behavioural points). Tuning the HIGH share touches ``POINTS_TABLE`` only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .settings import RULE_VERSION

POINTS_TABLE: dict[str, Mapping[str, float]] = {
    "COD_WITH_PRIOR_RTO": {"one_prior": 30, "two_or_more_prior": 40},
    "RTO_RATE_HIGH": {"points": 20, "min_orders": 2, "min_rate": 0.5},
    "FAILED_DELIVERY_HISTORY": {"two_or_more": 10, "one": 5},
    "RETURN_FREQ_HIGH": {"rate_ge_0.5": 25, "rate_ge_0.3": 12, "min_orders": 3},
    "RETURN_FREQ_RECENT": {"points": 10, "min_returns_30d": 2},
    "FAST_RETURN_HISTORY": {"points": 15, "min_returns": 2, "min_fast_share": 0.5},
    "FLAGGED_PRIOR_RETURN": {"points": 15},
    "RISKY_PINCODE_LIFT": {"max_points": 15, "lift_from": 1.2, "lift_to": 2.0},
    "HIGH_RETURN_SKU": {"max_points": 10, "lift_from": 1.3, "lift_to": 2.2},
    "HIGH_RETURN_CATEGORY": {"max_points": 6, "lift_from": 1.15, "lift_to": 1.5},
    "COD_HIGH_VALUE_VS_CATEGORY": {"points": 8, "min_ratio": 1.5},
    "COD_NEW_CUSTOMER": {"points": 5},
    "TTR_ABNORMALLY_FAST": {"points": 20, "max_hours": 6, "max_ratio_to_seller_median": 0.5},
    "SAME_CATEGORY_REPEAT": {"two_or_more": 10, "one": 5},
    "POLICY_WINDOW_VIOLATION": {"points": 15},
}
STAGE1_RULES = (
    "COD_WITH_PRIOR_RTO",
    "RTO_RATE_HIGH",
    "FAILED_DELIVERY_HISTORY",
    "RETURN_FREQ_HIGH",
    "RETURN_FREQ_RECENT",
    "FAST_RETURN_HISTORY",
    "FLAGGED_PRIOR_RETURN",
    "RISKY_PINCODE_LIFT",
    "HIGH_RETURN_SKU",
    "HIGH_RETURN_CATEGORY",
    "COD_HIGH_VALUE_VS_CATEGORY",
    "COD_NEW_CUSTOMER",
)
STAGE2_RULES = (
    "COD_WITH_PRIOR_RTO",
    "RETURN_FREQ_HIGH",
    "RETURN_FREQ_RECENT",
    "FAST_RETURN_HISTORY",
    "RISKY_PINCODE_LIFT",
    "HIGH_RETURN_SKU",
    "HIGH_RETURN_CATEGORY",
    "TTR_ABNORMALLY_FAST",
    "SAME_CATEGORY_REPEAT",
    "POLICY_WINDOW_VIOLATION",
    "COD_HIGH_VALUE_VS_CATEGORY",
)
FAULT_CODE = "SELLER_OR_CARRIER_FAULT_REASON"
NO_HISTORY_CODE = "NO_PRIOR_HISTORY"


def _scaled(lift: np.ndarray, lo: float, hi: float, max_pts: float) -> np.ndarray:
    return max_pts * np.clip((np.nan_to_num(lift, nan=1.0) - lo) / (hi - lo), 0.0, 1.0)


def compute_rules(f: pd.DataFrame, stage: int, window_days: int) -> pd.DataFrame:
    """Points per rule code plus ``total`` (capped at 100) for a stage feature frame."""
    pt = POINTS_TABLE
    n = len(f)
    cod = f["is_cod"].to_numpy() > 0
    po = f["prior_orders"].to_numpy()
    rate = np.nan_to_num(f["prior_return_rate"].to_numpy(), nan=0.0)
    rto_rate = np.nan_to_num(f["prior_rto_rate"].to_numpy(), nan=0.0)
    rto = f["prior_rto"].to_numpy()
    cols: dict[str, np.ndarray] = {}
    cols["COD_WITH_PRIOR_RTO"] = np.where(
        cod & (rto >= 2),
        pt["COD_WITH_PRIOR_RTO"]["two_or_more_prior"],
        np.where(cod & (rto >= 1), pt["COD_WITH_PRIOR_RTO"]["one_prior"], 0.0),
    )
    cols["RTO_RATE_HIGH"] = np.where(
        (po >= 2) & (rto_rate >= 0.5), pt["RTO_RATE_HIGH"]["points"], 0.0
    )
    failed = f["prior_failed"].to_numpy()
    cols["FAILED_DELIVERY_HISTORY"] = np.where(
        failed >= 2,
        pt["FAILED_DELIVERY_HISTORY"]["two_or_more"],
        np.where(failed >= 1, pt["FAILED_DELIVERY_HISTORY"]["one"], 0.0),
    )
    cols["RETURN_FREQ_HIGH"] = np.where(
        (po >= 3) & (rate >= 0.5),
        pt["RETURN_FREQ_HIGH"]["rate_ge_0.5"],
        np.where((po >= 3) & (rate >= 0.3), pt["RETURN_FREQ_HIGH"]["rate_ge_0.3"], 0.0),
    )
    cols["RETURN_FREQ_RECENT"] = np.where(
        f["recent_returns_30d"].to_numpy() >= 2, pt["RETURN_FREQ_RECENT"]["points"], 0.0
    )
    fast_share = np.nan_to_num(f["prior_fast_share"].to_numpy(), nan=0.0)
    cols["FAST_RETURN_HISTORY"] = np.where(
        (f["prior_elig_returns"].to_numpy() >= 2) & (fast_share >= 0.5),
        pt["FAST_RETURN_HISTORY"]["points"],
        0.0,
    )
    if stage == 1:
        cols["FLAGGED_PRIOR_RETURN"] = np.where(
            f["prior_flagged"].to_numpy() >= 1, pt["FLAGGED_PRIOR_RETURN"]["points"], 0.0
        )
        pin = f["pin_rto_lift"].to_numpy()
    else:
        pin = f["pin_ret_lift"].to_numpy()
    p = pt["RISKY_PINCODE_LIFT"]
    cols["RISKY_PINCODE_LIFT"] = _scaled(pin, p["lift_from"], p["lift_to"], p["max_points"])
    p = pt["HIGH_RETURN_SKU"]
    cols["HIGH_RETURN_SKU"] = _scaled(
        f["sku_ret_lift"].to_numpy(), p["lift_from"], p["lift_to"], p["max_points"]
    )
    p = pt["HIGH_RETURN_CATEGORY"]
    cols["HIGH_RETURN_CATEGORY"] = _scaled(
        f["cat_ret_lift"].to_numpy(), p["lift_from"], p["lift_to"], p["max_points"]
    )
    cols["COD_HIGH_VALUE_VS_CATEGORY"] = np.where(
        cod & (f["value_ratio"].to_numpy() >= pt["COD_HIGH_VALUE_VS_CATEGORY"]["min_ratio"]),
        pt["COD_HIGH_VALUE_VS_CATEGORY"]["points"],
        0.0,
    )
    if stage == 1:
        cols["COD_NEW_CUSTOMER"] = np.where(cod & (po == 0), pt["COD_NEW_CUSTOMER"]["points"], 0.0)
    else:
        ttr = f["ttr_h"].to_numpy()
        known = f["ttr_known"].to_numpy() > 0
        ratio = f["ttr_ratio"].to_numpy()
        t = pt["TTR_ABNORMALLY_FAST"]
        cols["TTR_ABNORMALLY_FAST"] = np.where(
            known & (ttr < t["max_hours"]) & (ratio < t["max_ratio_to_seller_median"]),
            t["points"],
            0.0,
        )
        sc = f["same_cat_returns"].to_numpy()
        cols["SAME_CATEGORY_REPEAT"] = np.where(
            sc >= 2,
            pt["SAME_CATEGORY_REPEAT"]["two_or_more"],
            np.where(sc >= 1, pt["SAME_CATEGORY_REPEAT"]["one"], 0.0),
        )
        cols["POLICY_WINDOW_VIOLATION"] = np.where(
            known & (ttr > window_days * 24), pt["POLICY_WINDOW_VIOLATION"]["points"], 0.0
        )
    codes = STAGE1_RULES if stage == 1 else STAGE2_RULES
    out = pd.DataFrame({c: cols[c] for c in codes})
    if stage == 2:
        fault = f["reason_eligible"].to_numpy() <= 0
        out.loc[fault, :] = 0.0
        out[FAULT_CODE] = fault.astype(float)
    out["total"] = np.minimum(out[list(codes)].sum(axis=1).to_numpy(), 100.0) if n else 0.0
    return out


def rule_fn(frame: pd.DataFrame, stage: int, window_days: int) -> pd.DataFrame:
    """Callable handed to the feature builder (keeps features.py free of rule imports)."""
    return compute_rules(frame, stage, window_days)


def _pct(x: Any) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.0f}%"


def _fmt(x: Any, nd: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# code -> (label(row), evidence(row)); rows are dicts of stage feature values
SIGNAL_TEXT: dict[str, tuple[Callable[[dict], str], Callable[[dict], str]]] = {
    "COD_WITH_PRIOR_RTO": (
        lambda r: (
            f"Cash-on-delivery order; {int(r['prior_rto'])} earlier order(s) returned to origin"
        ),
        lambda r: f"payment=COD, prior_rto={int(r['prior_rto'])}",
    ),
    "RTO_RATE_HIGH": (
        lambda r: "Most earlier deliveries by this customer failed",
        lambda r: f"prior_rto={int(r['prior_rto'])}, orders={int(r['prior_orders'])}",
    ),
    "FAILED_DELIVERY_HISTORY": (
        lambda r: f"{int(r['prior_failed'])} earlier failed deliveries",
        lambda r: f"prior_failed={int(r['prior_failed'])}",
    ),
    "RETURN_FREQ_HIGH": (
        lambda r: (
            f"{int(r['prior_elig_returns'])} returns in {int(r['prior_orders'])} earlier orders"
        ),
        lambda r: (
            f"returns={int(r['prior_elig_returns'])}, orders={int(r['prior_orders'])}, "
            f"rate={_pct(r['prior_return_rate'])}"
        ),
    ),
    "RETURN_FREQ_RECENT": (
        lambda r: f"{int(r['recent_returns_30d'])} returns in the last 30 days",
        lambda r: f"returns_30d={int(r['recent_returns_30d'])}",
    ),
    "FAST_RETURN_HISTORY": (
        lambda r: "Earlier returns were mostly raised within hours of delivery",
        lambda r: f"fast_return_share={_pct(r['prior_fast_share'])}",
    ),
    "FLAGGED_PRIOR_RETURN": (
        lambda r: "An earlier return from this customer was flagged for review",
        lambda r: f"flagged_prior_returns={int(r['prior_flagged'])}",
    ),
    "RISKY_PINCODE_LIFT": (
        lambda r: "Pincode has above-average failed deliveries or returns for this seller",
        lambda r: (
            "pincode_lift="
            + _fmt(r.get("pin_rto_lift") if r["_stage"] == 1 else r.get("pin_ret_lift"), 2)
            + " vs seller baseline"
        ),
    ),
    "HIGH_RETURN_SKU": (
        lambda r: "This SKU is returned more often than the seller average",
        lambda r: f"sku_return_lift={_fmt(r['sku_ret_lift'], 2)}",
    ),
    "HIGH_RETURN_CATEGORY": (
        lambda r: "This category is returned more often than the seller average",
        lambda r: f"category_return_lift={_fmt(r['cat_ret_lift'], 2)}",
    ),
    "COD_HIGH_VALUE_VS_CATEGORY": (
        lambda r: "Cash-on-delivery order well above the usual value for its category",
        lambda r: f"value_vs_category_median={_fmt(r['value_ratio'], 2)}x",
    ),
    "COD_NEW_CUSTOMER": (
        lambda r: "First cash-on-delivery order from this customer",
        lambda r: "prior_orders=0, payment=COD",
    ),
    "TTR_ABNORMALLY_FAST": (
        lambda r: (
            f"Return raised in {_fmt(r['ttr_h'])} h, about "
            f"{_fmt(1 / r['ttr_ratio'], 0) if r['ttr_ratio'] else 'n/a'}x faster than "
            "your median"
        ),
        lambda r: f"ttr_h={_fmt(r['ttr_h'])}, seller_median_ttr_h={_fmt(r['seller_median_ttr'])}",
    ),
    "SAME_CATEGORY_REPEAT": (
        lambda r: "Repeat returns in the same category",
        lambda r: f"same_category_returns={int(r['same_cat_returns'])}",
    ),
    "POLICY_WINDOW_VIOLATION": (
        lambda r: "Return requested outside the return window",
        lambda r: f"ttr_h={_fmt(r['ttr_h'])}",
    ),
    FAULT_CODE: (
        lambda r: "Reason points to a seller or carrier issue, so it is not treated as preventable",
        lambda r: "reason in DEFECTIVE / DAMAGED_IN_TRANSIT / WRONG_ITEM",
    ),
    NO_HISTORY_CODE: (
        lambda r: "No earlier orders: scored on SKU, category, pincode and payment signals",
        lambda r: "prior_orders=0 (cold start)",
    ),
}

__all__ = ["RULE_VERSION", "POINTS_TABLE", "compute_rules", "rule_fn", "SIGNAL_TEXT"]
