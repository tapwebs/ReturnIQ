"""Plain-English signals, confidence and reason text. No model internals leak to labels."""

from __future__ import annotations

import numpy as np
from returniq_contracts import RiskLevel, Signal, SignalSource

from .rules import FAULT_CODE, NO_HISTORY_CODE, SIGNAL_TEXT

FIELD_WEIGHTS_S1 = {
    "payment_mode": 0.25,
    "amount": 0.15,
    "sku": 0.15,
    "category": 0.15,
    "pincode": 0.30,
}
FIELD_WEIGHTS_S2 = {
    "payment_mode": 0.15,
    "amount": 0.10,
    "sku": 0.10,
    "category": 0.10,
    "pincode": 0.20,
    "delivered_at": 0.25,
    "return_reason": 0.10,
}
COLD_START_DEPTH_FLOOR = 0.5
DEPTH_FULL_AT_ORDERS = 5
MISSING_CORE_CONFIDENCE_CAP = 0.25

FEATURE_LABELS = {
    "prior_orders": "Number of earlier orders",
    "prior_rto": "Earlier returned-to-origin orders",
    "prior_failed": "Earlier failed deliveries",
    "prior_returns": "Earlier returns",
    "prior_elig_returns": "Earlier preventable-type returns",
    "prior_return_rate": "Customer return rate",
    "prior_rto_rate": "Customer failed-delivery rate",
    "recent_returns_30d": "Returns in the last 30 days",
    "prior_fast_share": "Share of fast returns",
    "prior_flagged": "Earlier flagged returns",
    "cust_median_ttr": "Customer median return time",
    "pin_rto_lift": "Pincode failed-delivery lift",
    "pin_ret_lift": "Pincode return lift",
    "sku_ret_lift": "SKU return lift",
    "cat_ret_lift": "Category return lift",
    "is_cod": "Cash on delivery",
    "amount_paise": "Order value",
    "value_ratio": "Value vs category",
    "cat_code": "Product category",
    "prior_delivered": "Earlier deliveries",
    "ttr_h": "Time to return",
    "ttr_known": "Delivery time known",
    "ttr_ratio": "Return speed vs median",
    "ttr_z": "Return speed vs seller pattern",
    "seller_median_ttr": "Seller median return time",
    "is_fast_ttr": "Return within 6 hours",
    "delivery_attempts": "Delivery attempts",
    "reason_eligible": "Return reason type",
    "same_cat_returns": "Repeat returns in category",
}


def confidence(
    stage: int,
    present: dict[str, np.ndarray],
    prior_orders: np.ndarray,
    has_delivered: np.ndarray | None,
) -> np.ndarray:
    """confidence = data completeness x history depth, capped when the core S2 field is missing."""
    weights = FIELD_WEIGHTS_S1 if stage == 1 else FIELD_WEIGHTS_S2
    n = len(prior_orders)
    comp = np.zeros(n)
    for k, w in weights.items():
        comp += w * present.get(k, np.ones(n, dtype=bool))
    comp /= sum(weights.values())
    depth = COLD_START_DEPTH_FLOOR + (1 - COLD_START_DEPTH_FLOOR) * np.minimum(
        1.0, prior_orders / DEPTH_FULL_AT_ORDERS
    )
    conf = comp * depth
    if stage == 2 and has_delivered is not None:
        conf = np.where(has_delivered, conf, np.minimum(conf, MISSING_CORE_CONFIDENCE_CAP))
    return np.round(conf, 3)


def rule_signals(points: dict[str, float], row: dict, stage: int) -> list[Signal]:
    """Signals for every rule that fired, sorted by (-contribution, code)."""
    row = {**row, "_stage": stage}
    out: list[Signal] = []
    for code, pts in points.items():
        if pts <= 0 or code in ("total", FAULT_CODE):
            continue
        label, evidence = SIGNAL_TEXT[code]
        out.append(
            Signal(
                code=code,
                label=label(row),
                evidence=evidence(row),
                source=SignalSource.RULE,
                contribution=round(float(pts), 1),
            )
        )
    if points.get(FAULT_CODE, 0) > 0:
        label, evidence = SIGNAL_TEXT[FAULT_CODE]
        out.append(
            Signal(
                code=FAULT_CODE,
                label=label(row),
                evidence=evidence(row),
                source=SignalSource.RULE,
                contribution=0.0,
            )
        )
    if row.get("prior_orders", 1) == 0:
        label, evidence = SIGNAL_TEXT[NO_HISTORY_CODE]
        out.append(
            Signal(
                code=NO_HISTORY_CODE,
                label=label(row),
                evidence=evidence(row),
                source=SignalSource.RULE,
                contribution=0.0,
            )
        )
    out.sort(key=lambda s: (-s.contribution, s.code))
    return out


def ml_signals(
    names: list[str], contrib: np.ndarray, ml_component: float, row: dict, top: int = 3
) -> list[Signal]:
    """Top positive LightGBM contributions as plain-English signals (points scaled to the score)."""
    pos = np.where(contrib > 0, contrib, 0.0)
    total = pos.sum()
    if total <= 0:
        return []
    out: list[Signal] = []
    for i in np.argsort(-pos, kind="stable")[:top]:
        if pos[i] <= 0:
            continue
        name = names[i]
        val = row.get(name)
        shown = "n/a" if val is None or (isinstance(val, float) and np.isnan(val)) else f"{val:.2f}"
        out.append(
            Signal(
                code=f"ML_{name.upper()}",
                label=f"Model: {FEATURE_LABELS.get(name, name)}",
                evidence=f"{name}={shown}",
                source=SignalSource.ML,
                contribution=round(float(pos[i] / total * ml_component), 1),
            )
        )
    return out


def reason_text(
    level: RiskLevel, score: int | None, signals: list[Signal], missing: list[str]
) -> str:
    """One-sentence summary built from the strongest signals."""
    if level == RiskLevel.INSUFFICIENT_DATA:
        why = ", ".join(missing) if missing else "limited history"
        return f"Not enough reliable data to score this order ({why})."
    if any(s.code == FAULT_CODE for s in signals):
        return (
            f"Risk index {score} ({level.value}): the return reason points to a seller or "
            "carrier issue, so it is not treated as preventable."
        )
    top = [s.label for s in signals if s.contribution > 0][:2]
    body = "; ".join(top) if top else "no strong risk signals"
    return f"Risk index {score} ({level.value}): {body}."
