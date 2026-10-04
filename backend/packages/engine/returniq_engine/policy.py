"""Policy simulator: target the top X% by risk index and estimate what it would capture.

Uses observed outcomes only (never the hidden generator labels). "Preventable" here is the
observable proxy: a reason-eligible return (Stage 2) or an RTO / reason-eligible return (Stage 1).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from returniq_contracts import PolicySimulation, Settings, Stage

from . import loss as loss_mod
from .assess import get_view, stage_scores
from .prepare import Dataset

if TYPE_CHECKING:
    from .ml import TrainedModel


def outcome_arrays(ds: Dataset, stage: Stage, s: Settings) -> dict[str, np.ndarray]:
    """Observed per-stage-row arrays: any return, preventable proxy, loss and order value."""
    v = get_view(ds, stage)
    m = ds.master.iloc[v.positions]
    cost = s.cost_defaults_paise
    ret_loss, _, _ = loss_mod.return_losses(m, cost)
    rto = (m["delivery_status"] == "RTO").to_numpy()
    fwd = m["fwd_paise"].fillna(cost.forward_ship_paise).to_numpy(dtype=float)
    pack = m["pack_paise"].fillna(cost.packaging_paise).to_numpy(dtype=float)
    rto_loss = np.where(rto, fwd + cost.reverse_ship_paise + pack, 0.0)
    is_ret = m["ret_ts"].notna().to_numpy()
    elig_ret = is_ret & m["reason_eligible"].to_numpy(dtype=bool)
    proxy = elig_ret if stage == Stage.POST_DELIVERY else (elig_ret | rto)
    return {
        "is_return": is_ret,
        "proxy": proxy,
        "loss": np.nan_to_num(ret_loss) + rto_loss,
        "value": m["amount_paise"].to_numpy(dtype=float),
    }


def rank_order(scores: np.ndarray, order_ids: list[str]) -> np.ndarray:
    """Indices by score desc (NaN last), ties broken by order_id for determinism."""
    filled = np.where(np.isnan(scores), -1.0, scores)
    ids = np.array(order_ids)
    return np.lexsort((ids, -filled))


def simulate_policy(
    ds: Dataset, stage: Stage, top_pct: float, s: Settings, model: TrainedModel | None
) -> PolicySimulation:
    """Top ``top_pct`` percent (0..100) of the stage population by risk index."""
    if not 0 <= top_pct <= 100:
        raise ValueError("top_pct must be between 0 and 100")
    v = get_view(ds, stage)
    scores = stage_scores(ds, stage, s, model)
    n = len(scores)
    k = min(n, math.ceil(n * top_pct / 100.0))
    sel = np.zeros(n, dtype=bool)
    sel[rank_order(scores, v.order_ids)[:k]] = True
    o = outcome_arrays(ds, stage, s)
    proxy_total = int(o["proxy"].sum())
    ret_total = int(o["is_return"].sum())
    cost = s.cost_defaults_paise
    eligible_loss = float(o["loss"][sel & o["proxy"]].sum())
    net = (
        eligible_loss * s.effectiveness_assumption.base
        - k * cost.intervention_cost_paise
        - float(o["value"][sel].sum()) * cost.friction_margin_loss_pct / 100.0
    )
    return PolicySimulation(
        top_pct=top_pct,
        stage=stage,
        orders_targeted=k,
        preventable_captured_pct=round(float((sel & o["proxy"]).sum() / proxy_total * 100), 2)
        if proxy_total
        else None,
        returns_captured_pct=round(float((sel & o["is_return"]).sum() / ret_total * 100), 2)
        if ret_total
        else None,
        est_net_avoidable_paise=int(round(net)),
        false_positive_estimate=round(float((sel & ~o["proxy"]).sum() / k), 4) if k else None,
    )
