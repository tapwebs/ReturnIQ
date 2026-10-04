"""Loss model (0.3): observed operating loss excludes order value; scenario net-avoidable loss."""

from __future__ import annotations

import numpy as np
import pandas as pd
from returniq_contracts import (
    CostDefaults,
    LossBreakdown,
    LossEstimate,
    LossSummary,
    ScenarioAmounts,
    Settings,
)

COUNTED_STATUSES = ("ACCEPTED", "REFUNDED")
WRITEOFF_QC = ("USED", "DAMAGED_BY_CUSTOMER")


def _pick(col: pd.Series, default: float, counted: np.ndarray) -> tuple[np.ndarray, bool]:
    provided = col.notna().to_numpy()
    vals = np.where(provided, col.fillna(0).to_numpy(dtype=float), float(default))
    return vals, bool((counted & ~provided).any())


def return_losses(m: pd.DataFrame, cost: CostDefaults) -> tuple[np.ndarray, dict, list[str]]:
    """Per-row loss (paise) for accepted returns, breakdown totals and assumed fields."""
    counted = m["return_status"].isin(COUNTED_STATUSES).to_numpy()
    fwd, a1 = _pick(m["fwd_paise"], cost.forward_ship_paise, counted)
    rev, a2 = _pick(m["rev_paise"], cost.reverse_ship_paise, counted)
    pack, a3 = _pick(m["pack_paise"], cost.packaging_paise, counted)
    hand, a4 = _pick(m["handling_paise"], cost.handling_paise, counted)
    wo_default = np.where(
        m["qc_result"].isin(WRITEOFF_QC).to_numpy(),
        np.round(m["amount_paise"].to_numpy() * cost.writeoff_pct / 100.0),
        0.0,
    )
    wo_provided = m["writeoff_paise"].notna().to_numpy()
    wo = np.where(wo_provided, m["writeoff_paise"].fillna(0).to_numpy(dtype=float), wo_default)
    a5 = bool((counted & ~wo_provided & m["qc_result"].isin(WRITEOFF_QC).to_numpy()).any())
    rec_provided = m["recovery_paise"].notna().to_numpy()
    rec = np.where(rec_provided, m["recovery_paise"].fillna(0).to_numpy(dtype=float), 0.0)
    a6 = bool((counted & ~rec_provided).any())
    total = fwd + rev + pack + hand + wo - rec
    loss = np.where(counted, total, np.nan)
    br = {
        k: float(np.where(counted, v, 0.0).sum())
        for k, v in (
            ("forward_ship_paise", fwd),
            ("reverse_ship_paise", rev),
            ("packaging_paise", pack),
            ("handling_paise", hand),
            ("writeoff_paise", wo),
            ("cost_recovery_paise", rec),
        )
    }
    assumed = [
        n
        for n, flag in (
            ("forward_ship", a1),
            ("reverse_ship", a2),
            ("packaging", a3),
            ("handling", a4),
            ("writeoff", a5),
            ("cost_recovery", a6),
        )
        if flag
    ]
    return loss, br, assumed


def summarize(m: pd.DataFrame, cost: CostDefaults) -> LossSummary:
    """Observed operating loss on accepted returns, plus RTO loss reported separately."""
    loss, br, assumed = return_losses(m, cost)
    counted = ~np.isnan(loss)
    rto = (m["delivery_status"] == "RTO").to_numpy()
    fwd = np.where(m["fwd_paise"].notna(), m["fwd_paise"].fillna(0), cost.forward_ship_paise)
    pack = np.where(m["pack_paise"].notna(), m["pack_paise"].fillna(0), cost.packaging_paise)
    rto_loss = float(np.where(rto, fwd + cost.reverse_ship_paise + pack, 0.0).sum())
    return LossSummary(
        observed_paise=int(round(np.nansum(loss))),
        breakdown=LossBreakdown(**{k: int(round(v)) for k, v in br.items()}),
        assumed_fields=assumed,
        rto_observed_paise=int(round(rto_loss)),
        returns_counted=int(counted.sum()),
    )


def estimate(m: pd.DataFrame, flagged: np.ndarray, s: Settings) -> LossEstimate:
    """Scenario: eligible loss x effectiveness - intervention cost - friction margin loss."""
    cost = s.cost_defaults_paise
    loss, _, assumed = return_losses(m, cost)
    observed = summarize(m, cost)
    flagged = flagged & True
    pool = float(np.nansum(np.where(flagged, loss, np.nan))) if flagged.any() else 0.0
    n_flagged = int(flagged.sum())
    intervention = n_flagged * cost.intervention_cost_paise
    friction = int(
        round(m["amount_paise"].to_numpy()[flagged].sum() * cost.friction_margin_loss_pct / 100.0)
    )
    eff = s.effectiveness_assumption
    orders = max(len(m), 1)

    def net(e: float) -> int:
        return int(round(pool * e - intervention - friction))

    nets = ScenarioAmounts(low=net(eff.low), base=net(eff.base), high=net(eff.high))
    per100 = ScenarioAmounts(
        low=round(nets.low / orders * 100),
        base=round(nets.base / orders * 100),
        high=round(nets.high / orders * 100),
    )
    return LossEstimate(
        observed=observed,
        eligible_loss_paise=int(round(pool)),
        flagged_eligible_returns=n_flagged,
        intervention_cost_paise=int(intervention),
        friction_margin_loss_paise=friction,
        net_avoidable=nets,
        loss_avoided_per_100_paise=per100,
        effectiveness=eff,
        assumed_fields=sorted(
            {*assumed, "intervention_cost", "friction_margin_loss", "effectiveness"}
        ),
    )
