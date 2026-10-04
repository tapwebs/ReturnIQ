"""Analytics, readiness, timeline and customer DNA. RTO and returns are never merged."""

from __future__ import annotations

import numpy as np
import pandas as pd
from returniq_contracts import (
    Analytics,
    CohortRow,
    Filters,
    LossEstimate,
    Readiness,
    ReadinessMode,
    ReasonRow,
    ReturnDNA,
    ReturnReason,
    ReturnsMetric,
    RtoMetric,
    ScenarioAmounts,
    ScoringMethod,
    SegmentRow,
    Settings,
    Timeline,
    TtrBucket,
    TtrBucketCount,
    TtrSummary,
    WaterfallStep,
)

from . import loss as loss_mod
from .features import FAST_TTR_H
from .prepare import Dataset
from .settings import ELIGIBLE_EXCLUDED_REASONS

MIN_DAYS_OBSERVED = 30
MIN_COMPLETED_WINDOWS = 100
MIN_TIMESTAMP_COMPLETENESS = 0.8
TOP_N = 5
BUCKET_ORDER = (
    TtrBucket.LT_6H,
    TtrBucket.H6_24,
    TtrBucket.D1_3,
    TtrBucket.D3_7,
    TtrBucket.GT_7D,
    TtrBucket.UNKNOWN,
)


def ttr_bucket(hours: float | None) -> TtrBucket:
    """Bucket a time-to-return in hours; missing is UNKNOWN (never 0)."""
    if hours is None or np.isnan(hours):
        return TtrBucket.UNKNOWN
    if hours < 6:
        return TtrBucket.LT_6H
    if hours < 24:
        return TtrBucket.H6_24
    if hours < 72:
        return TtrBucket.D1_3
    if hours <= 168:
        return TtrBucket.D3_7
    return TtrBucket.GT_7D


def ttr_bucket_counts(ttr_h: pd.Series) -> list[TtrBucketCount]:
    """Vectorised bucket counts, every bucket present."""
    v = ttr_h.to_numpy(dtype=float)
    names = np.select(
        [np.isnan(v), v < 6, v < 24, v < 72, v <= 168],
        ["UNKNOWN", "LT_6H", "H6_24", "D1_3", "D3_7"],
        default="GT_7D",
    )
    return [TtrBucketCount(bucket=b, count=int((names == b.value).sum())) for b in BUCKET_ORDER]


def filter_mask(ds: Dataset, f: Filters) -> np.ndarray:
    m = ds.master
    mask = np.ones(len(m), dtype=bool)
    if f.from_ is not None:
        mask &= (m["order_ts"] >= pd.Timestamp(f.from_).tz_convert("UTC")).to_numpy()
    if f.to is not None:
        mask &= (m["order_ts"] <= pd.Timestamp(f.to).tz_convert("UTC")).to_numpy()
    if f.payment_mode is not None:
        mask &= (m["payment_mode"] == f.payment_mode.value).to_numpy()
    return mask


def flagged_mask(ds: Dataset, s: Settings) -> np.ndarray:
    """Rules-only Stage 2 flag per master row: rule score >= MEDIUM and reason-eligible."""
    p = ds.prepare()
    out = np.zeros(len(ds.master), dtype=bool)
    tot = p.rules2["total"].to_numpy()
    out[p.s2_pos] = (tot >= s.thresholds.medium) & (p.s2["reason_eligible"].to_numpy() > 0)
    return out


def readiness(ds: Dataset, s: Settings) -> Readiness:
    """OBSERVATION until enough days, completed return windows and timestamp completeness."""
    m = ds.master
    if m.empty:
        return Readiness(
            mode=ReadinessMode.OBSERVATION,
            days_observed=0.0,
            orders=0,
            completed_return_windows=0,
            timestamp_completeness=0.0,
            ready=False,
            reason="No orders imported yet.",
        )
    end = max(
        m["order_ts"].max(),
        m["ret_ts"].max() if m["ret_ts"].notna().any() else m["order_ts"].max(),
        m["delivered_ts"].max() if m["delivered_ts"].notna().any() else m["order_ts"].max(),
    )
    days = float((m["order_ts"].max() - m["order_ts"].min()).total_seconds() / 86400.0)
    delivered = m["delivery_status"] == "DELIVERED"
    with_ts = delivered & m["delivered_ts"].notna()
    windows = int(
        (with_ts & (m["delivered_ts"] + pd.Timedelta(days=s.return_window_days) <= end)).sum()
    )
    completeness = float(with_ts.sum() / delivered.sum()) if delivered.any() else 0.0
    reasons = []
    if days < MIN_DAYS_OBSERVED:
        reasons.append(f"only {days:.0f} days observed (need {MIN_DAYS_OBSERVED})")
    if windows < MIN_COMPLETED_WINDOWS:
        reasons.append(f"{windows} completed return windows (need {MIN_COMPLETED_WINDOWS})")
    if completeness < MIN_TIMESTAMP_COMPLETENESS:
        reasons.append(
            f"delivery timestamp completeness {completeness:.0%} "
            f"(need {MIN_TIMESTAMP_COMPLETENESS:.0%})"
        )
    ready = not reasons
    return Readiness(
        mode=ReadinessMode.ACTIVE if ready else ReadinessMode.OBSERVATION,
        days_observed=round(days, 1),
        orders=len(m),
        completed_return_windows=windows,
        timestamp_completeness=round(completeness, 4),
        ready=ready,
        reason="Enough history to act." if ready else "; ".join(reasons),
    )


def _rate(a: int, b: int) -> float | None:
    return round(a / b * 100.0, 2) if b else None


def _segments(
    m: pd.DataFrame, key: pd.Series, loss: np.ndarray, min_orders: int = 1
) -> list[SegmentRow]:
    g = pd.DataFrame(
        {
            "key": key.to_numpy(),
            "ret": m["ret_ts"].notna().to_numpy(),
            "rto": (m["delivery_status"] == "RTO").to_numpy(),
            "loss": np.nan_to_num(loss),
        }
    )
    agg = g.groupby("key", dropna=True).agg(
        orders=("ret", "size"), returns=("ret", "sum"), rto=("rto", "sum"), loss=("loss", "sum")
    )
    agg = agg[agg["orders"] >= min_orders]
    agg["score"] = agg["returns"] + agg["rto"]
    agg = agg.sort_values(["score", "orders"], ascending=False, kind="stable").head(TOP_N)
    return [
        SegmentRow(
            key=str(k),
            orders=int(r.orders),
            returns=int(r.returns),
            rto=int(r.rto),
            return_rate=_rate(int(r.returns), int(r.orders)),
            rto_rate=_rate(int(r.rto), int(r.orders)),
            loss_paise=int(r.loss),
        )
        for k, r in agg.iterrows()
    ]


def analytics(ds: Dataset, f: Filters, s: Settings) -> Analytics:
    """Core analytics with explicit denominators (rules-only flags)."""
    mask = filter_mask(ds, f)
    m = ds.master[mask]
    flagged = flagged_mask(ds, s)[mask]
    status = m["delivery_status"]
    dispatched = int(m["shipped_ts"].notna().sum())
    resolved = int(status.isin(["DELIVERED", "RTO", "NDR", "LOST"]).sum())
    delivered = int((status == "DELIVERED").sum())
    rto = int((status == "RTO").sum())
    requested = int(m["ret_ts"].notna().sum())
    accepted = int(m["return_status"].isin(loss_mod.COUNTED_STATUSES).sum())
    est = loss_mod.estimate(m, flagged, s)
    loss_arr, _, _ = loss_mod.return_losses(m, s.cost_defaults_paise)
    ret_rows = m[m["ret_ts"].notna()]
    reasons = ret_rows["return_reason"].value_counts()
    reason_rows = [
        ReasonRow(
            reason=ReturnReason(r),
            count=int(c),
            share=round(c / requested, 4),
            preventable_eligible=r not in ELIGIBLE_EXCLUDED_REASONS,
        )
        for r, c in reasons.items()
    ]
    per_cust = ret_rows.groupby("customer_id").size()
    seg_key = m["category"].fillna("UNKNOWN").astype(str) + " / " + m["payment_mode"].astype(str)
    period_from = m["order_ts"].min() if len(m) else None
    period_to = m["order_ts"].max() if len(m) else None
    return Analytics(
        period_from=period_from.to_pydatetime() if period_from is not None else None,
        period_to=period_to.to_pydatetime() if period_to is not None else None,
        orders=len(m),
        dispatched=dispatched,
        delivered=delivered,
        rto=RtoMetric(
            count=rto,
            rate=_rate(rto, resolved),
            denominator=resolved,
            pending=int((status == "IN_TRANSIT").sum()),
        ),
        returns=ReturnsMetric(
            requested=requested,
            accepted=accepted,
            rate=_rate(accepted, delivered),
            denominator=delivered,
            requested_rate=_rate(requested, delivered),
        ),
        flagged_returns=int(flagged.sum()),
        estimated_prr=_rate(int(flagged.sum()), delivered),
        confirmed_prr=None,
        loss=est.observed,
        est_net_avoidable=est.net_avoidable,
        loss_avoided_per_100_paise=est.loss_avoided_per_100_paise.base,
        ttr=TtrSummary(
            median_h=(
                None
                if ret_rows["ttr_h"].dropna().empty
                else round(float(ret_rows["ttr_h"].median()), 2)
            ),
            buckets=ttr_bucket_counts(ret_rows["ttr_h"]),
        ),
        top_segments=_segments(m, seg_key, loss_arr),
        top_pincodes=_segments(m, m["pincode"], loss_arr, min_orders=5),
        top_skus=_segments(m, m["sku_id"], loss_arr, min_orders=5),
        reason_breakdown=reason_rows,
        repeat_returners=int((per_cust >= 2).sum()),
        readiness=readiness(ds, s),
        scoring_method=ScoringMethod.RULES,
    )


def estimate_loss(ds: Dataset, f: Filters, s: Settings) -> LossEstimate:
    mask = filter_mask(ds, f)
    return loss_mod.estimate(ds.master[mask], flagged_mask(ds, s)[mask], s)


def timeline(ds: Dataset, f: Filters, s: Settings) -> Timeline:
    """TTR buckets, flagged-vs-unflagged cohort compare and the loss waterfall."""
    mask = filter_mask(ds, f)
    m = ds.master[mask]
    flagged = flagged_mask(ds, s)[mask]
    delivered = int((m["delivery_status"] == "DELIVERED").sum())
    loss_arr, br, assumed = loss_mod.return_losses(m, s.cost_defaults_paise)
    has_ret = m["ret_ts"].notna().to_numpy()
    rows = []
    for name, sel in (("FLAGGED", has_ret & flagged), ("NOT_FLAGGED", has_ret & ~flagged)):
        sub = m[sel]
        med = sub["ttr_h"].dropna()
        rows.append(
            CohortRow(
                cohort=name,
                returns=int(sel.sum()),
                median_ttr_h=None if med.empty else round(float(med.median()), 2),
                return_rate=_rate(int(sel.sum()), delivered),
                loss_paise=int(np.nansum(loss_arr[sel])),
            )
        )
    labels = (
        ("Forward shipping", "forward_ship_paise", "forward_ship"),
        ("Reverse shipping", "reverse_ship_paise", "reverse_ship"),
        ("Packaging", "packaging_paise", "packaging"),
        ("Handling", "handling_paise", "handling"),
        ("Write-off", "writeoff_paise", "writeoff"),
        ("Cost recovery", "cost_recovery_paise", "cost_recovery"),
    )
    steps = [
        WaterfallStep(
            label=lab,
            amount_paise=int(round(br[k])) * (-1 if k.startswith("cost_rec") else 1),
            assumed=a in assumed,
        )
        for lab, k, a in labels
    ]
    total = sum(st.amount_paise for st in steps)
    steps.append(WaterfallStep(label="Observed operating loss", amount_paise=total))
    ret = m[has_ret]
    med_all = ret["ttr_h"].dropna()
    return Timeline(
        ttr_buckets=ttr_bucket_counts(ret["ttr_h"]),
        cohort_compare=rows,
        loss_waterfall=steps,
        median_ttr_h=None if med_all.empty else round(float(med_all.median()), 2),
    )


def customer_dna(ds: Dataset, customer_id: str, s: Settings | None = None) -> ReturnDNA:
    """Summary of a customer's full history in the dataset (as of the last event)."""
    m = ds.master[ds.master["customer_id"] == customer_id]
    if m.empty:
        raise KeyError(customer_id)
    rets = m[m["ret_ts"].notna()]
    elig = rets[rets["reason_eligible"]]
    ttr = elig["ttr_h"].dropna()
    fast = float((ttr < FAST_TTR_H).mean()) if len(ttr) else None
    p = ds.prepare()
    flagged = int(p.flagged_s2[np.isin(p.s2_pos, m.index.to_numpy())].sum())
    delivered = int((m["delivery_status"] == "DELIVERED").sum())
    rate = round(len(elig) / len(m), 4) if len(m) else None
    cats = {str(k): int(v) for k, v in m["category"].fillna("UNKNOWN").value_counts().items()}
    summary = (
        f"{len(m)} orders, {len(rets)} returns ({len(elig)} preventable-type), "
        f"{int((m['delivery_status'] == 'RTO').sum())} returned-to-origin."
    )
    return ReturnDNA(
        customer_id=customer_id,
        as_of=m["order_ts"].max().to_pydatetime(),
        orders=len(m),
        delivered=delivered,
        rto=int((m["delivery_status"] == "RTO").sum()),
        failed_deliveries=int(m["delivery_status"].isin(["RTO", "NDR", "LOST"]).sum()),
        returns=len(rets),
        eligible_returns=len(elig),
        return_rate=rate,
        median_ttr_h=None if ttr.empty else round(float(ttr.median()), 2),
        fast_return_share=fast,
        flagged_returns=flagged,
        categories=cats,
        summary=summary,
    )


__all__ = [
    "ScenarioAmounts",
    "analytics",
    "customer_dna",
    "estimate_loss",
    "filter_mask",
    "flagged_mask",
    "readiness",
    "timeline",
    "ttr_bucket",
    "ttr_bucket_counts",
]
