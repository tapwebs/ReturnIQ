"""Contract views over a Dataset (orders, order detail). Pure conversions, no I/O."""

from __future__ import annotations

import pandas as pd
from returniq_contracts import (
    Category,
    DeliveryStatus,
    Order,
    OrderDetail,
    PaymentMode,
    QcResult,
    ReturnReason,
    ReturnRecord,
    ReturnStatus,
    RiskAssessment,
    ShipmentRecord,
    TimelineEvent,
)

from .analytics import ttr_bucket
from .prepare import Dataset


def _opt(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NaT else v


def order_from_row(row: pd.Series, latest: RiskAssessment | None = None) -> Order:
    """Build the contract ``Order`` for one master row, with the latest risk if supplied."""
    cat = _opt(row["category"])
    status = _opt(row["delivery_status"])
    rstatus = _opt(row["return_status"])
    return Order(
        order_id=row["order_id"],
        customer_id=row["customer_id"],
        order_date=row["order_ts"].to_pydatetime(),
        sku_id=row["sku_id"],
        category=None if cat is None else Category(cat),
        quantity=int(row["quantity"]),
        payment_mode=PaymentMode(row["payment_mode"]),
        order_amount_paise=int(row["amount_paise"]),
        pincode=_opt(row["pincode"]),
        delivery_status=None if status is None else DeliveryStatus(status),
        return_status=None if rstatus is None else ReturnStatus(rstatus),
        risk_score=None if latest is None else latest.risk_score,
        risk_level=None if latest is None else latest.risk_level,
    )


def order_detail(ds: Dataset, order_id: str, assessments: list[RiskAssessment]) -> OrderDetail:
    """Order + shipment + return + prior orders + timeline events + assessments."""
    m = ds.master
    row = m.iloc[ds.order_pos[order_id]]
    latest = assessments[-1] if assessments else None
    shipment = None
    if _opt(row["delivery_status"]) is not None:
        shipment = ShipmentRecord(
            order_id=order_id,
            shipped_at=_opt(row["shipped_ts"]) and row["shipped_ts"].to_pydatetime(),
            delivered_at=None
            if pd.isna(row["delivered_ts"])
            else row["delivered_ts"].to_pydatetime(),
            delivery_status=DeliveryStatus(row["delivery_status"]),
            delivery_attempts=None
            if pd.isna(row["delivery_attempts"])
            else int(row["delivery_attempts"]),
        )
    ret = None
    events = [
        TimelineEvent(
            at=row["order_ts"].to_pydatetime(),
            kind="ORDER_PLACED",
            detail=f"{row['payment_mode']} order placed",
        )
    ]
    if shipment is not None and shipment.shipped_at is not None:
        events.append(TimelineEvent(at=shipment.shipped_at, kind="SHIPPED", detail="Dispatched"))
    if shipment is not None and shipment.delivered_at is not None:
        events.append(TimelineEvent(at=shipment.delivered_at, kind="DELIVERED", detail="Delivered"))
    if pd.notna(row["ret_ts"]):
        ttr = None if pd.isna(row["ttr_h"]) else float(row["ttr_h"])
        qc = _opt(row["qc_result"])
        ret = ReturnRecord(
            return_id=row["return_id"],
            order_id=order_id,
            return_requested_at=row["ret_ts"].to_pydatetime(),
            return_accepted_at=None
            if pd.isna(row["ret_acc_ts"])
            else row["ret_acc_ts"].to_pydatetime(),
            return_reason=ReturnReason(row["return_reason"]),
            return_status=ReturnStatus(row["return_status"]),
            qc_result=None if qc is None else QcResult(qc),
            ttr_h=ttr,
            ttr_bucket=ttr_bucket(ttr),
        )
        events.append(
            TimelineEvent(
                at=ret.return_requested_at,
                kind="RETURN_REQUESTED",
                detail=f"Reason: {ret.return_reason.value}",
            )
        )
    events.sort(key=lambda e: (e.at, e.kind))
    prior = m[(m["customer_id"] == row["customer_id"]) & (m["order_ts"] < row["order_ts"])]
    return OrderDetail(
        order=order_from_row(row, latest),
        shipment=shipment,
        return_=ret,
        prior_orders=[order_from_row(r) for _, r in prior.iterrows()],
        events=events,
        assessments=assessments,
    )
