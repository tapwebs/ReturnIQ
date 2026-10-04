"""`Engine` protocol (0.8) and its implementation. No I/O, no HTTP, no DB."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd
from returniq_contracts import (
    ActionType,
    Analytics,
    Filters,
    LossEstimate,
    Outcome,
    PolicySimulation,
    Provenance,
    Readiness,
    ReturnDNA,
    RiskAssessment,
    ScorePreviewRequest,
    ScorePreviewResponse,
    Settings,
    Stage,
    Timeline,
)

from . import analytics as an
from . import assess as asmt
from . import policy as pol
from .actions import recommend as _recommend
from .generator import GeneratedDataset, generate_dataset
from .prepare import Dataset, build_master
from .settings import ENGINE_VERSION

if TYPE_CHECKING:
    from .ml import TrainedModel


class Engine(Protocol):
    """Contract between the risk engine and the backend (Part 0.8)."""

    version: str

    def generate_dataset(self, n_orders: int, seed: int) -> GeneratedDataset: ...
    def readiness(self, ds: Dataset, s: Settings) -> Readiness: ...
    def analytics(self, ds: Dataset, f: Filters, s: Settings) -> Analytics: ...
    def timeline(self, ds: Dataset, f: Filters, s: Settings) -> Timeline: ...
    def assess(
        self,
        ds: Dataset,
        order_id: str,
        stage: Stage,
        as_of: datetime,
        s: Settings,
        model: TrainedModel | None,
    ) -> RiskAssessment: ...
    def assess_batch(
        self, ds: Dataset, stage: Stage, s: Settings, model: TrainedModel | None
    ) -> list[RiskAssessment]: ...
    def preview(
        self,
        req: ScorePreviewRequest,
        ds: Dataset | None,
        s: Settings,
        models: dict[Stage, TrainedModel],
    ) -> ScorePreviewResponse: ...
    def customer_dna(self, ds: Dataset, customer_id: str) -> ReturnDNA: ...
    def recommend(self, a: RiskAssessment, s: Settings) -> list[ActionType]: ...
    def estimate_loss(self, ds: Dataset, f: Filters, s: Settings) -> LossEstimate: ...
    def simulate_policy(
        self, ds: Dataset, stage: Stage, top_pct: float, s: Settings, model: TrainedModel | None
    ) -> PolicySimulation: ...
    def train(self, ds: Dataset, stage: Stage, outcomes: list[Outcome] | None) -> TrainedModel: ...
    def load_model(self, artifact: bytes) -> TrainedModel: ...


def _preview_frames(req: ScorePreviewRequest) -> pd.DataFrame:
    """Build a tiny master frame from a preview request (history + order + return)."""
    o = req.order
    orders, ships, rets = [], [], []

    def iso(dt: datetime | None) -> str | None:
        return None if dt is None else dt.isoformat()

    for i, h in enumerate(req.history):
        oid = f"ord_preview_h{i + 1}"
        orders.append(
            {
                "order_id": oid,
                "customer_id": o.customer_id,
                "order_date": iso(h.order_date),
                "sku_id": f"sku_preview_h{i + 1}",
                "category": None if h.category is None else h.category.value,
                "quantity": 1,
                "payment_mode": h.payment_mode.value,
                "order_amount_inr": h.order_amount_paise / 100.0,
                "pincode": o.pincode,
            }
        )
        ships.append(
            {
                "order_id": oid,
                "shipped_at": iso(h.order_date + pd.Timedelta(hours=6)),
                "delivered_at": iso(h.delivered_at),
                "delivery_status": h.delivery_status.value,
                "delivery_attempts": 1,
            }
        )
        if h.return_requested_at is not None and h.return_reason is not None:
            rets.append(
                {
                    "return_id": f"ret_preview_h{i + 1}",
                    "order_id": oid,
                    "return_requested_at": iso(h.return_requested_at),
                    "return_accepted_at": None,
                    "return_reason": h.return_reason.value,
                    "return_status": "REQUESTED",
                }
            )
    orders.append(
        {
            "order_id": o.order_id,
            "customer_id": o.customer_id,
            "order_date": iso(o.order_date),
            "sku_id": o.sku_id,
            "category": None if o.category is None else o.category.value,
            "quantity": o.quantity,
            "payment_mode": o.payment_mode.value,
            "order_amount_inr": o.order_amount_paise / 100.0,
            "pincode": o.pincode,
        }
    )
    if req.return_request is not None:
        rr = req.return_request
        ships.append(
            {
                "order_id": o.order_id,
                "shipped_at": iso(o.order_date + pd.Timedelta(hours=6)),
                "delivered_at": iso(rr.delivered_at),
                "delivery_status": "DELIVERED",
                "delivery_attempts": 1,
            }
        )
        rets.append(
            {
                "return_id": "ret_preview",
                "order_id": o.order_id,
                "return_requested_at": iso(rr.return_requested_at),
                "return_accepted_at": None,
                "return_reason": rr.return_reason.value,
                "return_status": "REQUESTED",
            }
        )
    return build_master(pd.DataFrame(orders), pd.DataFrame(ships), pd.DataFrame(rets))


class ReturnIQEngine:
    """Default engine: rules always, LightGBM blend when a trained model is passed."""

    version = ENGINE_VERSION

    def generate_dataset(self, n_orders: int, seed: int) -> GeneratedDataset:
        return generate_dataset(n_orders, seed)

    def readiness(self, ds: Dataset, s: Settings) -> Readiness:
        return an.readiness(ds, s)

    def analytics(self, ds: Dataset, f: Filters, s: Settings) -> Analytics:
        return an.analytics(ds, f, s)

    def timeline(self, ds: Dataset, f: Filters, s: Settings) -> Timeline:
        return an.timeline(ds, f, s)

    def assess(
        self,
        ds: Dataset,
        order_id: str,
        stage: Stage,
        as_of: datetime,
        s: Settings,
        model: TrainedModel | None,
    ) -> RiskAssessment:
        return asmt.assess(ds, order_id, stage, as_of, s, model)

    def assess_batch(
        self, ds: Dataset, stage: Stage, s: Settings, model: TrainedModel | None
    ) -> list[RiskAssessment]:
        return asmt.assess_batch(ds, stage, s, model)

    def preview(
        self,
        req: ScorePreviewRequest,
        ds: Dataset | None,
        s: Settings,
        models: dict[Stage, TrainedModel],
    ) -> ScorePreviewResponse:
        pm = _preview_frames(req)
        base = ds.master if ds is not None else pm.iloc[0:0]
        combined = pd.concat([base, pm], ignore_index=True)
        combined = combined.sort_values(["order_ts", "order_id"], kind="stable").reset_index(
            drop=True
        )
        tmp = Dataset(combined, ds.provenance if ds is not None else Provenance.SYNTHETIC)
        oid = req.order.order_id
        pre = asmt.assess(
            tmp, oid, Stage.PRE_DISPATCH, req.order.order_date, s, models.get(Stage.PRE_DISPATCH)
        )
        post = None
        if req.return_request is not None:
            post = asmt.assess(
                tmp,
                oid,
                Stage.POST_DELIVERY,
                req.return_request.return_requested_at,
                s,
                models.get(Stage.POST_DELIVERY),
            )
        return ScorePreviewResponse(pre_dispatch=pre, post_delivery=post)

    def customer_dna(self, ds: Dataset, customer_id: str) -> ReturnDNA:
        return an.customer_dna(ds, customer_id)

    def recommend(self, a: RiskAssessment, s: Settings) -> list[ActionType]:
        return _recommend(a, s)

    def estimate_loss(self, ds: Dataset, f: Filters, s: Settings) -> LossEstimate:
        return an.estimate_loss(ds, f, s)

    def simulate_policy(
        self, ds: Dataset, stage: Stage, top_pct: float, s: Settings, model: TrainedModel | None
    ) -> PolicySimulation:
        return pol.simulate_policy(ds, stage, top_pct, s, model)

    def train(self, ds: Dataset, stage: Stage, outcomes: list[Outcome] | None) -> TrainedModel:
        from . import ml

        return ml.train(ds, stage, outcomes)

    def load_model(self, artifact: bytes) -> TrainedModel:
        from . import ml

        return ml.load_model(artifact)


__all__ = ["Engine", "ReturnIQEngine", "np"]
