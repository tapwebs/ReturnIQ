"""`Dataset`: typed frames (paise, UTC) plus the once-computed point-in-time feature frames."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from returniq_contracts import Provenance

from .features import RTO_KNOWN_AFTER_H, Prepared, build_stage_frames
from .generator import GeneratedDataset
from .rules import rule_fn
from .settings import ELIGIBLE_EXCLUDED_REASONS
from .validate import RejectedRow, validate_csv_frames

DEFAULT_FLAG_THRESHOLD = 40
FAIL_STATUSES = ("RTO", "NDR", "LOST")


def _ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce", format="ISO8601")


def _paise(s: pd.Series) -> pd.Series:
    return (pd.to_numeric(s, errors="coerce") * 100).round()


def build_master(
    orders: pd.DataFrame, shipments: pd.DataFrame | None, returns: pd.DataFrame | None
) -> pd.DataFrame:
    """One row per order: orders LEFT JOIN shipments LEFT JOIN returns, canonical columns."""
    o = orders.drop_duplicates("order_id", keep="first")
    m = pd.DataFrame(
        {
            "order_id": o["order_id"].astype(str),
            "customer_id": o["customer_id"].astype(str),
            "order_ts": _ts(o["order_date"]),
            "sku_id": o["sku_id"].astype(str),
            "category": o.get("category", np.nan),
            "quantity": pd.to_numeric(o["quantity"]).astype("int64"),
            "payment_mode": o["payment_mode"].astype(str),
            "amount_paise": _paise(o["order_amount_inr"]).astype("int64"),
            "pincode": o.get("pincode", np.nan),
        }
    ).reset_index(drop=True)
    m["category"] = m["category"].where(m["category"].notna(), np.nan).astype(object)
    m["pincode"] = m["pincode"].where(m["pincode"].notna(), np.nan).astype(object)
    sh = (
        shipments.drop_duplicates("order_id", keep="first")
        if shipments is not None and len(shipments)
        else pd.DataFrame(columns=["order_id"])
    )
    s = pd.DataFrame(
        {
            "order_id": sh["order_id"].astype(str),
            "shipped_ts": _ts(sh["shipped_at"]) if "shipped_at" in sh else pd.NaT,
            "delivered_ts": _ts(sh["delivered_at"]) if "delivered_at" in sh else pd.NaT,
            "delivery_status": sh.get("delivery_status", np.nan),
            "delivery_attempts": pd.to_numeric(sh["delivery_attempts"], errors="coerce")
            if "delivery_attempts" in sh
            else np.nan,
            "fwd_paise": _paise(sh["forward_shipping_inr"])
            if "forward_shipping_inr" in sh
            else np.nan,
            "pack_paise": _paise(sh["packaging_inr"]) if "packaging_inr" in sh else np.nan,
        }
    )
    rt = (
        returns.drop_duplicates("return_id", keep="first").drop_duplicates("order_id")
        if returns is not None and len(returns)
        else pd.DataFrame(columns=["order_id"])
    )

    def col(name: str) -> pd.Series | float:
        return rt.get(name, np.nan)

    r = pd.DataFrame(
        {
            "order_id": rt["order_id"].astype(str),
            "return_id": col("return_id"),
            "ret_ts": _ts(rt["return_requested_at"]) if "return_requested_at" in rt else pd.NaT,
            "ret_acc_ts": _ts(rt["return_accepted_at"]) if "return_accepted_at" in rt else pd.NaT,
            "return_reason": col("return_reason"),
            "return_status": col("return_status"),
            "qc_result": col("qc_result"),
            "refund_paise": _paise(rt["refund_amount_inr"])
            if "refund_amount_inr" in rt
            else np.nan,
            "rev_paise": _paise(rt["reverse_shipping_inr"])
            if "reverse_shipping_inr" in rt
            else np.nan,
            "handling_paise": _paise(rt["handling_inr"]) if "handling_inr" in rt else np.nan,
            "writeoff_paise": _paise(rt["writeoff_inr"]) if "writeoff_inr" in rt else np.nan,
            "recovery_paise": _paise(rt["cost_recovery_inr"])
            if "cost_recovery_inr" in rt
            else np.nan,
        }
    )
    m = m.merge(s, on="order_id", how="left").merge(r, on="order_id", how="left")
    m = m.sort_values(["order_ts", "order_id"], kind="stable").reset_index(drop=True)
    return derive_columns(m)


def derive_columns(m: pd.DataFrame) -> pd.DataFrame:
    """Derived columns: time-to-return, reason eligibility and when a failed outcome is known."""
    m = m.copy()
    ttr = (m["ret_ts"] - m["delivered_ts"]).dt.total_seconds() / 3600.0
    m["ttr_h"] = ttr.where(ttr >= 0)
    m["reason_eligible"] = m["return_reason"].notna() & ~m["return_reason"].isin(
        ELIGIBLE_EXCLUDED_REASONS
    )
    failed = m["delivery_status"].isin(FAIL_STATUSES)
    base = m["shipped_ts"].fillna(m["order_ts"] + pd.Timedelta(hours=24))
    m["fail_known_ts"] = (base + pd.Timedelta(hours=RTO_KNOWN_AFTER_H)).where(failed)
    return m


@dataclass
class Dataset:
    """Typed canonical frames plus (lazily) the prepared point-in-time features."""

    master: pd.DataFrame
    provenance: Provenance = Provenance.SYNTHETIC
    labels: pd.DataFrame | None = None  # hidden ground truth; evaluation/synthetic training only
    rejected: list[RejectedRow] = field(default_factory=list)
    flag_threshold: int = DEFAULT_FLAG_THRESHOLD
    window_days: int = 7
    _prepared: Prepared | None = field(default=None, repr=False)

    @property
    def dataset_version(self) -> str:
        h = hashlib.sha256()
        cols = [
            "order_id",
            "customer_id",
            "order_ts",
            "amount_paise",
            "delivery_status",
            "delivered_ts",
            "ret_ts",
            "return_reason",
        ]
        h.update(
            pd.util.hash_pandas_object(self.master[cols].astype(str), index=False).values.tobytes()
        )
        return h.hexdigest()[:16]

    @property
    def dataset_id(self) -> str:
        return f"ds_{self.dataset_version[:8]}"

    def prepare(self) -> Prepared:
        """Compute every point-in-time feature once (vectorised); cached afterwards."""
        if self._prepared is None:
            self._prepared = build_stage_frames(
                self.master, rule_fn, self.flag_threshold, self.window_days
            )
        return self._prepared

    @property
    def order_pos(self) -> dict[str, int]:
        if "_order_pos" not in self.__dict__:
            self.__dict__["_order_pos"] = {o: i for i, o in enumerate(self.master["order_id"])}
        return self.__dict__["_order_pos"]

    @classmethod
    def from_csv_frames(
        cls,
        orders: pd.DataFrame,
        shipments: pd.DataFrame | None = None,
        returns: pd.DataFrame | None = None,
        provenance: Provenance = Provenance.SELLER_UPLOADED,
        labels: pd.DataFrame | None = None,
    ) -> Dataset:
        """Validate the CSV-form frames, quarantine bad rows and build a Dataset."""
        rejected = validate_csv_frames(orders, shipments, returns)
        bad_o = {r.key for r in rejected if r.file == "orders"}
        bad_s = {r.key for r in rejected if r.file == "shipments"}
        bad_r = {r.key for r in rejected if r.file == "returns"}
        o = orders[~orders["order_id"].isin(bad_o)]
        s = shipments[~shipments["order_id"].isin(bad_s | bad_o)] if shipments is not None else None
        r = (
            returns[~returns["return_id"].isin(bad_r) & ~returns["order_id"].isin(bad_o)]
            if returns is not None
            else None
        )
        return cls(build_master(o, s, r), provenance, labels, rejected)

    @classmethod
    def from_generated(cls, g: GeneratedDataset) -> Dataset:
        """Dataset from generator output (SYNTHETIC provenance, hidden labels attached)."""
        return cls.from_csv_frames(g.orders, g.shipments, g.returns, Provenance.SYNTHETIC, g.labels)
