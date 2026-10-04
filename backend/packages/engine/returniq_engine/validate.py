"""0.4 CSV contract validator. Pure functions over DataFrames (no I/O)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

ORDER_REQUIRED = (
    "order_id",
    "customer_id",
    "order_date",
    "sku_id",
    "quantity",
    "payment_mode",
    "order_amount_inr",
)
ORDER_COLUMNS = (*ORDER_REQUIRED, "category", "pincode", "customer_contact")
SHIPMENT_COLUMNS = (
    "order_id",
    "shipped_at",
    "delivered_at",
    "delivery_status",
    "delivery_attempts",
    "forward_shipping_inr",
    "packaging_inr",
)
RETURN_COLUMNS = (
    "return_id",
    "order_id",
    "return_requested_at",
    "return_accepted_at",
    "return_reason",
    "return_status",
    "qc_result",
    "refund_amount_inr",
    "reverse_shipping_inr",
    "handling_inr",
    "writeoff_inr",
    "cost_recovery_inr",
)
PAYMENT = {"COD", "PREPAID", "PARTIAL_COD"}
CATEGORY = {"FASHION", "BEAUTY", "ELECTRONICS", "LIFESTYLE", "OTHER"}
DELIVERY = {"DELIVERED", "RTO", "IN_TRANSIT", "NDR", "LOST"}
REASON = {
    "SIZE_FIT",
    "CHANGED_MIND",
    "NOT_AS_DESCRIBED",
    "DEFECTIVE",
    "DAMAGED_IN_TRANSIT",
    "WRONG_ITEM",
    "OTHER",
}
RSTATUS = {"REQUESTED", "ACCEPTED", "REJECTED", "REFUNDED"}
QC = {"UNUSED", "USED", "DAMAGED_BY_CUSTOMER", "UNKNOWN"}
_TZ = r"(?:Z|[+-]\d{2}:?\d{2})$"


@dataclass(frozen=True)
class RejectedRow:
    """A quarantined row with a machine-readable reason code."""

    file: str
    row: int
    key: str
    code: str
    message: str


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype("string").str.strip() == "")


def _collect(
    out: list[RejectedRow], file: str, mask: pd.Series, keys: pd.Series, code: str, msg: str
) -> None:
    for idx in mask[mask].index:
        out.append(RejectedRow(file, int(idx), str(keys.loc[idx]), code, msg))


def _bad_time(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (unparseable, ambiguous-no-timezone) masks for non-blank values."""
    nb = ~_blank(s)
    txt = s.astype("string")
    ambiguous = nb & ~txt.str.contains(_TZ, regex=True, na=False)
    parsed = pd.to_datetime(s, utc=True, errors="coerce", format="ISO8601")
    unparse = nb & parsed.isna() & ~ambiguous
    return unparse, ambiguous


def validate_csv_frames(
    orders: pd.DataFrame, shipments: pd.DataFrame | None = None, returns: pd.DataFrame | None = None
) -> list[RejectedRow]:
    """Validate the three CSV frames. Identical duplicate rows are not rejected (de-duplicated)."""
    out: list[RejectedRow] = []
    missing = [c for c in ORDER_REQUIRED if c not in orders.columns]
    for col in missing:
        out.append(RejectedRow("orders", -1, col, "MISSING_COLUMN", f"expected column {col}"))
    if missing:
        return out
    o = orders.reset_index(drop=True)
    keys = o["order_id"].astype("string").fillna("")
    for col in ORDER_REQUIRED:
        _collect(out, "orders", _blank(o[col]), keys, "MISSING_VALUE", f"{col} is required")
    pm = o["payment_mode"].astype("string")
    _collect(
        out,
        "orders",
        ~_blank(pm) & ~pm.isin(PAYMENT),
        keys,
        "INVALID_ENUM",
        "payment_mode must be COD, PREPAID or PARTIAL_COD",
    )
    if "category" in o:
        cat = o["category"].astype("string")
        _collect(
            out,
            "orders",
            ~_blank(cat) & ~cat.isin(CATEGORY),
            keys,
            "INVALID_ENUM",
            "category not recognised",
        )
    qty = pd.to_numeric(o["quantity"], errors="coerce")
    _collect(
        out,
        "orders",
        ~_blank(o["quantity"]) & ~(qty >= 1),
        keys,
        "INVALID_QUANTITY",
        "quantity must be >= 1",
    )
    amt = pd.to_numeric(o["order_amount_inr"], errors="coerce")
    _collect(
        out,
        "orders",
        ~_blank(o["order_amount_inr"]) & ~(amt >= 0),
        keys,
        "INVALID_MONEY",
        "order_amount_inr must be a non-negative number",
    )
    if "pincode" in o:
        pin = o["pincode"].astype("string")
        _collect(
            out,
            "orders",
            ~_blank(pin) & ~pin.str.fullmatch(r"\d{6}").fillna(False),
            keys,
            "INVALID_PINCODE",
            "pincode must be 6 digits",
        )
    bad, amb = _bad_time(o["order_date"])
    _collect(out, "orders", bad, keys, "INVALID_DATE", "order_date is not ISO-8601")
    _collect(out, "orders", amb, keys, "AMBIGUOUS_DATE", "order_date needs a timezone offset")
    dup_id = o.duplicated("order_id", keep="first")
    dup_full = o.duplicated(keep="first")
    _collect(
        out,
        "orders",
        dup_id & ~dup_full,
        keys,
        "DUPLICATE_CONFLICT",
        "duplicate order_id with conflicting values",
    )
    rejected_orders = {r.key for r in out if r.file == "orders"}
    valid_ids = set(keys[~keys.isin(rejected_orders)])
    if shipments is not None and len(shipments):
        _validate_shipments(out, shipments.reset_index(drop=True), valid_ids)
    if returns is not None and len(returns):
        _validate_returns(out, returns.reset_index(drop=True), valid_ids, orders, shipments)
    return out


def _validate_shipments(out: list[RejectedRow], s: pd.DataFrame, valid_ids: set[str]) -> None:
    keys = s["order_id"].astype("string").fillna("")
    _collect(
        out, "shipments", ~keys.isin(valid_ids), keys, "ORPHAN_ROW", "order_id not found in orders"
    )
    st = s["delivery_status"].astype("string")
    _collect(out, "shipments", ~st.isin(DELIVERY), keys, "INVALID_ENUM", "delivery_status invalid")
    for col in ("shipped_at", "delivered_at"):
        bad, amb = _bad_time(s[col])
        _collect(out, "shipments", bad, keys, "INVALID_DATE", f"{col} is not ISO-8601")
        _collect(out, "shipments", amb, keys, "AMBIGUOUS_DATE", f"{col} needs a timezone")
    sh = pd.to_datetime(s["shipped_at"], utc=True, errors="coerce", format="ISO8601")
    dl = pd.to_datetime(s["delivered_at"], utc=True, errors="coerce", format="ISO8601")
    _collect(
        out,
        "shipments",
        (dl < sh).fillna(False),
        keys,
        "NEGATIVE_DURATION",
        "delivered_at before shipped_at",
    )
    _collect(
        out,
        "shipments",
        (st == "IN_TRANSIT") & dl.notna(),
        keys,
        "IMPOSSIBLE_STATE",
        "IN_TRANSIT shipment has delivered_at",
    )
    dup_id = s.duplicated("order_id", keep="first")
    _collect(
        out,
        "shipments",
        dup_id & ~s.duplicated(keep="first"),
        keys,
        "DUPLICATE_CONFLICT",
        "duplicate shipment for order",
    )


def _validate_returns(
    out: list[RejectedRow],
    r: pd.DataFrame,
    valid_ids: set[str],
    orders: pd.DataFrame,
    shipments: pd.DataFrame | None,
) -> None:
    keys = r["return_id"].astype("string").fillna("")
    oid = r["order_id"].astype("string")
    _collect(
        out, "returns", ~oid.isin(valid_ids), keys, "ORPHAN_ROW", "order_id not found in orders"
    )
    _collect(
        out,
        "returns",
        ~r["return_reason"].astype("string").isin(REASON),
        keys,
        "INVALID_ENUM",
        "return_reason invalid",
    )
    _collect(
        out,
        "returns",
        ~r["return_status"].astype("string").isin(RSTATUS),
        keys,
        "INVALID_ENUM",
        "return_status invalid",
    )
    if "qc_result" in r:
        qc = r["qc_result"].astype("string")
        _collect(
            out, "returns", ~_blank(qc) & ~qc.isin(QC), keys, "INVALID_ENUM", "qc_result invalid"
        )
    for col in ("return_requested_at", "return_accepted_at"):
        bad, amb = _bad_time(r[col])
        _collect(out, "returns", bad, keys, "INVALID_DATE", f"{col} is not ISO-8601")
        _collect(out, "returns", amb, keys, "AMBIGUOUS_DATE", f"{col} needs a timezone")
    for col in (
        "refund_amount_inr",
        "reverse_shipping_inr",
        "handling_inr",
        "writeoff_inr",
        "cost_recovery_inr",
    ):
        if col in r:
            v = pd.to_numeric(r[col], errors="coerce")
            _collect(
                out,
                "returns",
                ~_blank(r[col]) & ~(v >= 0),
                keys,
                "INVALID_MONEY",
                f"{col} must be a non-negative number",
            )
    _collect(
        out,
        "returns",
        r.duplicated("return_id", keep="first") & ~r.duplicated(keep="first"),
        keys,
        "DUPLICATE_CONFLICT",
        "duplicate return_id",
    )
    _collect(
        out,
        "returns",
        r.duplicated("order_id", keep="first") & ~r.duplicated("return_id"),
        keys,
        "DUPLICATE_CONFLICT",
        "more than one return for an order",
    )
    req = pd.to_datetime(r["return_requested_at"], utc=True, errors="coerce", format="ISO8601")
    acc = pd.to_datetime(r["return_accepted_at"], utc=True, errors="coerce", format="ISO8601")
    _collect(
        out,
        "returns",
        (acc < req).fillna(False),
        keys,
        "NEGATIVE_DURATION",
        "return_accepted_at before return_requested_at",
    )
    od = orders.assign(
        _od=pd.to_datetime(orders["order_date"], utc=True, errors="coerce", format="ISO8601")
    ).drop_duplicates("order_id")
    od_map = od.set_index("order_id")["_od"]
    _collect(
        out,
        "returns",
        (req < oid.map(od_map)).fillna(False),
        keys,
        "NEGATIVE_DURATION",
        "return requested before the order",
    )
    if shipments is not None and len(shipments) and "delivered_at" in shipments:
        sd = (
            shipments.drop_duplicates("order_id")
            .assign(
                _d=pd.to_datetime(
                    shipments.drop_duplicates("order_id")["delivered_at"],
                    utc=True,
                    errors="coerce",
                    format="ISO8601",
                )
            )
            .set_index("order_id")["_d"]
        )
        _collect(
            out,
            "returns",
            (req < oid.map(sd)).fillna(False),
            keys,
            "NEGATIVE_DURATION",
            "return requested before delivery (negative time-to-return)",
        )
