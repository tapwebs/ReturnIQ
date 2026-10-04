import pandas as pd

from returniq_engine.export_fixtures import invalid_orders_csv
from returniq_engine.validate import validate_csv_frames


def test_invalid_orders_fixture_yields_reason_codes(g42):
    rej = validate_csv_frames(invalid_orders_csv(g42.orders))
    codes = {r.code for r in rej}
    assert {
        "MISSING_VALUE",
        "INVALID_ENUM",
        "INVALID_QUANTITY",
        "DUPLICATE_CONFLICT",
        "INVALID_PINCODE",
        "AMBIGUOUS_DATE",
        "INVALID_MONEY",
    } <= codes


def test_missing_required_column_is_named():
    rej = validate_csv_frames(pd.DataFrame({"order_id": ["a"]}))
    assert any(r.code == "MISSING_COLUMN" and r.key == "customer_id" for r in rej)


def test_identical_duplicate_rows_are_deduplicated_not_rejected(g42):
    o = pd.concat([g42.orders.head(3), g42.orders.head(1)])
    assert validate_csv_frames(o) == []


def test_negative_ttr_rejected(g42):
    r = g42.returns.head(1).copy()
    sh = g42.shipments.set_index("order_id").loc[[r["order_id"].iloc[0]]].reset_index()
    sh["delivered_at"] = "2030-01-01T00:00:00+05:30"
    rej = validate_csv_frames(g42.orders, sh, r)
    assert any(x.code == "NEGATIVE_DURATION" and x.file == "returns" for x in rej)
