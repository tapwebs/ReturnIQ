import re

import numpy as np

from returniq_engine import generate_dataset
from returniq_engine.generator import RISKY_PINCODES, to_csv_bytes
from returniq_engine.validate import validate_csv_frames


def test_shape_and_ids(g42):
    assert len(g42.orders) == 1200
    assert g42.orders["order_id"].is_unique and g42.returns["return_id"].is_unique
    assert g42.orders["order_id"].str.match(r"^ord_").all()
    assert g42.orders["customer_id"].str.match(r"^cus_").all()
    assert g42.returns["return_id"].str.match(r"^ret_").all()


def test_validator_passes_with_zero_rejections(g42):
    assert validate_csv_frames(g42.orders, g42.shipments, g42.returns) == []


def test_labels_hidden_from_csv(g42):
    for frame in (g42.orders, g42.shipments, g42.returns):
        text = to_csv_bytes(frame).decode()
        assert "is_preventable_return" not in text and "cohort" not in text
    assert {"is_rto", "is_return", "is_preventable_return", "cohort"} <= set(g42.labels.columns)


def test_pincodes_six_digit_and_cod_share(g42):
    pins = g42.orders["pincode"].dropna()
    assert pins.map(lambda p: bool(re.fullmatch(r"\d{6}", p))).all()
    cod = g42.orders["payment_mode"].isin(["COD", "PARTIAL_COD"]).mean()
    assert 0.45 < cod < 0.70
    assert set(RISKY_PINCODES)


def test_controls_present(g42):
    r = g42.returns
    assert r["return_reason"].isin(["DEFECTIVE", "DAMAGED_IN_TRANSIT", "WRONG_ITEM"]).any()
    deliv = g42.shipments[g42.shipments["delivery_status"] == "DELIVERED"]
    assert deliv["delivered_at"].isna().any()  # missing timestamps
    assert g42.shipments["delivery_status"].isin(["NDR", "LOST", "RTO"]).any()


def test_risky_pincodes_have_higher_rto(g42):
    o = g42.orders.merge(g42.shipments, on="order_id")
    risky = o["pincode"].isin(RISKY_PINCODES)
    rto = o["delivery_status"] == "RTO"
    assert rto[risky].mean() > 1.5 * rto[~risky].mean()


def test_large_dataset_runs_fast():
    import time

    t = time.perf_counter()
    g = generate_dataset(25000, 3)
    assert len(g.orders) == 25000
    assert time.perf_counter() - t < 10
    assert np.isfinite(g.orders["order_amount_inr"]).all()
