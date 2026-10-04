"""LightGBM stage models: time-based 70/10/20 split, isotonic calibration, pred_contrib.

Training labels: the generator's hidden labels when the dataset carries them (SYNTHETIC data
only); otherwise ``train`` needs seller outcomes. Hidden labels never reach CSV or features.
The artifact is pickle-free (zlib JSON holding the LightGBM text model + isotonic thresholds).
"""

from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from returniq_contracts import (
    BandRow,
    CalibrationBin,
    CalibrationReport,
    ModelInfo,
    ModelMetrics,
    Outcome,
    OutcomeResult,
    Settings,
    Stage,
    Thresholds,
)
from sklearn.metrics import roc_auc_score

from .assess import get_view
from .calibrate import band_table, fit_isotonic
from .features import STAGE1_FEATURES, STAGE2_FEATURES
from .prepare import Dataset
from .settings import default_settings

TOP_FRACTION = 0.12
SPLIT = (0.70, 0.10, 0.20)
PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 10,
    "feature_fraction": 0.9,
    "bagging_fraction": 1.0,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
}
NUM_ROUNDS = 300
EARLY_STOP = 25
MIN_ROWS = 60


@dataclass
class TrainedModel:
    """A calibrated LightGBM stage model plus its serialised artifact."""

    stage: Stage
    feature_names: list[str]
    booster: lgb.Booster
    iso_x: np.ndarray
    iso_y: np.ndarray
    info: ModelInfo
    artifact: bytes = field(repr=False, default=b"")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Calibrated probability in [0, 1]."""
        raw = self.booster.predict(x[self.feature_names])
        return np.clip(np.interp(raw, self.iso_x, self.iso_y), 0.0, 1.0)

    def predict_component(self, x: pd.DataFrame) -> np.ndarray:
        """Calibrated probability x 100 (the ML component, 0..100)."""
        return self.predict_proba(x) * 100.0

    def contributions(self, x: pd.DataFrame) -> np.ndarray:
        """Per-feature contributions (log-odds), bias column dropped."""
        return self.booster.predict(x[self.feature_names], pred_contrib=True)[:, :-1]


def top_fraction_metrics(
    y: np.ndarray, score: np.ndarray, frac: float = TOP_FRACTION
) -> tuple[float | None, float | None]:
    """Precision and recall within the top ``frac`` of ``score`` (ties broken by index)."""
    n = len(y)
    k = max(1, int(np.ceil(n * frac)))
    idx = np.lexsort((np.arange(n), -score))[:k]
    hits = float(y[idx].sum())
    pos = float(y.sum())
    return hits / k, (hits / pos if pos else None)


def auc_safe(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def _labels(ds: Dataset, stage: Stage, outcomes: list[Outcome] | None) -> np.ndarray:
    """Binary label aligned to the stage view rows."""
    v = get_view(ds, stage)
    ids = pd.Index(v.order_ids)
    y = np.full(len(ids), np.nan)
    if ds.labels is not None:
        lab = ds.labels.set_index("order_id").reindex(ids)
        col = "is_rto" if stage == Stage.PRE_DISPATCH else "is_preventable_return"
        y = lab[col].to_numpy(dtype=float)
    if outcomes:
        for o in outcomes:
            oid = getattr(o, "order_id", None)
            if oid in v.index_of:
                if o.outcome == OutcomeResult.SUCCESS:
                    y[v.index_of[oid]] = 1.0
                elif o.outcome == OutcomeResult.OVERRIDDEN:
                    y[v.index_of[oid]] = 0.0
    if np.isnan(y).all():
        raise ValueError("no training labels: dataset has no labels and no outcomes were given")
    return y


def split_indices(as_of: list, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-based split by ``as_of``: earliest 70% train, next 10% validation, last 20% test."""
    order = np.lexsort((np.arange(n), np.array([t.timestamp() for t in as_of])))
    a, b = int(n * SPLIT[0]), int(n * (SPLIT[0] + SPLIT[1]))
    return order[:a], order[a:b], order[b:]


def _calibration_report(p: np.ndarray, y: np.ndarray) -> CalibrationReport:
    edges = np.linspace(0, 1, 11)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        c = int(sel.sum())
        bins.append(
            CalibrationBin(
                lower=round(lo, 2),
                upper=round(hi, 2),
                count=c,
                mean_predicted=round(float(p[sel].mean()), 4) if c else None,
                observed_rate=round(float(y[sel].mean()), 4) if c else None,
            )
        )
    brier = float(np.mean((p - y) ** 2)) if len(y) else None
    return CalibrationReport(
        bins=bins,
        brier=None if brier is None else round(brier, 5),
        label="test slice, calibrated probability vs observed rate "
        "(simulated, on planted synthetic patterns)",
    )


def _pack(model: TrainedModel) -> bytes:
    payload = {
        "format": "returniq-model-1",
        "stage": model.stage.value,
        "feature_names": model.feature_names,
        "model": model.booster.model_to_string(),
        "iso_x": [float(v) for v in model.iso_x],
        "iso_y": [float(v) for v in model.iso_y],
        "info": json.loads(model.info.model_dump_json()),
    }
    return zlib.compress(json.dumps(payload, sort_keys=True).encode(), 9)


def train(
    ds: Dataset, stage: Stage, outcomes: list[Outcome] | None = None, s: Settings | None = None
) -> TrainedModel:
    """Train + calibrate a stage model; metrics are measured on the held-out last 20%."""
    s = s or default_settings()
    v = get_view(ds, stage)
    cols = list(STAGE1_FEATURES if stage == Stage.PRE_DISPATCH else STAGE2_FEATURES)
    x = v.feature_frame[cols].reset_index(drop=True)
    y = _labels(ds, stage, outcomes)
    n = len(x)
    if n < MIN_ROWS:
        raise ValueError(f"need at least {MIN_ROWS} rows to train, got {n}")
    tr, va, te = split_indices(v.as_of, n)
    tr, va, te = (i[~np.isnan(y[i])] for i in (tr, va, te))
    dtrain = lgb.Dataset(x.iloc[tr], label=y[tr])
    dval = lgb.Dataset(x.iloc[va], label=y[va], reference=dtrain)
    booster = lgb.train(
        PARAMS,
        dtrain,
        num_boost_round=NUM_ROUNDS,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )
    raw_val = booster.predict(x.iloc[va])
    if len(np.unique(y[va])) > 1:
        iso = fit_isotonic(raw_val, y[va])
        iso_x, iso_y = iso.X_thresholds_, iso.y_thresholds_
    else:
        iso_x, iso_y = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    p_test = np.clip(np.interp(booster.predict(x.iloc[te]), iso_x, iso_y), 0.0, 1.0)
    y_test = y[te]
    rules_test = v.rule[te]
    prec, rec = top_fraction_metrics(y_test, p_test)
    ml_comp = p_test * 100.0
    w = s.weights
    blended = np.clip(np.rint((w.rule * rules_test + w.ml * ml_comp) / (w.rule + w.ml)), 0, 100)
    stage_tag = "s1" if stage == Stage.PRE_DISPATCH else "s2"
    digest = hashlib.sha256(
        (
            booster.model_to_string()
            + ",".join(f"{a:.9f}" for a in iso_x)
            + ",".join(f"{a:.9f}" for a in iso_y)
        ).encode()
    ).hexdigest()[:6]
    trained_at = max(v.as_of)
    rows_train = int(len(tr) + len(va))
    info = ModelInfo(
        id=f"mdl_{stage_tag}_{digest}",
        stage=stage,
        trained_at=trained_at,
        rows=rows_train,
        metrics=ModelMetrics(
            auc=auc_safe(y_test, p_test),
            rules_auc=auc_safe(y_test, rules_test),
            precision_at_12=prec,
            recall_at_12=rec,
            preventable_captured_pct=None if rec is None else round(rec * 100, 2),
        ),
        split="TIME_BASED",
        provenance=ds.provenance,
        band_table=band_table(blended, y_test.astype(bool), Thresholds()),
        calibration=_calibration_report(p_test, y_test),
        feature_names=cols,
    )
    model = TrainedModel(stage, cols, booster, np.asarray(iso_x), np.asarray(iso_y), info)
    model.artifact = _pack(model)
    return model


def load_model(artifact: bytes) -> TrainedModel:
    """Rebuild a model from artifact bytes (no pickle)."""
    payload = json.loads(zlib.decompress(artifact).decode())
    if payload.get("format") != "returniq-model-1":
        raise ValueError("unrecognised model artifact")
    info = ModelInfo.model_validate(payload["info"])
    return TrainedModel(
        stage=Stage(payload["stage"]),
        feature_names=list(payload["feature_names"]),
        booster=lgb.Booster(model_str=payload["model"]),
        iso_x=np.array(payload["iso_x"]),
        iso_y=np.array(payload["iso_y"]),
        info=info,
        artifact=artifact,
    )


__all__ = [
    "BandRow",
    "TrainedModel",
    "load_model",
    "train",
    "top_fraction_metrics",
    "auc_safe",
    "split_indices",
]
