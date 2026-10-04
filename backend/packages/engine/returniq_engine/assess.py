"""Risk assessment construction. Everything here is a lookup on the prepared frames."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from returniq_contracts import (
    Components,
    RiskAssessment,
    RiskLevel,
    ScoringMethod,
    Settings,
    Signal,
    SignalSource,
    Stage,
)

from .actions import actions_for
from .anomaly import anomaly_percentiles
from .calibrate import band, blend
from .explain import confidence, ml_signals, reason_text, rule_signals
from .prepare import Dataset
from .settings import RULE_VERSION

if TYPE_CHECKING:
    from .ml import TrainedModel

ANOMALY_SIGNAL_MIN = 90.0
RULE_COLS_SKIP = ("total",)


@dataclass
class StageView:
    """Column-wise arrays for one stage, built once per prepared dataset."""

    stage: Stage
    positions: np.ndarray
    order_ids: list[str]
    customer_ids: list[str]
    as_of: list[datetime]
    rule: np.ndarray
    conf: np.ndarray
    feats: dict[str, np.ndarray]
    points: dict[str, np.ndarray]
    present: dict[str, np.ndarray]
    index_of: dict[str, int]
    feature_frame: pd.DataFrame
    ml_cache: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    anomaly: np.ndarray | None = None


def settings_hash(s: Settings) -> str:
    return hashlib.sha1(s.model_dump_json().encode()).hexdigest()[:10]


def get_view(ds: Dataset, stage: Stage) -> StageView:
    """Build (once) and return the column view for a stage."""
    views: dict[Stage, StageView] = ds.__dict__.setdefault("_views", {})
    if stage in views:
        return views[stage]
    p = ds.prepare()
    m = ds.master
    if stage == Stage.PRE_DISPATCH:
        pos = np.arange(len(m))
        frame, rules = p.s1, p.rules1
        ts = m["order_ts"]
        has_deliv = None
        present = {
            "category": m["category"].notna().to_numpy(),
            "pincode": m["pincode"].notna().to_numpy(),
        }
        conf = confidence(1, present, frame["prior_orders"].to_numpy(), None)
    else:
        pos = p.s2_pos
        frame, rules = p.s2, p.rules2
        ts = m["ret_ts"].iloc[pos]
        has_deliv = m["delivered_ts"].notna().to_numpy()[pos]
        present = {
            "category": m["category"].notna().to_numpy()[pos],
            "pincode": m["pincode"].notna().to_numpy()[pos],
            "delivered_at": has_deliv,
        }
        conf = confidence(2, present, frame["prior_orders"].to_numpy(), has_deliv)
    ids = m["order_id"].to_numpy()[pos].tolist()
    view = StageView(
        stage=stage,
        positions=pos,
        order_ids=ids,
        customer_ids=m["customer_id"].to_numpy()[pos].tolist(),
        as_of=[t.to_pydatetime() for t in ts.tolist()],
        rule=rules["total"].to_numpy(),
        conf=conf,
        feats={c: frame[c].to_numpy() for c in frame.columns},
        points={c: rules[c].to_numpy() for c in rules.columns if c not in RULE_COLS_SKIP},
        present=present,
        index_of={o: i for i, o in enumerate(ids)},
        feature_frame=frame,
    )
    views[stage] = view
    return view


def anomaly_scores(view: StageView) -> np.ndarray:
    """Percentile-ranked IsolationForest score within the seller's own history (cached)."""
    if view.anomaly is None:
        view.anomaly = anomaly_percentiles(view.feature_frame)
    return view.anomaly


def _ml(view: StageView, model: TrainedModel) -> tuple[np.ndarray, np.ndarray]:
    key = model.info.id
    if key not in view.ml_cache:
        x = view.feature_frame[model.feature_names]
        view.ml_cache[key] = (model.predict_component(x), model.contributions(x))
    return view.ml_cache[key]


def stage_scores(ds: Dataset, stage: Stage, s: Settings, model: TrainedModel | None) -> np.ndarray:
    """Vectorised blended risk index per stage row (NaN for INSUFFICIENT_DATA)."""
    v = get_view(ds, stage)
    w = s.weights
    num, den = w.rule * v.rule, w.rule
    if model is not None and model.stage == stage:
        ml, _ = _ml(v, model)
        num, den = num + w.ml * ml, den + w.ml
    if s.anomaly_enabled:
        num, den = num + w.anomaly * anomaly_scores(v), den + w.anomaly
    score = num / den
    score = np.clip(np.rint(score), 0, 100)
    return np.where(v.conf < s.min_confidence, np.nan, score)


def _missing(view: StageView, i: int) -> list[str]:
    return [k for k, arr in view.present.items() if not arr[i]]


def _make(
    ds: Dataset,
    view: StageView,
    i: int,
    s: Settings,
    model: TrainedModel | None,
    sh: str,
    latency_ms: float,
) -> RiskAssessment:
    use_ml = model is not None and model.stage == view.stage
    stage_no = 1 if view.stage == Stage.PRE_DISPATCH else 2
    rule = float(view.rule[i])
    ml_val = contrib = None
    if use_ml:
        assert model is not None
        comp, cm = _ml(view, model)
        ml_val, contrib = float(comp[i]), cm[i]
    anomaly_val = float(anomaly_scores(view)[i]) if s.anomaly_enabled else None
    conf = float(view.conf[i])
    insufficient = conf < s.min_confidence
    row = {c: (v[i].item() if hasattr(v[i], "item") else v[i]) for c, v in view.feats.items()}
    pts = {c: float(v[i]) for c, v in view.points.items()}
    signals = rule_signals(pts, row, stage_no)
    if use_ml and contrib is not None and not insufficient:
        assert model is not None and ml_val is not None
        signals += ml_signals(model.feature_names, contrib, ml_val, row)
        signals.sort(key=lambda sg: (-sg.contribution, sg.code))
    if anomaly_val is not None and anomaly_val >= ANOMALY_SIGNAL_MIN and not insufficient:
        signals.append(
            Signal(
                code="UNUSUAL_PROFILE",
                label="Order profile is unusual compared with this seller's own history",
                evidence=f"anomaly_percentile={anomaly_val:.0f}",
                source=SignalSource.ANOMALY,
                contribution=round(anomaly_val * s.weights.anomaly, 1),
            )
        )
        signals.sort(key=lambda sg: (-sg.contribution, sg.code))
    weights_in = {"rule": s.weights.rule, "ml": s.weights.ml, "anomaly": s.weights.anomaly}
    if insufficient:
        score, level, comps, used = None, RiskLevel.INSUFFICIENT_DATA, Components(), Components()
    else:
        score, used_w = blend({"rule": rule, "ml": ml_val, "anomaly": anomaly_val}, weights_in)
        level = band(score, s.thresholds)
        comps = Components(
            rule=round(rule, 1),
            ml=None if ml_val is None else round(ml_val, 1),
            anomaly=None if anomaly_val is None else round(anomaly_val, 1),
        )
        used = Components(**{k: round(v, 4) for k, v in used_w.items()})
    missing = _missing(view, i)
    acts = actions_for(view.stage, level, s.mode)
    order_id = view.order_ids[i]
    mv = model.info.id if use_ml and model is not None else None
    rid = hashlib.sha1(
        f"{order_id}|{view.stage.value}|{view.as_of[i].isoformat()}|{RULE_VERSION}|{mv}|{sh}".encode()
    ).hexdigest()[:12]
    return RiskAssessment(
        id=f"ras_{rid}",
        order_id=order_id,
        customer_id=view.customer_ids[i],
        stage=view.stage,
        as_of=view.as_of[i],
        risk_score=score,
        risk_level=level,
        confidence=conf,
        components=comps,
        weights=used,
        signals=signals,
        reason=reason_text(level, score, signals, missing),
        recommended_action=acts[0],
        alternative_actions=acts[1:],
        missing_fields=missing,
        scoring_method=(
            ScoringMethod.RULES_AND_MODEL
            if use_ml or anomaly_val is not None
            else ScoringMethod.RULES
        ),
        rule_version=RULE_VERSION,
        model_version=mv,
        provenance=ds.provenance,
        latency_ms=round(latency_ms, 3),
    )


def _no_return(
    ds: Dataset, order_id: str, stage: Stage, as_of: datetime, s: Settings
) -> RiskAssessment:
    m = ds.master
    row = m.iloc[ds.order_pos[order_id]]
    acts = actions_for(stage, RiskLevel.INSUFFICIENT_DATA, s.mode)
    return RiskAssessment(
        id="ras_" + hashlib.sha1(f"{order_id}|{stage.value}|noreturn".encode()).hexdigest()[:12],
        order_id=order_id,
        customer_id=str(row["customer_id"]),
        stage=stage,
        as_of=as_of,
        risk_score=None,
        risk_level=RiskLevel.INSUFFICIENT_DATA,
        confidence=0.0,
        components=Components(),
        weights=Components(),
        signals=[],
        reason="No return request exists for this order, so there is nothing to assess.",
        recommended_action=acts[0],
        alternative_actions=acts[1:],
        missing_fields=["return_request"],
        scoring_method=ScoringMethod.RULES,
        rule_version=RULE_VERSION,
        model_version=None,
        provenance=ds.provenance,
    )


def assess(
    ds: Dataset,
    order_id: str,
    stage: Stage,
    as_of: datetime,
    s: Settings,
    model: TrainedModel | None,
) -> RiskAssessment:
    """Assess one order at its decision point (row lookup; ``as_of`` must not precede it)."""
    t0 = time.perf_counter()
    if order_id not in ds.order_pos:
        raise KeyError(order_id)
    view = get_view(ds, stage)
    i = view.index_of.get(order_id)
    if i is None:
        return _no_return(ds, order_id, stage, as_of, s)
    if pd.Timestamp(as_of).tz_convert("UTC") < pd.Timestamp(view.as_of[i]):
        raise ValueError(f"as_of {as_of} precedes the decision point {view.as_of[i]}")
    return _make(ds, view, i, s, model, settings_hash(s), (time.perf_counter() - t0) * 1000.0)


def assess_batch(
    ds: Dataset, stage: Stage, s: Settings, model: TrainedModel | None
) -> list[RiskAssessment]:
    """Assess every order (Stage 1) or every return request (Stage 2), sorted by order_id."""
    t0 = time.perf_counter()
    view = get_view(ds, stage)
    sh = settings_hash(s)
    order = sorted(range(len(view.order_ids)), key=view.order_ids.__getitem__)
    out = [_make(ds, view, i, s, model, sh, 0.0) for i in order]
    per = (time.perf_counter() - t0) * 1000.0 / max(len(out), 1)
    for a in out:
        a.latency_ms = round(per, 3)
    return out
