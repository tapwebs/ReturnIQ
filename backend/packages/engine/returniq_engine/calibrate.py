"""Score blending, bands, band tables and isotonic calibration (0.6a)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from returniq_contracts import BandRow, RiskLevel, Thresholds
from sklearn.isotonic import IsotonicRegression

COMPONENTS = ("rule", "ml", "anomaly")


def blend(
    components: Mapping[str, float | None], weights: Mapping[str, float]
) -> tuple[int, dict[str, float]]:
    """``sum(w_i*c_i)/sum(w_i)`` over present components; weights are renormalised.

    Returns the rounded 0..100 risk index and the weights actually used (sum to 1).
    """
    present = {k: float(v) for k, v in components.items() if v is not None}
    if not present:
        raise ValueError("at least one component is required")
    total_w = sum(weights[k] for k in present)
    used = {k: weights[k] / total_w for k in present}
    score = sum(used[k] * min(max(present[k], 0.0), 100.0) for k in present)
    return int(min(max(round(score), 0), 100)), used


def band(score: int | None, thresholds: Thresholds) -> RiskLevel:
    """MEDIUM >= thresholds.medium, HIGH >= thresholds.high."""
    if score is None:
        return RiskLevel.INSUFFICIENT_DATA
    if score >= thresholds.high:
        return RiskLevel.HIGH
    if score >= thresholds.medium:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def largest_remainder(counts: Sequence[int], total_units: int = 1000) -> list[int]:
    """Apportion ``total_units`` proportionally so the parts sum exactly to the total."""
    n = sum(counts)
    if n == 0:
        return [0] * len(counts)
    raw = [c * total_units / n for c in counts]
    base = [int(x) for x in raw]
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - base[i]), i))
    for i in order[: total_units - sum(base)]:
        base[i] += 1
    return base


def band_table(scores: np.ndarray, outcome: np.ndarray, thresholds: Thresholds) -> list[BandRow]:
    """Per band: % of orders (sums to 100.0), observed outcome rate and lift vs overall.

    ``scores`` holds NaN for INSUFFICIENT_DATA; ``outcome`` is a boolean array.
    """
    n = len(scores)
    overall = float(outcome.mean()) if n else None
    levels = np.where(
        np.isnan(scores),
        "INSUFFICIENT_DATA",
        np.where(
            scores >= thresholds.high,
            "HIGH",
            np.where(scores >= thresholds.medium, "MEDIUM", "LOW"),
        ),
    )
    order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.INSUFFICIENT_DATA]
    counts = [int((levels == b.value).sum()) for b in order]
    pct = largest_remainder(counts)
    rows: list[BandRow] = []
    for b, c, p in zip(order, counts, pct, strict=True):
        rate = float(outcome[levels == b.value].mean()) if c else None
        lift = rate / overall if (rate is not None and overall) else None
        rows.append(
            BandRow(band=b, orders=c, pct_orders=p / 10.0, observed_rate=rate, lift_vs_overall=lift)
        )
    return rows


def fit_isotonic(val_scores: np.ndarray, val_y: np.ndarray) -> IsotonicRegression:
    """Monotone calibrator on the validation slice (deterministic)."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(val_scores, val_y)
    return iso
