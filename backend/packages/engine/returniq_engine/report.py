"""`python -m returniq_engine.report --seed 42`: analytics, readiness, band tables, model metrics.

Every figure printed here is simulated, on planted synthetic patterns. Hidden labels are used
for evaluation only.
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC

import numpy as np
from returniq_contracts import Filters, Settings, Stage

from .api import ReturnIQEngine
from .assess import get_view, stage_scores
from .calibrate import band_table
from .ml import auc_safe, split_indices, top_fraction_metrics
from .prepare import Dataset
from .settings import RULE_VERSION, default_settings

NOTE = "simulated, on planted synthetic patterns"


def label_array(ds: Dataset, stage: Stage) -> np.ndarray:
    """Hidden evaluation label aligned to the stage rows: RTO (Stage 1) / preventable (Stage 2)."""
    assert ds.labels is not None
    v = get_view(ds, stage)
    col = "is_rto" if stage == Stage.PRE_DISPATCH else "is_preventable_return"
    return ds.labels.set_index("order_id")[col].reindex(v.order_ids).to_numpy(dtype=bool)


def stage_report(ds: Dataset, stage: Stage, s: Settings, model=None) -> dict:
    """Band table, shares, AUC and top-12% metrics for one stage (full population + test slice)."""
    v = get_view(ds, stage)
    scores = stage_scores(ds, stage, s, model)
    y = label_array(ds, stage)
    table = band_table(scores, y, s.thresholds)
    _, _, te = split_indices(v.as_of, len(scores))
    ok = ~np.isnan(scores)
    prec, rec = top_fraction_metrics(y[te], np.nan_to_num(scores[te]))
    return {
        "stage": stage.value,
        "n": len(scores),
        "band_table": table,
        "insufficient_share": float((~ok).mean()),
        "auc_all": auc_safe(y[ok], scores[ok]),
        "auc_test": auc_safe(y[te], np.nan_to_num(scores[te])),
        "precision_at_12_test": prec,
        "recall_at_12_test": rec,
        "high_share": float(np.nansum(scores >= s.thresholds.high) / len(scores)),
    }


def perf(seed: int = 1, n: int = 25000) -> dict:
    """Measured timings: generate, prepare, assess_batch(25k), assess() p50/p95 over 1,000 calls."""
    from datetime import datetime

    eng = ReturnIQEngine()
    s = default_settings()
    t = time.perf_counter()
    g = eng.generate_dataset(n, seed)
    t_gen = time.perf_counter() - t
    t = time.perf_counter()
    ds = Dataset.from_generated(g)
    t_build = time.perf_counter() - t
    t = time.perf_counter()
    ds.prepare()
    t_prep = time.perf_counter() - t
    t = time.perf_counter()
    out = eng.assess_batch(ds, Stage.PRE_DISPATCH, s, None)
    t_batch = time.perf_counter() - t
    ids = ds.master["order_id"].to_numpy()
    rng = np.random.default_rng(0)
    pick = rng.choice(ids, size=1000)
    now = datetime(2027, 1, 1, tzinfo=UTC)
    lat = []
    for oid in pick:
        t = time.perf_counter()
        eng.assess(ds, str(oid), Stage.PRE_DISPATCH, now, s, None)
        lat.append((time.perf_counter() - t) * 1000.0)
    return {
        "n": n,
        "generate_s": t_gen,
        "build_s": t_build,
        "prepare_s": t_prep,
        "assess_batch_s": t_batch,
        "assessed": len(out),
        "assess_p50_ms": float(np.percentile(lat, 50)),
        "assess_p95_ms": float(np.percentile(lat, 95)),
    }


def evaluate_seeds(seeds: tuple[int, ...] = (41, 42, 43), n: int = 10000) -> list[dict]:
    """Train and evaluate both stage models per seed on the held-out last 20% (time split)."""
    eng = ReturnIQEngine()
    out = []
    for seed in seeds:
        ds = Dataset.from_generated(eng.generate_dataset(n, seed))
        for stage in (Stage.PRE_DISPATCH, Stage.POST_DELIVERY):
            m = eng.train(ds, stage, None).info.metrics
            out.append(
                {
                    "seed": seed,
                    "n": n,
                    "stage": stage.value,
                    "ml_auc": m.auc,
                    "rules_auc": m.rules_auc,
                    "p_at_12": m.precision_at_12,
                    "r_at_12": m.recall_at_12,
                }
            )
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--perf", action="store_true", help="also print 25,000-order timings")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--eval", action="store_true", help="3 seeds x n=10,000 model evaluation")
    a = ap.parse_args(argv)
    eng = ReturnIQEngine()
    s = default_settings()
    ds = Dataset.from_generated(eng.generate_dataset(a.n, a.seed))
    an = eng.analytics(ds, Filters(), s)
    print(f"ReturnIQ report | seed={a.seed} n={a.n} | rules={RULE_VERSION} | {NOTE}")
    print(f"dataset_version={ds.dataset_version} orders={an.orders} delivered={an.delivered}")
    print(f"RTO: {an.rto.count}/{an.rto.denominator} = {an.rto.rate}% (pending {an.rto.pending})")
    print(
        f"Returns: accepted {an.returns.accepted}/{an.returns.denominator} = {an.returns.rate}% "
        f"(requested {an.returns.requested}, {an.returns.requested_rate}%)"
    )
    print(f"Flagged returns: {an.flagged_returns}; estimated PRR = {an.estimated_prr}% (estimate)")
    print(
        f"Observed operating loss (ex order value): Rs {an.loss.observed_paise / 100:,.2f}; "
        f"assumed fields: {', '.join(an.loss.assumed_fields)}"
    )
    na = an.est_net_avoidable
    print(
        f"Est. net avoidable (low/base/high) Rs: {na.low / 100:,.0f} / {na.base / 100:,.0f} / "
        f"{na.high / 100:,.0f} (scenario, assumed costs)"
    )
    print(f"Readiness: {an.readiness.mode.value} ({an.readiness.reason})")
    print(
        f"TTR median {an.ttr.median_h} h; buckets "
        + ", ".join(f"{b.bucket.value}={b.count}" for b in an.ttr.buckets)
    )
    models = {}
    for stage in (Stage.PRE_DISPATCH, Stage.POST_DELIVERY):
        variants = [("rules-only", None)]
        if not a.no_model:
            models[stage] = eng.train(ds, stage, None)
            variants.append(("rules+ml (0.5/0.5)", models[stage]))
        for name, model in variants:
            r = stage_report(ds, stage, s, model)
            label = "RTO" if stage == Stage.PRE_DISPATCH else "preventable-return"
            scope = "all rows" if model is None else "all rows, in-sample for the model"
            print(
                f"\n{stage.value} band table [{name}] ({scope}; n={r['n']}, observed {label} rate)"
            )
            print(f"{'band':<18}{'orders':>8}{'% orders':>10}{'observed':>10}{'lift':>8}")
            for b in r["band_table"]:
                obs = "n/a" if b.observed_rate is None else f"{b.observed_rate * 100:.1f}%"
                lift = "n/a" if b.lift_vs_overall is None else f"{b.lift_vs_overall:.2f}x"
                print(f"{b.band.value:<18}{b.orders:>8}{b.pct_orders:>9.1f}%{obs:>10}{lift:>8}")
            print(
                f"INSUFFICIENT_DATA share {r['insufficient_share'] * 100:.1f}% | HIGH share "
                f"{r['high_share'] * 100:.1f}% | AUC(all rows) {r['auc_all']} "
                f"| AUC(test) {r['auc_test']} "
                f"| P@12 {r['precision_at_12_test']} R@12 {r['recall_at_12_test']}"
            )
        if stage in models:
            m = models[stage].info.metrics
            print(f"{stage.value} band table [rules+ml, held-out test slice from ModelInfo]")
            for b in models[stage].info.band_table:
                obs = "n/a" if b.observed_rate is None else f"{b.observed_rate * 100:.1f}%"
                print(f"  {b.band.value:<18}{b.orders:>6}{b.pct_orders:>8.1f}%{obs:>9}")
            print(
                f"ML test metrics: AUC={m.auc:.4f} rules_AUC={m.rules_auc:.4f} "
                f"P@12={m.precision_at_12:.3f} R@12={m.recall_at_12:.3f}"
            )
    if a.eval:
        print("\nmulti-seed evaluation (test slice = last 20% by time)")
        for r in evaluate_seeds():
            print(
                f"seed={r['seed']} n={r['n']} {r['stage']:<13} ML AUC={r['ml_auc']:.4f} "
                f"rules AUC={r['rules_auc']:.4f} P@12={r['p_at_12']:.3f} R@12={r['r_at_12']:.3f}"
            )
    if a.perf:
        print("\nperf:", {k: round(v, 4) if isinstance(v, float) else v for k, v in perf().items()})


if __name__ == "__main__":
    main()
