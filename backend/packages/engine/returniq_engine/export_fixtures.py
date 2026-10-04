"""Generate the shared fixtures: CSVs, one JSON per P0 response (Envelope-wrapped), scenarios.

Usage: ``python -m returniq_engine.export_fixtures --seed 42 --n 1200 --out DIR``.
Output is byte-deterministic (fixed ``generated_at``, sorted keys). Hidden labels are never written.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel
from returniq_contracts import (
    CONTRACT_VERSION,
    Action,
    ActionStatus,
    ActionTarget,
    ActionType,
    AgentProvider,
    AgentRun,
    AgentRunStatus,
    AgentStep,
    AgentStepStatus,
    AuditEntry,
    Filters,
    Job,
    JobStatus,
    OrdersPage,
    RequestedBy,
    Stage,
    is_critical,
)

from .api import ReturnIQEngine
from .generator import generate_dataset, to_csv_bytes
from .prepare import Dataset
from .settings import default_settings
from .views import order_detail, order_from_row

FIXED_AT = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
PAGE_SIZE = 25
FIXED_LATENCY_MS = 1.0

SCENARIOS = (
    (
        1,
        "Normal prepaid order",
        "ord_scn_01",
        Stage.PRE_DISPATCH,
        ["LOW"],
        ["NORMAL_FULFILMENT"],
        "Prepaid, clean history.",
    ),
    (
        2,
        "COD order, 2 prior RTOs",
        "ord_scn_02",
        Stage.PRE_DISPATCH,
        ["HIGH"],
        ["OTP_VERIFICATION", "PREPAID_ONLY", "HOLD_FULFILMENT"],
        "Stage 1 friction before dispatch.",
    ),
    (
        3,
        "Legitimate fast defect return",
        "ord_scn_03",
        Stage.POST_DELIVERY,
        ["LOW"],
        ["INSTANT_RETURN"],
        "DEFECTIVE in 3 h: never treated as preventable.",
    ),
    (
        4,
        "Repeat fast returner",
        "ord_scn_04",
        Stage.POST_DELIVERY,
        ["HIGH"],
        ["MANUAL_REVIEW", "REFUND_AFTER_QC"],
        "4 of 5 orders returned, median 3.5 h, same category.",
    ),
    (
        5,
        "Late return, first-time customer",
        "ord_scn_05",
        Stage.POST_DELIVERY,
        ["LOW", "MEDIUM"],
        ["INSTANT_RETURN", "PHOTO_VERIFICATION", "EXCHANGE_OR_STORE_CREDIT"],
        "Cold start: scored on SKU, category, pincode and payment signals.",
    ),
    (
        6,
        "Missing delivery timestamp",
        "ord_scn_06",
        Stage.POST_DELIVERY,
        ["INSUFFICIENT_DATA"],
        ["INSTANT_RETURN"],
        "TTR unknown, so the engine refuses to score.",
    ),
    (
        7,
        "Next order after a flagged return",
        "ord_scn_07",
        Stage.PRE_DISPATCH,
        ["MEDIUM", "HIGH"],
        [
            "OTP_VERIFICATION",
            "UPI_PREPAID_INCENTIVE",
            "PARTIAL_PREPAID",
            "PREPAID_ONLY",
            "HOLD_FULFILMENT",
        ],
        "Same customer as scenario 4: the earlier flag raises Stage 1 risk.",
    ),
)


def _fix_latency(obj: Any) -> Any:
    """Measured latency is wall-clock noise: fixtures pin it to a constant."""
    if isinstance(obj, dict):
        return {
            k: (FIXED_LATENCY_MS if k == "latency_ms" else _fix_latency(v)) for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_fix_latency(v) for v in obj]
    return obj


def _dump(model: BaseModel | list[BaseModel]) -> Any:
    if isinstance(model, list):
        return [_fix_latency(m.model_dump(mode="json", by_alias=True)) for m in model]
    return _fix_latency(model.model_dump(mode="json", by_alias=True))


def envelope(data: Any, provenance: str = "SYNTHETIC") -> dict:
    """Envelope with fixed meta so fixtures are deterministic."""
    return {
        "success": True,
        "data": data,
        "error": None,
        "meta": {
            "request_id": "req_fixture",
            "provenance": provenance,
            "generated_at": FIXED_AT.isoformat(),
            "contract_version": CONTRACT_VERSION,
            "next_cursor": None,
        },
    }


def _json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def invalid_orders_csv(orders: pd.DataFrame) -> pd.DataFrame:
    """A deliberately broken slice used by ingestion validation tests."""
    bad = orders.head(9).copy().reset_index(drop=True)
    bad.loc[1, "payment_mode"] = ""
    bad.loc[2, "payment_mode"] = "CARD"
    bad.loc[3, "quantity"] = -1
    bad.loc[4, "order_id"] = bad.loc[0, "order_id"]
    bad.loc[4, "order_amount_inr"] = 1.0
    bad.loc[5, "pincode"] = "12345"
    bad.loc[6, "order_date"] = "2026-01-05T10:00:00"
    bad.loc[7, "order_amount_inr"] = -50.0
    return bad


def build_fixtures(seed: int = 42, n: int = 1200, include_model: bool = True) -> dict[str, bytes]:
    """Return ``{relative_path: bytes}`` for every fixture file."""
    eng = ReturnIQEngine()
    s = default_settings()
    g = generate_dataset(n, seed)
    ds = Dataset.from_generated(g)
    files: dict[str, bytes] = {
        "csv/orders.csv": to_csv_bytes(g.orders),
        "csv/shipments.csv": to_csv_bytes(g.shipments),
        "csv/returns.csv": to_csv_bytes(g.returns),
        "csv/invalid_orders.csv": to_csv_bytes(invalid_orders_csv(g.orders)),
    }
    models = {}
    model_infos = []
    if include_model:
        model = eng.train(ds, Stage.POST_DELIVERY, None)
        models[Stage.POST_DELIVERY] = model
        model_infos.append(model.info)
    f = Filters()
    files["analytics.json"] = _json_bytes(envelope(_dump(eng.analytics(ds, f, s))))
    files["timeline.json"] = _json_bytes(envelope(_dump(eng.timeline(ds, f, s))))
    files["readiness.json"] = _json_bytes(envelope(_dump(eng.readiness(ds, s))))
    files["settings.json"] = _json_bytes(envelope(_dump(s)))
    files["loss_estimate.json"] = _json_bytes(envelope(_dump(eng.estimate_loss(ds, f, s))))
    files["models.json"] = _json_bytes(envelope(_dump(model_infos)))
    for stage, name in (
        (Stage.POST_DELIVERY, "policy_sim.json"),
        (Stage.PRE_DISPATCH, "policy_sim_pre.json"),
    ):
        files[name] = _json_bytes(
            envelope(_dump(eng.simulate_policy(ds, stage, 12.0, s, models.get(stage))))
        )
    files["jobs_completed.json"] = _json_bytes(
        envelope(
            _dump(
                Job(
                    id="job_fixture_0001",
                    status=JobStatus.COMPLETED,
                    dataset_id=ds.dataset_id,
                    accepted=len(ds.master),
                    rejected=0,
                    deduplicated=0,
                    missing_fields=["customer_contact"],
                )
            )
        )
    )

    now = datetime(2027, 1, 1, tzinfo=UTC)
    scen_rows = []
    assessed: dict[str, dict[Stage, Any]] = {}
    for num, name, oid, stage, bands, acts, note in SCENARIOS:
        a = eng.assess(ds, oid, stage, now, s, models.get(stage))
        assessed.setdefault(oid, {})[stage] = a
        tag = "pre" if stage == Stage.PRE_DISPATCH else "post"
        files[f"risk_{oid}_{tag}.json"] = _json_bytes(envelope(_dump(a)))
        row = ds.master.iloc[ds.order_pos[oid]]
        scen_rows.append(
            {
                "scenario": num,
                "name": name,
                "order_id": oid,
                "customer_id": row["customer_id"],
                "stage": stage.value,
                "expected_bands": bands,
                "expected_actions": acts,
                "note": note,
                "actual_band": a.risk_level.value,
                "actual_score": a.risk_score,
            }
        )
    files["scenarios.json"] = _json_bytes(
        {"seed": seed, "n_orders": n, "generator_spec": "gen-1.0.0", "scenarios": scen_rows}
    )
    for oid in ("ord_scn_04", "ord_scn_02", "ord_scn_03", "ord_scn_06"):
        pres = eng.assess(ds, oid, Stage.PRE_DISPATCH, now, s, models.get(Stage.PRE_DISPATCH))
        asm = [pres]
        if ds.master.iloc[ds.order_pos[oid]]["ret_ts"] is not pd.NaT and pd.notna(
            ds.master.iloc[ds.order_pos[oid]]["ret_ts"]
        ):
            asm.append(
                assessed.get(oid, {}).get(Stage.POST_DELIVERY)
                or eng.assess(ds, oid, Stage.POST_DELIVERY, now, s, models.get(Stage.POST_DELIVERY))
            )
        files[f"order_detail_{oid}.json"] = _json_bytes(envelope(_dump(order_detail(ds, oid, asm))))
        if Stage.PRE_DISPATCH not in assessed.get(oid, {}):
            files[f"risk_{oid}_pre.json"] = _json_bytes(envelope(_dump(pres)))
    dna_customers = {"cus_scn_04", "cus_scn_02"}
    for cid in sorted(dna_customers):
        files[f"customer_{cid}_dna.json"] = _json_bytes(envelope(_dump(eng.customer_dna(ds, cid))))
    # orders page 1: first PAGE_SIZE orders by id, latest assessment attached
    pre = {
        a.order_id: a
        for a in eng.assess_batch(ds, Stage.PRE_DISPATCH, s, models.get(Stage.PRE_DISPATCH))
    }
    post = {
        a.order_id: a
        for a in eng.assess_batch(ds, Stage.POST_DELIVERY, s, models.get(Stage.POST_DELIVERY))
    }
    ids = sorted(ds.master["order_id"])[:PAGE_SIZE]
    items = [order_from_row(ds.master.iloc[ds.order_pos[o]], post.get(o) or pre[o]) for o in ids]
    page = OrdersPage(items=items, next_cursor=ids[-1])
    env = envelope(_dump(page))
    env["meta"]["next_cursor"] = ids[-1]
    files["orders_page1.json"] = _json_bytes(env)

    # pending action + scripted agent run sequence built from the real scenario-4 assessment
    a4 = assessed["ord_scn_04"][Stage.POST_DELIVERY]
    row4 = ds.master.iloc[ds.order_pos["ord_scn_04"]]
    act_type = ActionType.REFUND_AFTER_QC
    action = Action(
        id="act_fixture_0001",
        type=act_type,
        stage=Stage.POST_DELIVERY,
        target=ActionTarget(
            order_id="ord_scn_04",
            return_id=str(row4["return_id"]),
            customer_id=str(row4["customer_id"]),
        ),
        assessment_id=a4.id,
        critical=is_critical(act_type),
        requested_by=RequestedBy.AGENT,
        agent_run_id="run_fixture_0001",
        summary="Hold the refund until the item passes QC",
        evidence=[sg.evidence for sg in a4.signals[:3]],
        customer_impact="Refund delayed by about 2 days",
        estimated_cost_paise=0,
        estimated_impact_paise=12_000,
        status=ActionStatus.PENDING_APPROVAL,
        version=1,
        expires_at=datetime(2026, 7, 2, 12, 0, tzinfo=UTC),
        audit=[
            AuditEntry(at=FIXED_AT, actor="agent", event="PROPOSED"),
            AuditEntry(at=FIXED_AT, actor="system", event="PENDING_APPROVAL"),
        ],
    )
    files["actions_pending.json"] = _json_bytes(envelope(_dump([action])))
    steps = [
        AgentStep(
            idx=1,
            tool="get_order_history",
            input={"order_id": "ord_scn_04"},
            output_summary="5 orders, 4 returns, median return time 3.5 h",
            status=AgentStepStatus.DONE,
            ms=31,
        ),
        AgentStep(
            idx=2,
            tool="assess_risk",
            input={"order_id": "ord_scn_04", "stage": "POST_DELIVERY"},
            output_summary=f"Risk index {a4.risk_score} ({a4.risk_level.value})",
            status=AgentStepStatus.DONE,
            ms=14,
        ),
        AgentStep(
            idx=3,
            tool="propose_action",
            input={"type": "REFUND_AFTER_QC", "assessment_id": a4.id},
            output_summary="Critical action: approval required",
            status=AgentStepStatus.DONE,
            ms=9,
        ),
    ]
    ctx = {"dataset_id": ds.dataset_id, "order_id": "ord_scn_04"}
    task = "Review this return request against our policy"
    runs = [
        AgentRun(
            id="run_fixture_0001",
            task=task,
            context=ctx,
            provider=AgentProvider.PLAYBOOK,
            status=AgentRunStatus.QUEUED,
        ),
        AgentRun(
            id="run_fixture_0001",
            task=task,
            context=ctx,
            provider=AgentProvider.PLAYBOOK,
            status=AgentRunStatus.RUNNING,
            steps=steps[:1],
        ),
        AgentRun(
            id="run_fixture_0001",
            task=task,
            context=ctx,
            provider=AgentProvider.PLAYBOOK,
            status=AgentRunStatus.AWAITING_APPROVAL,
            steps=steps,
            proposals=["act_fixture_0001"],
            summary="Proposed holding the refund until QC; waiting for approval.",
        ),
        AgentRun(
            id="run_fixture_0001",
            task=task,
            context=ctx,
            provider=AgentProvider.PLAYBOOK,
            status=AgentRunStatus.SUCCEEDED,
            steps=steps,
            proposals=["act_fixture_0001"],
            summary="Refund hold approved and executed in the sandbox.",
        ),
    ]
    for i, run in enumerate(runs, start=1):
        files[f"agent_run_{i}.json"] = _json_bytes(envelope(_dump(run)))
    return files


def write_fixtures(out: Path, seed: int, n: int, include_model: bool = True) -> list[str]:
    """Write all fixtures under ``out``; returns the sorted relative paths."""
    files = build_fixtures(seed, n, include_model)
    for rel, data in files.items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return sorted(files)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-model", action="store_true", help="skip training the Stage 2 model")
    a = ap.parse_args(argv)
    names = write_fixtures(a.out, a.seed, a.n, include_model=not a.no_model)
    print(f"wrote {len(names)} fixture files to {a.out}")


if __name__ == "__main__":
    main()
