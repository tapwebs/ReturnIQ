"""Engine outputs and every fixture validate against the generated schemas/*.json."""

import json
import pathlib
import re

import jsonschema
import pytest
from returniq_contracts import Filters, ScorePreviewRequest, Stage
from returniq_contracts.export import DEFAULT_OUT, exported_models, render_schema

from returniq_engine.export_fixtures import build_fixtures

from .conftest import FAR_FUTURE

SCHEMAS = pathlib.Path(DEFAULT_OUT)


def schema(name: str) -> dict:
    return json.loads((SCHEMAS / f"{name}.json").read_text())


def check(name: str, instance) -> None:
    jsonschema.validate(instance, schema(name))


@pytest.fixture(scope="module")
def fixtures():
    return build_fixtures(42, 1200, include_model=True)


def test_committed_schemas_are_up_to_date():
    for name, model in exported_models().items():
        assert (SCHEMAS / f"{name}.json").read_text() == render_schema(model), name
    assert (SCHEMAS.parent / "CONTRACT_VERSION").read_text().strip() == "1.0.0"


def test_engine_outputs_validate(eng, ds42, settings, model_s2):
    dump = lambda m: m.model_dump(mode="json", by_alias=True)  # noqa: E731
    f = Filters()
    for stage, model in ((Stage.PRE_DISPATCH, None), (Stage.POST_DELIVERY, model_s2)):
        for a in eng.assess_batch(ds42, stage, settings, model)[:50]:
            check("RiskAssessment", dump(a))
    check("Analytics", dump(eng.analytics(ds42, f, settings)))
    check("Timeline", dump(eng.timeline(ds42, f, settings)))
    check("Readiness", dump(eng.readiness(ds42, settings)))
    check("LossEstimate", dump(eng.estimate_loss(ds42, f, settings)))
    check("Settings", dump(settings))
    check("ModelInfo", dump(model_s2.info))
    check(
        "PolicySimulation", dump(eng.simulate_policy(ds42, Stage.POST_DELIVERY, 12, settings, None))
    )
    check("ReturnDNA", dump(eng.customer_dna(ds42, "cus_scn_04")))
    req = ScorePreviewRequest.model_validate(
        {
            "order": {
                "order_date": "2026-05-01T10:00:00+05:30",
                "payment_mode": "COD",
                "order_amount_paise": 150000,
                "category": "FASHION",
                "pincode": "110001",
            },
            "return_request": {
                "return_requested_at": "2026-05-05T10:00:00+05:30",
                "delivered_at": "2026-05-05T06:00:00+05:30",
                "return_reason": "CHANGED_MIND",
            },
        }
    )
    resp = eng.preview(req, None, settings, {})
    check("ScorePreviewResponse", dump(resp))
    assert resp.post_delivery is not None


def test_every_fixture_validates(fixtures):
    envelope = schema("Envelope")
    mapping = [
        (r"^analytics", "Analytics", False),
        (r"^timeline", "Timeline", False),
        (r"^readiness", "Readiness", False),
        (r"^settings", "Settings", False),
        (r"^loss_estimate", "LossEstimate", False),
        (r"^policy_sim", "PolicySimulation", False),
        (r"^jobs_completed", "Job", False),
        (r"^risk_", "RiskAssessment", False),
        (r"^order_detail_", "OrderDetail", False),
        (r"^customer_.*_dna", "ReturnDNA", False),
        (r"^orders_page1", "OrdersPage", False),
        (r"^actions_pending", "Action", True),
        (r"^agent_run_", "AgentRun", False),
        (r"^models", "ModelInfo", True),
    ]
    seen = 0
    for rel, blob in fixtures.items():
        if not rel.endswith(".json") or rel == "scenarios.json":
            continue
        doc = json.loads(blob)
        jsonschema.validate(doc, envelope)
        for pattern, name, is_list in mapping:
            if re.match(pattern, rel):
                items = doc["data"] if is_list else [doc["data"]]
                for item in items:
                    check(name, item)
                seen += 1
                break
        else:
            pytest.fail(f"no schema mapping for fixture {rel}")
    assert seen >= 25


def test_fixture_set_is_complete(fixtures):
    for required in (
        "csv/orders.csv",
        "csv/shipments.csv",
        "csv/returns.csv",
        "csv/invalid_orders.csv",
        "analytics.json",
        "orders_page1.json",
        "order_detail_ord_scn_04.json",
        "risk_ord_scn_04_post.json",
        "actions_pending.json",
        "policy_sim.json",
        "models.json",
        "scenarios.json",
    ):
        assert required in fixtures
    assert not any("is_preventable_return" in v.decode() for v in fixtures.values())


def test_pydantic_rejects_unknown_fields(eng, ds42, settings):
    from returniq_contracts import RiskAssessment

    a = eng.assess(ds42, "ord_scn_01", Stage.PRE_DISPATCH, FAR_FUTURE, settings, None)
    bad = a.model_dump() | {"surprise": 1}
    with pytest.raises(ValueError):
        RiskAssessment.model_validate(bad)


def test_engine_implements_protocol_signatures():
    import inspect

    from returniq_engine.api import Engine, ReturnIQEngine

    for name, proto in inspect.getmembers(Engine, inspect.isfunction):
        if name.startswith("_"):
            continue
        impl = getattr(ReturnIQEngine, name)
        assert list(inspect.signature(impl).parameters) == list(inspect.signature(proto).parameters)
