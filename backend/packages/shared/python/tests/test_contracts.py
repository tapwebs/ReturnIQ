"""Contracts package: schema export is deterministic and models round-trip."""

import json

from returniq_contracts import (
    CONTRACT_VERSION,
    ActionType,
    Envelope,
    RiskAssessment,
    Settings,
    is_critical,
)
from returniq_contracts.export import export, exported_models, render_schema


def test_one_schema_file_per_model_and_idempotent(tmp_path):
    out = tmp_path / "schemas"
    export(out)
    first = {p.name: p.read_bytes() for p in out.glob("*.json")}
    export(out)
    second = {p.name: p.read_bytes() for p in out.glob("*.json")}
    assert first == second
    assert set(first) == {f"{n}.json" for n in exported_models()}
    assert (tmp_path / "CONTRACT_VERSION").read_text().strip() == CONTRACT_VERSION == "1.0.0"


def test_envelope_round_trip():
    env = Envelope[dict](success=True, data={"a": 1})
    assert Envelope.model_validate(json.loads(env.model_dump_json())).data == {"a": 1}


def test_settings_defaults_and_critical_flags():
    s = Settings()
    assert (s.thresholds.medium, s.thresholds.high) == (40, 70)
    assert s.cost_defaults_paise.forward_ship_paise == 6000
    assert is_critical(ActionType.REFUND_AFTER_QC) and not is_critical(ActionType.OTP_VERIFICATION)


def test_risk_assessment_schema_has_expected_fields():
    props = json.loads(render_schema(RiskAssessment))["properties"]
    for f in (
        "risk_score",
        "risk_level",
        "confidence",
        "components",
        "weights",
        "signals",
        "recommended_action",
        "scoring_method",
        "rule_version",
        "model_version",
        "provenance",
        "latency_ms",
    ):
        assert f in props
