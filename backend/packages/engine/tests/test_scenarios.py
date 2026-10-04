"""The seven A.6 demo scenarios: expected band and action on seed 42, n=1200."""

import json

import pytest
from returniq_contracts import Stage

from returniq_engine.export_fixtures import SCENARIOS

from .conftest import FAR_FUTURE


@pytest.mark.parametrize("sc", SCENARIOS, ids=[f"scenario_{s[0]}" for s in SCENARIOS])
def test_scenario_band_and_action(eng, ds42, settings, sc):
    num, _, oid, stage, bands, acts, _ = sc
    a = eng.assess(ds42, oid, stage, FAR_FUTURE, settings, None)
    assert a.risk_level.value in bands, (a.risk_score, a.signals)
    offered = [a.recommended_action.value, *[x.value for x in a.alternative_actions]]
    assert a.recommended_action.value in acts
    assert set(offered) <= set(acts)


def test_scenario_1_prepaid_is_low(eng, ds42, settings):
    a = eng.assess(ds42, "ord_scn_01", Stage.PRE_DISPATCH, FAR_FUTURE, settings, None)
    assert a.risk_level.value == "LOW" and a.recommended_action.value == "NORMAL_FULFILMENT"


def test_scenario_3_defect_not_preventable(eng, ds42, settings):
    a = eng.assess(ds42, "ord_scn_03", Stage.POST_DELIVERY, FAR_FUTURE, settings, None)
    assert "SELLER_OR_CARRIER_FAULT_REASON" in {s.code for s in a.signals}
    assert a.risk_level.value == "LOW" and a.recommended_action.value == "INSTANT_RETURN"


def test_scenario_4_includes_refund_after_qc(eng, ds42, settings):
    a = eng.assess(ds42, "ord_scn_04", Stage.POST_DELIVERY, FAR_FUTURE, settings, None)
    assert a.risk_level.value == "HIGH"
    assert "REFUND_AFTER_QC" in [
        a.recommended_action.value,
        *[x.value for x in a.alternative_actions],
    ]
    assert {"TTR_ABNORMALLY_FAST", "RETURN_FREQ_HIGH"} <= {s.code for s in a.signals}


def test_scenario_5_shows_cold_start_signals(eng, ds42, settings):
    a = eng.assess(ds42, "ord_scn_05", Stage.POST_DELIVERY, FAR_FUTURE, settings, None)
    codes = {s.code for s in a.signals}
    assert "NO_PRIOR_HISTORY" in codes
    assert codes & {"COD_HIGH_VALUE_VS_CATEGORY", "HIGH_RETURN_CATEGORY", "HIGH_RETURN_SKU"}
    assert a.risk_level.value in ("LOW", "MEDIUM")


def test_scenario_6_insufficient_and_ttr_unknown(eng, ds42, settings):
    a = eng.assess(ds42, "ord_scn_06", Stage.POST_DELIVERY, FAR_FUTURE, settings, None)
    assert a.risk_level.value == "INSUFFICIENT_DATA" and a.risk_score is None
    assert "delivered_at" in a.missing_fields and a.confidence < settings.min_confidence
    assert a.components.rule is None


def test_scenario_7_next_one_loop(eng, ds42, settings):
    """Same customer as scenario 4; the earlier flagged return raises the Stage 1 risk."""
    s4 = ds42.master.iloc[ds42.order_pos["ord_scn_04"]]["customer_id"]
    s7 = ds42.master.iloc[ds42.order_pos["ord_scn_07"]]["customer_id"]
    assert s4 == s7
    a = eng.assess(ds42, "ord_scn_07", Stage.PRE_DISPATCH, FAR_FUTURE, settings, None)
    assert a.risk_level.value in ("MEDIUM", "HIGH")
    assert "FLAGGED_PRIOR_RETURN" in {s.code for s in a.signals}
    # without the earlier flag the same order would score lower
    no_flag = sum(s.contribution for s in a.signals if s.code != "FLAGGED_PRIOR_RETURN")
    assert a.risk_score > no_flag


def test_scenarios_json_matches_engine(tmp_path):
    from returniq_engine.export_fixtures import build_fixtures

    files = build_fixtures(42, 1200, include_model=False)
    data = json.loads(files["scenarios.json"])
    assert len(data["scenarios"]) == 7
    for row in data["scenarios"]:
        assert row["actual_band"] in row["expected_bands"]


def test_insufficient_data_share_under_15_percent(eng, ds42, settings):
    for stage in (Stage.PRE_DISPATCH, Stage.POST_DELIVERY):
        batch = eng.assess_batch(ds42, stage, settings, None)
        share = sum(a.risk_level.value == "INSUFFICIENT_DATA" for a in batch) / len(batch)
        print(f"{stage.value} INSUFFICIENT_DATA share = {share:.4f}")
        assert share < 0.15


def test_high_share_at_most_about_20_percent(eng, ds42, settings):
    batch = eng.assess_batch(ds42, Stage.PRE_DISPATCH, settings, None)
    share = sum(a.risk_level.value == "HIGH" for a in batch) / len(batch)
    print(f"Stage 1 HIGH share = {share:.4f}")
    assert share <= 0.20
