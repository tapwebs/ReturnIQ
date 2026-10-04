import pytest
from returniq_contracts import ActionType, ReadinessMode, RiskLevel, Stage, is_critical

from returniq_engine.actions import MATRIX, actions_for, recommend


def test_matrix_covers_every_stage_and_band():
    assert len(MATRIX) == len(Stage) * len(RiskLevel)


@pytest.mark.parametrize("key", list(MATRIX))
def test_observation_mode_never_recommends_critical(key):
    stage, level = key
    acts = actions_for(stage, level, ReadinessMode.OBSERVATION)
    assert acts and not any(is_critical(a) for a in acts)


def test_matrix_values_from_spec():
    assert MATRIX[(Stage.PRE_DISPATCH, RiskLevel.MEDIUM)] == (
        ActionType.OTP_VERIFICATION,
        ActionType.UPI_PREPAID_INCENTIVE,
        ActionType.PARTIAL_PREPAID,
    )
    assert MATRIX[(Stage.POST_DELIVERY, RiskLevel.HIGH)] == (
        ActionType.MANUAL_REVIEW,
        ActionType.REFUND_AFTER_QC,
    )
    assert is_critical(ActionType.REFUND_AFTER_QC) and not is_critical(ActionType.MANUAL_REVIEW)


def test_recommend_uses_settings_mode(eng, ds42, settings):
    from .conftest import FAR_FUTURE

    a = eng.assess(ds42, "ord_scn_04", Stage.POST_DELIVERY, FAR_FUTURE, settings, None)
    assert eng.recommend(a, settings)[0] == ActionType.MANUAL_REVIEW
    assert ActionType.REFUND_AFTER_QC in recommend(a, settings)
    obs = settings.model_copy(update={"mode": ReadinessMode.OBSERVATION})
    assert ActionType.REFUND_AFTER_QC not in eng.recommend(a, obs)
