"""Action matrix (A.4). In OBSERVATION mode critical actions are never recommended."""

from __future__ import annotations

from returniq_contracts import (
    ActionType,
    ReadinessMode,
    RiskAssessment,
    RiskLevel,
    Settings,
    Stage,
    is_critical,
)

A = ActionType
MATRIX: dict[tuple[Stage, RiskLevel], tuple[ActionType, ...]] = {
    (Stage.PRE_DISPATCH, RiskLevel.LOW): (A.NORMAL_FULFILMENT,),
    (Stage.PRE_DISPATCH, RiskLevel.MEDIUM): (
        A.OTP_VERIFICATION,
        A.UPI_PREPAID_INCENTIVE,
        A.PARTIAL_PREPAID,
    ),
    (Stage.PRE_DISPATCH, RiskLevel.HIGH): (A.OTP_VERIFICATION, A.PREPAID_ONLY, A.HOLD_FULFILMENT),
    (Stage.PRE_DISPATCH, RiskLevel.INSUFFICIENT_DATA): (A.NORMAL_FULFILMENT,),
    (Stage.POST_DELIVERY, RiskLevel.LOW): (A.INSTANT_RETURN,),
    (Stage.POST_DELIVERY, RiskLevel.MEDIUM): (A.PHOTO_VERIFICATION, A.EXCHANGE_OR_STORE_CREDIT),
    (Stage.POST_DELIVERY, RiskLevel.HIGH): (A.MANUAL_REVIEW, A.REFUND_AFTER_QC),
    (Stage.POST_DELIVERY, RiskLevel.INSUFFICIENT_DATA): (A.INSTANT_RETURN,),
}


def actions_for(stage: Stage, level: RiskLevel, mode: ReadinessMode) -> list[ActionType]:
    """[primary, *alternatives] for a stage/band, dropping critical actions in OBSERVATION."""
    acts = list(MATRIX[(stage, level)])
    if mode == ReadinessMode.OBSERVATION:
        acts = [a for a in acts if not is_critical(a)]
    return acts


def recommend(a: RiskAssessment, s: Settings) -> list[ActionType]:
    """Recommended actions for an assessment, primary first."""
    return actions_for(a.stage, a.risk_level, s.mode)
