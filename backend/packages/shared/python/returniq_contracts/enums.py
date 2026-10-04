"""Shared enums. All values are UPPER_SNAKE strings."""

from __future__ import annotations

from enum import StrEnum

_StrEnum = StrEnum


class Stage(_StrEnum):
    PRE_DISPATCH = "PRE_DISPATCH"
    POST_DELIVERY = "POST_DELIVERY"


class RiskLevel(_StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ActionType(_StrEnum):
    NORMAL_FULFILMENT = "NORMAL_FULFILMENT"
    OTP_VERIFICATION = "OTP_VERIFICATION"
    UPI_PREPAID_INCENTIVE = "UPI_PREPAID_INCENTIVE"
    PARTIAL_PREPAID = "PARTIAL_PREPAID"
    PREPAID_ONLY = "PREPAID_ONLY"
    HOLD_FULFILMENT = "HOLD_FULFILMENT"
    SELLER_RESTRICTION = "SELLER_RESTRICTION"
    INSTANT_RETURN = "INSTANT_RETURN"
    PHOTO_VERIFICATION = "PHOTO_VERIFICATION"
    EXCHANGE_OR_STORE_CREDIT = "EXCHANGE_OR_STORE_CREDIT"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    REFUND_AFTER_QC = "REFUND_AFTER_QC"
    RESTRICT_RETURN_METHOD = "RESTRICT_RETURN_METHOD"
    RESTRICT_COD_PINCODE = "RESTRICT_COD_PINCODE"
    CHANGE_POLICY_THRESHOLD = "CHANGE_POLICY_THRESHOLD"


CRITICAL_ACTIONS: frozenset[ActionType] = frozenset(
    {
        ActionType.PREPAID_ONLY,
        ActionType.HOLD_FULFILMENT,
        ActionType.SELLER_RESTRICTION,
        ActionType.REFUND_AFTER_QC,
        ActionType.RESTRICT_RETURN_METHOD,
        ActionType.RESTRICT_COD_PINCODE,
        ActionType.CHANGE_POLICY_THRESHOLD,
    }
)


def is_critical(action: ActionType) -> bool:
    """Return True when the action needs explicit human approval."""
    return action in CRITICAL_ACTIONS


class Provenance(_StrEnum):
    SYNTHETIC = "SYNTHETIC"
    SELLER_UPLOADED = "SELLER_UPLOADED"
    LIVE = "LIVE"


class ScoringMethod(_StrEnum):
    RULES = "RULES"
    RULES_AND_MODEL = "RULES_AND_MODEL"


class SignalSource(_StrEnum):
    RULE = "RULE"
    ML = "ML"
    ANOMALY = "ANOMALY"


class PaymentMode(_StrEnum):
    COD = "COD"
    PREPAID = "PREPAID"
    PARTIAL_COD = "PARTIAL_COD"


class Category(_StrEnum):
    FASHION = "FASHION"
    BEAUTY = "BEAUTY"
    ELECTRONICS = "ELECTRONICS"
    LIFESTYLE = "LIFESTYLE"
    OTHER = "OTHER"


class DeliveryStatus(_StrEnum):
    DELIVERED = "DELIVERED"
    RTO = "RTO"
    IN_TRANSIT = "IN_TRANSIT"
    NDR = "NDR"
    LOST = "LOST"


class ReturnReason(_StrEnum):
    SIZE_FIT = "SIZE_FIT"
    CHANGED_MIND = "CHANGED_MIND"
    NOT_AS_DESCRIBED = "NOT_AS_DESCRIBED"
    DEFECTIVE = "DEFECTIVE"
    DAMAGED_IN_TRANSIT = "DAMAGED_IN_TRANSIT"
    WRONG_ITEM = "WRONG_ITEM"
    OTHER = "OTHER"


class ReturnStatus(_StrEnum):
    REQUESTED = "REQUESTED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    REFUNDED = "REFUNDED"


class QcResult(_StrEnum):
    UNUSED = "UNUSED"
    USED = "USED"
    DAMAGED_BY_CUSTOMER = "DAMAGED_BY_CUSTOMER"
    UNKNOWN = "UNKNOWN"


class TtrBucket(_StrEnum):
    LT_6H = "LT_6H"
    H6_24 = "H6_24"
    D1_3 = "D1_3"
    D3_7 = "D3_7"
    GT_7D = "GT_7D"
    UNKNOWN = "UNKNOWN"


class ActionStatus(_StrEnum):
    PROPOSED = "PROPOSED"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    EXECUTING = "EXECUTING"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"


class RequestedBy(_StrEnum):
    AGENT = "AGENT"
    USER = "USER"


class OutcomeResult(_StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    OVERRIDDEN = "OVERRIDDEN"


class AgentRunStatus(_StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    EXECUTING = "EXECUTING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class AgentProvider(_StrEnum):
    GEMINI = "GEMINI"
    GROQ = "GROQ"
    PLAYBOOK = "PLAYBOOK"


class AgentStepStatus(_StrEnum):
    DONE = "DONE"
    ERROR = "ERROR"


class JobStatus(_StrEnum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ReadinessMode(_StrEnum):
    OBSERVATION = "OBSERVATION"
    ACTIVE = "ACTIVE"
