"""Pydantic v2 contract models. Money is integer paise, keys are snake_case."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from .enums import (
    ActionStatus,
    ActionType,
    AgentProvider,
    AgentRunStatus,
    AgentStepStatus,
    Category,
    DeliveryStatus,
    JobStatus,
    OutcomeResult,
    PaymentMode,
    Provenance,
    QcResult,
    ReadinessMode,
    RequestedBy,
    ReturnReason,
    ReturnStatus,
    RiskLevel,
    ScoringMethod,
    SignalSource,
    Stage,
    TtrBucket,
)


class Contract(BaseModel):
    """Base class: unknown fields are rejected so drift is caught early."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)


T = TypeVar("T")


# --------------------------------------------------------------------------- envelope
class ResponseMeta(Contract):
    request_id: str | None = None
    provenance: Provenance | None = None
    generated_at: datetime | None = None
    contract_version: str | None = None
    next_cursor: str | None = None


class ApiError(Contract):
    code: str
    message: str
    request_id: str | None = None
    field_errors: dict[str, str] = Field(default_factory=dict)


class Envelope(Contract, Generic[T]):
    """`{success, data, error, meta}` wrapper used by every HTTP response."""

    success: bool
    data: T | None = None
    error: ApiError | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


# --------------------------------------------------------------------------- filters etc.
class Filters(Contract):
    dataset_id: str | None = None
    from_: datetime | None = Field(default=None, alias="from")
    to: datetime | None = None
    stage: Stage | None = None
    payment_mode: PaymentMode | None = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class DatasetInfo(Contract):
    dataset_id: str
    dataset_version: str
    provenance: Provenance
    orders: int
    shipments: int
    returns: int
    first_order_at: datetime | None = None
    last_order_at: datetime | None = None


class Job(Contract):
    id: str
    status: JobStatus
    dataset_id: str | None = None
    accepted: int = 0
    rejected: int = 0
    deduplicated: int = 0
    missing_fields: list[str] = Field(default_factory=list)
    errors_url: str | None = None


# --------------------------------------------------------------------------- risk
class Signal(Contract):
    code: str
    label: str
    evidence: str
    source: SignalSource
    contribution: float


class Components(Contract):
    rule: float | None = None
    ml: float | None = None
    anomaly: float | None = None


class RiskAssessment(Contract):
    id: str
    order_id: str
    customer_id: str
    stage: Stage
    as_of: datetime
    risk_score: int | None = Field(default=None, ge=0, le=100)
    risk_level: RiskLevel
    confidence: float = Field(ge=0, le=1)
    components: Components
    weights: Components
    signals: list[Signal] = Field(default_factory=list)
    reason: str
    recommended_action: ActionType
    alternative_actions: list[ActionType] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    scoring_method: ScoringMethod
    rule_version: str
    model_version: str | None = None
    provenance: Provenance
    latency_ms: float = 0.0


# --------------------------------------------------------------------------- orders
class Order(Contract):
    order_id: str
    customer_id: str
    order_date: datetime
    sku_id: str
    category: Category | None = None
    quantity: int = Field(ge=1)
    payment_mode: PaymentMode
    order_amount_paise: int = Field(ge=0)
    pincode: str | None = Field(default=None, pattern=r"^\d{6}$")
    delivery_status: DeliveryStatus | None = None
    return_status: ReturnStatus | None = None
    risk_score: int | None = None
    risk_level: RiskLevel | None = None


class ShipmentRecord(Contract):
    order_id: str
    shipped_at: datetime | None = None
    delivered_at: datetime | None = None
    delivery_status: DeliveryStatus
    delivery_attempts: int | None = None


class ReturnRecord(Contract):
    return_id: str
    order_id: str
    return_requested_at: datetime
    return_accepted_at: datetime | None = None
    return_reason: ReturnReason
    return_status: ReturnStatus
    qc_result: QcResult | None = None
    ttr_h: float | None = None
    ttr_bucket: TtrBucket = TtrBucket.UNKNOWN


class TimelineEvent(Contract):
    at: datetime
    kind: str
    detail: str


class OrderDetail(Contract):
    order: Order
    shipment: ShipmentRecord | None = None
    return_: ReturnRecord | None = Field(default=None, alias="return")
    prior_orders: list[Order] = Field(default_factory=list)
    events: list[TimelineEvent] = Field(default_factory=list)
    assessments: list[RiskAssessment] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class OrdersPage(Contract):
    items: list[Order]
    next_cursor: str | None = None


# --------------------------------------------------------------------------- actions
class ActionTarget(Contract):
    order_id: str | None = None
    return_id: str | None = None
    customer_id: str | None = None
    pincode: str | None = None


class ActionExecution(Contract):
    execution_id: str | None = None
    result: str | None = None
    error: str | None = None


class AuditEntry(Contract):
    at: datetime
    actor: str
    event: str


class Action(Contract):
    id: str
    type: ActionType
    stage: Stage | None = None
    target: ActionTarget
    assessment_id: str | None = None
    critical: bool
    requested_by: RequestedBy
    agent_run_id: str | None = None
    summary: str
    evidence: list[str] = Field(default_factory=list)
    customer_impact: str = ""
    estimated_cost_paise: int = 0
    estimated_impact_paise: int = 0
    status: ActionStatus
    version: int = 1
    expires_at: datetime | None = None
    execution: ActionExecution = Field(default_factory=ActionExecution)
    outcome: OutcomeResult | None = None
    audit: list[AuditEntry] = Field(default_factory=list)


class Outcome(Contract):
    """Outcome record posted after an executed action (feedback loop)."""

    action_id: str
    order_id: str | None = None
    outcome: OutcomeResult
    note: str | None = None
    recorded_at: datetime | None = None


# --------------------------------------------------------------------------- agent
class AgentStep(Contract):
    idx: int
    tool: str
    input: dict[str, Any] = Field(default_factory=dict)
    output_summary: str = ""
    status: AgentStepStatus
    ms: int = 0


class AgentRun(Contract):
    id: str
    task: str
    context: dict[str, Any] = Field(default_factory=dict)
    provider: AgentProvider
    status: AgentRunStatus
    steps: list[AgentStep] = Field(default_factory=list)
    proposals: list[str] = Field(default_factory=list)
    summary: str = ""
    error: str | None = None


# --------------------------------------------------------------------------- readiness
class Readiness(Contract):
    mode: ReadinessMode
    days_observed: float
    orders: int
    completed_return_windows: int
    timestamp_completeness: float = Field(ge=0, le=1)
    ready: bool
    reason: str


# --------------------------------------------------------------------------- loss
class LossBreakdown(Contract):
    forward_ship_paise: int = 0
    reverse_ship_paise: int = 0
    packaging_paise: int = 0
    handling_paise: int = 0
    writeoff_paise: int = 0
    cost_recovery_paise: int = 0


class LossSummary(Contract):
    """Observed operating loss on accepted post-delivery returns (order value excluded)."""

    observed_paise: int
    breakdown: LossBreakdown
    assumed_fields: list[str] = Field(default_factory=list)
    rto_observed_paise: int = 0
    returns_counted: int = 0


class ScenarioAmounts(Contract):
    low: int
    base: int
    high: int


class LossEstimate(Contract):
    observed: LossSummary
    eligible_loss_paise: int
    flagged_eligible_returns: int
    intervention_cost_paise: int
    friction_margin_loss_paise: int
    net_avoidable: ScenarioAmounts
    loss_avoided_per_100_paise: ScenarioAmounts
    effectiveness: Effectiveness
    assumed_fields: list[str] = Field(default_factory=list)
    note: str = "Scenario estimate on assumed costs, not realised savings."


# --------------------------------------------------------------------------- analytics
class RtoMetric(Contract):
    count: int
    rate: float | None
    denominator: int
    pending: int = 0


class ReturnsMetric(Contract):
    requested: int
    accepted: int
    rate: float | None
    denominator: int
    requested_rate: float | None = None


class TtrBucketCount(Contract):
    bucket: TtrBucket
    count: int


class TtrSummary(Contract):
    median_h: float | None
    buckets: list[TtrBucketCount]


class SegmentRow(Contract):
    key: str
    orders: int
    returns: int
    rto: int
    return_rate: float | None
    rto_rate: float | None
    loss_paise: int = 0


class ReasonRow(Contract):
    reason: ReturnReason
    count: int
    share: float
    preventable_eligible: bool


class Analytics(Contract):
    period_from: datetime | None = None
    period_to: datetime | None = None
    orders: int
    dispatched: int
    delivered: int
    rto: RtoMetric
    returns: ReturnsMetric
    flagged_returns: int
    estimated_prr: float | None
    confirmed_prr: float | None = None
    loss: LossSummary
    est_net_avoidable: ScenarioAmounts
    loss_avoided_per_100_paise: int
    ttr: TtrSummary
    top_segments: list[SegmentRow] = Field(default_factory=list)
    top_pincodes: list[SegmentRow] = Field(default_factory=list)
    top_skus: list[SegmentRow] = Field(default_factory=list)
    reason_breakdown: list[ReasonRow] = Field(default_factory=list)
    repeat_returners: int = 0
    readiness: Readiness
    scoring_method: ScoringMethod = ScoringMethod.RULES
    estimate_note: str = "Estimated PRR is the share of returns the system would have flagged."


class CohortRow(Contract):
    cohort: str
    returns: int
    median_ttr_h: float | None
    return_rate: float | None
    loss_paise: int


class WaterfallStep(Contract):
    label: str
    amount_paise: int
    assumed: bool = False


class Timeline(Contract):
    ttr_buckets: list[TtrBucketCount]
    cohort_compare: list[CohortRow]
    loss_waterfall: list[WaterfallStep]
    median_ttr_h: float | None = None


# --------------------------------------------------------------------------- policy, models
class PolicySimulation(Contract):
    top_pct: float
    stage: Stage
    orders_targeted: int
    preventable_captured_pct: float | None
    returns_captured_pct: float | None
    est_net_avoidable_paise: int
    false_positive_estimate: float | None
    note: str = "Estimate on observed outcomes; not a guarantee."


class BandRow(Contract):
    band: RiskLevel
    orders: int
    pct_orders: float
    observed_rate: float | None
    lift_vs_overall: float | None


class CalibrationBin(Contract):
    lower: float
    upper: float
    count: int
    mean_predicted: float | None
    observed_rate: float | None


class CalibrationReport(Contract):
    bins: list[CalibrationBin]
    brier: float | None
    label: str


class ModelMetrics(Contract):
    auc: float | None
    rules_auc: float | None = None
    precision_at_12: float | None
    recall_at_12: float | None
    preventable_captured_pct: float | None


class ModelInfo(Contract):
    id: str
    stage: Stage
    trained_at: datetime
    rows: int
    metrics: ModelMetrics
    split: str = "TIME_BASED"
    provenance: Provenance
    scoring_method: ScoringMethod = ScoringMethod.RULES_AND_MODEL
    band_table: list[BandRow] = Field(default_factory=list)
    calibration: CalibrationReport | None = None
    feature_names: list[str] = Field(default_factory=list)
    note: str = "simulated, on planted synthetic patterns"


# --------------------------------------------------------------------------- settings
class Thresholds(Contract):
    medium: int = 40
    high: int = 70


class Weights(Contract):
    rule: float = 0.4
    ml: float = 0.4
    anomaly: float = 0.2


class CostDefaults(Contract):
    forward_ship_paise: int = 6000
    reverse_ship_paise: int = 8000
    packaging_paise: int = 1500
    handling_paise: int = 2500
    writeoff_pct: float = 10.0
    intervention_cost_paise: int = 1000
    friction_margin_loss_pct: float = 2.0


class Effectiveness(Contract):
    low: float = 0.2
    base: float = 0.4
    high: float = 0.6


class LlmInfo(Contract):
    provider: AgentProvider = AgentProvider.PLAYBOOK
    model: str | None = None
    configured: bool = False


class Settings(Contract):
    thresholds: Thresholds = Field(default_factory=Thresholds)
    weights: Weights = Field(default_factory=Weights)
    cost_defaults_paise: CostDefaults = Field(default_factory=CostDefaults)
    return_window_days: int = 7
    min_confidence: float = 0.3
    effectiveness_assumption: Effectiveness = Field(default_factory=Effectiveness)
    critical_actions: list[ActionType] = Field(default_factory=list)
    mode: ReadinessMode = ReadinessMode.ACTIVE
    anomaly_enabled: bool = False
    llm: LlmInfo = Field(default_factory=LlmInfo)


# --------------------------------------------------------------------------- dna + preview
class ReturnDNA(Contract):
    customer_id: str
    as_of: datetime | None = None
    orders: int
    delivered: int
    rto: int
    failed_deliveries: int
    returns: int
    eligible_returns: int
    return_rate: float | None
    median_ttr_h: float | None
    fast_return_share: float | None
    flagged_returns: int
    categories: dict[str, int] = Field(default_factory=dict)
    summary: str = ""


class PreviewOrder(Contract):
    order_id: str = "ord_preview"
    customer_id: str = "cus_preview"
    order_date: datetime
    sku_id: str = "sku_preview"
    category: Category | None = None
    quantity: int = Field(default=1, ge=1)
    payment_mode: PaymentMode
    order_amount_paise: int = Field(ge=0)
    pincode: str | None = Field(default=None, pattern=r"^\d{6}$")


class PreviewHistoryItem(Contract):
    order_date: datetime
    category: Category | None = None
    payment_mode: PaymentMode = PaymentMode.PREPAID
    order_amount_paise: int = Field(default=0, ge=0)
    delivery_status: DeliveryStatus = DeliveryStatus.DELIVERED
    delivered_at: datetime | None = None
    return_requested_at: datetime | None = None
    return_reason: ReturnReason | None = None


class PreviewReturnRequest(Contract):
    return_requested_at: datetime
    delivered_at: datetime | None = None
    return_reason: ReturnReason


class ScorePreviewRequest(Contract):
    order: PreviewOrder
    history: list[PreviewHistoryItem] = Field(default_factory=list)
    return_request: PreviewReturnRequest | None = None


class ScorePreviewResponse(Contract):
    pre_dispatch: RiskAssessment
    post_delivery: RiskAssessment | None = None
