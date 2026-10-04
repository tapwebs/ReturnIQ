"""Default Settings (A.3) and constants shared across engine modules."""

from __future__ import annotations

from returniq_contracts import CRITICAL_ACTIONS, Settings

RULE_VERSION = "rules-1.0.0"
ENGINE_VERSION = "0.2.0"
ELIGIBLE_EXCLUDED_REASONS = frozenset({"DEFECTIVE", "DAMAGED_IN_TRANSIT", "WRONG_ITEM"})


def default_settings() -> Settings:
    """Settings with the published A.3 defaults (costs are assumed, in paise)."""
    return Settings(critical_actions=sorted(CRITICAL_ACTIONS, key=lambda a: a.value))
