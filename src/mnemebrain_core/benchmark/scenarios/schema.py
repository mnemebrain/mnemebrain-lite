"""Scenario data structures for the system benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field

VALID_ACTION_TYPES = {"store", "retract", "query", "explain", "wait_days", "revise", "sandbox_fork", "sandbox_assume", "sandbox_resolve", "sandbox_discard", "add_attack"}
VALID_CATEGORIES = {"contradiction", "retraction", "decay", "dedup", "extraction", "lifecycle", "belief_revision", "evidence_tracking", "temporal", "counterfactual"}


@dataclass
class Action:
    label: str
    type: str
    claim: str | None = None
    evidence: list[dict] | None = None
    belief_type: str | None = None
    target_label: str | None = None
    wait_days: int | None = None
    # Sandbox actions
    scenario_label: str | None = None
    sandbox_label: str | None = None
    truth_state_override: str | None = None
    belief_label: str | None = None
    # Attack actions
    attacker_label: str | None = None
    attack_type: str | None = None
    weight: float | None = None


@dataclass
class Expectation:
    action_label: str
    beliefs_stored: int | None = None
    merged: bool | None = None
    contradiction_detected: bool | None = None
    truth_state: str | None = None
    query_returns_claim: bool | None = None
    query_returns_nothing: bool | None = None
    explanation_has_evidence: bool | None = None
    confidence_above: float | None = None
    confidence_below: float | None = None
    affected_beliefs: int | None = None
    # Evidence tracking expectations
    explanation_supporting_count_gte: int | None = None
    explanation_attacking_count_gte: int | None = None
    explanation_expired_count_gte: int | None = None
    # Sandbox expectations
    sandbox_resolved_state: str | None = None
    sandbox_canonical_unchanged: bool | None = None
    # Revision expectations
    revision_superseded_count_gte: int | None = None


@dataclass
class Scenario:
    name: str
    description: str
    category: str
    requires: list[str]
    actions: list[Action]
    expectations: list[Expectation]
