"""MemorySystem interface for the system benchmark."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class Capability(str, Enum):
    STORE = "store"
    QUERY = "query"
    RETRACT = "retract"
    EXPLAIN = "explain"
    CONTRADICTION = "contradiction"
    DECAY = "decay"
    REVISE = "revise"
    SANDBOX = "sandbox"
    ATTACK = "attack"


@dataclass
class StoreResult:
    belief_id: str
    merged: bool
    contradiction_detected: bool
    truth_state: str | None
    confidence: float | None


@dataclass
class QueryResult:
    belief_id: str
    claim: str
    confidence: float | None
    truth_state: str | None


@dataclass
class RetractResult:
    affected_beliefs: int
    truth_states_changed: int


@dataclass
class ExplainResult:
    claim: str
    has_evidence: bool
    supporting_count: int
    attacking_count: int
    truth_state: str | None
    confidence: float | None
    expired_count: int = 0


@dataclass
class ReviseResult:
    belief_id: str
    truth_state: str | None
    confidence: float | None
    superseded_count: int


@dataclass
class SandboxResult:
    sandbox_id: str
    resolved_truth_state: str | None = None
    canonical_unchanged: bool = True


@dataclass
class AttackResult:
    edge_id: str
    attacker_id: str
    target_id: str


class MemorySystem(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def capabilities(self) -> set[Capability]: ...

    @abstractmethod
    def store(self, claim: str, evidence: list[dict]) -> StoreResult: ...

    @abstractmethod
    def query(self, claim: str) -> list[QueryResult]: ...

    def retract(self, evidence_id: str) -> RetractResult:
        raise NotImplementedError

    def explain(self, claim: str) -> ExplainResult:
        raise NotImplementedError

    def set_time_offset_days(self, days: int) -> None:
        raise NotImplementedError

    def revise(self, belief_id: str, evidence: list[dict]) -> ReviseResult:
        raise NotImplementedError

    def sandbox_fork(self, scenario_label: str = "") -> SandboxResult:
        raise NotImplementedError

    def sandbox_assume(self, sandbox_id: str, belief_id: str, truth_state: str) -> SandboxResult:
        raise NotImplementedError

    def sandbox_resolve(self, sandbox_id: str, belief_id: str) -> SandboxResult:
        raise NotImplementedError

    def sandbox_discard(self, sandbox_id: str) -> None:
        raise NotImplementedError

    def add_attack(self, attacker_id: str, target_id: str, attack_type: str, weight: float) -> AttackResult:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None: ...
