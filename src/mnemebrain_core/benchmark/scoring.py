"""Scoring engine for the system benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field

from mnemebrain_core.benchmark.interface import (
    ExplainResult,
    QueryResult,
    RetractResult,
    ReviseResult,
    SandboxResult,
    StoreResult,
)
from mnemebrain_core.benchmark.scenarios.schema import Expectation


@dataclass
class CheckResult:
    name: str
    passed: bool
    expected: object
    actual: object


@dataclass
class ScenarioScore:
    scenario_name: str
    category: str
    checks: list[CheckResult]
    skipped: bool

    def score(self) -> float | None:
        if self.skipped or not self.checks:
            return None
        return sum(1 for c in self.checks if c.passed) / len(self.checks)


@dataclass
class CategoryScore:
    category: str
    score: float | None
    skipped: bool
    scenario_scores: list[ScenarioScore]


def _extract_truth_state(result: object) -> str | None:
    """Extract truth_state from any result type that carries it."""
    if isinstance(result, StoreResult):
        return result.truth_state
    if isinstance(result, ReviseResult):
        return result.truth_state
    if isinstance(result, ExplainResult):
        return result.truth_state
    if isinstance(result, list) and result:
        return result[0].truth_state
    return None


def _extract_confidence(result: object) -> float | None:
    """Extract confidence from any result type that carries it."""
    if isinstance(result, StoreResult):
        return result.confidence
    if isinstance(result, ReviseResult):
        return result.confidence
    if isinstance(result, ExplainResult):
        return result.confidence
    if isinstance(result, list) and result:
        return result[0].confidence
    return None


def evaluate_expectations(
    expectations: list[Expectation],
    action_results: dict[str, object],
) -> list[CheckResult]:
    """Evaluate all expectations against collected action results.

    Each expectation field that is not None produces one CheckResult.
    Returns a flat list of all checks across all expectations.
    """
    checks: list[CheckResult] = []

    for exp in expectations:
        result = action_results.get(exp.action_label)

        if exp.beliefs_stored is not None:
            actual = 1 if isinstance(result, StoreResult) else 0
            checks.append(CheckResult(
                f"{exp.action_label}.beliefs_stored",
                actual >= exp.beliefs_stored,
                exp.beliefs_stored,
                actual,
            ))

        if exp.contradiction_detected is not None and isinstance(result, StoreResult):
            checks.append(CheckResult(
                f"{exp.action_label}.contradiction_detected",
                result.contradiction_detected == exp.contradiction_detected,
                exp.contradiction_detected,
                result.contradiction_detected,
            ))

        if exp.truth_state is not None:
            actual_ts = _extract_truth_state(result)
            checks.append(CheckResult(
                f"{exp.action_label}.truth_state",
                actual_ts == exp.truth_state,
                exp.truth_state,
                actual_ts,
            ))

        if exp.confidence_above is not None:
            actual_conf = _extract_confidence(result)
            passed = actual_conf is not None and actual_conf > exp.confidence_above
            checks.append(CheckResult(
                f"{exp.action_label}.confidence_above",
                passed,
                exp.confidence_above,
                actual_conf,
            ))

        if exp.confidence_below is not None:
            actual_conf = _extract_confidence(result)
            passed = actual_conf is not None and actual_conf < exp.confidence_below
            checks.append(CheckResult(
                f"{exp.action_label}.confidence_below",
                passed,
                exp.confidence_below,
                actual_conf,
            ))

        if exp.merged is not None and isinstance(result, StoreResult):
            checks.append(CheckResult(
                f"{exp.action_label}.merged",
                result.merged == exp.merged,
                exp.merged,
                result.merged,
            ))

        if exp.query_returns_claim is not None:
            actual_has = isinstance(result, list) and len(result) > 0
            checks.append(CheckResult(
                f"{exp.action_label}.query_returns_claim",
                actual_has == exp.query_returns_claim,
                exp.query_returns_claim,
                actual_has,
            ))

        if exp.query_returns_nothing is not None:
            actual_empty = isinstance(result, list) and len(result) == 0
            checks.append(CheckResult(
                f"{exp.action_label}.query_returns_nothing",
                actual_empty == exp.query_returns_nothing,
                exp.query_returns_nothing,
                actual_empty,
            ))

        if exp.explanation_has_evidence is not None and isinstance(result, ExplainResult):
            checks.append(CheckResult(
                f"{exp.action_label}.explanation_has_evidence",
                result.has_evidence == exp.explanation_has_evidence,
                exp.explanation_has_evidence,
                result.has_evidence,
            ))

        if exp.affected_beliefs is not None and isinstance(result, RetractResult):
            checks.append(CheckResult(
                f"{exp.action_label}.affected_beliefs",
                result.affected_beliefs >= exp.affected_beliefs,
                exp.affected_beliefs,
                result.affected_beliefs,
            ))

        if exp.explanation_supporting_count_gte is not None and isinstance(result, ExplainResult):
            checks.append(CheckResult(
                f"{exp.action_label}.explanation_supporting_count_gte",
                result.supporting_count >= exp.explanation_supporting_count_gte,
                exp.explanation_supporting_count_gte,
                result.supporting_count,
            ))

        if exp.explanation_attacking_count_gte is not None and isinstance(result, ExplainResult):
            checks.append(CheckResult(
                f"{exp.action_label}.explanation_attacking_count_gte",
                result.attacking_count >= exp.explanation_attacking_count_gte,
                exp.explanation_attacking_count_gte,
                result.attacking_count,
            ))

        if exp.explanation_expired_count_gte is not None and isinstance(result, ExplainResult):
            expired_count = getattr(result, "expired_count", 0)
            checks.append(CheckResult(
                f"{exp.action_label}.explanation_expired_count_gte",
                expired_count >= exp.explanation_expired_count_gte,
                exp.explanation_expired_count_gte,
                expired_count,
            ))

        if exp.sandbox_resolved_state is not None and isinstance(result, SandboxResult):
            checks.append(CheckResult(
                f"{exp.action_label}.sandbox_resolved_state",
                result.resolved_truth_state == exp.sandbox_resolved_state,
                exp.sandbox_resolved_state,
                result.resolved_truth_state,
            ))

        if exp.sandbox_canonical_unchanged is not None and isinstance(result, SandboxResult):
            checks.append(CheckResult(
                f"{exp.action_label}.sandbox_canonical_unchanged",
                result.canonical_unchanged == exp.sandbox_canonical_unchanged,
                exp.sandbox_canonical_unchanged,
                result.canonical_unchanged,
            ))

        if exp.revision_superseded_count_gte is not None and isinstance(result, ReviseResult):
            checks.append(CheckResult(
                f"{exp.action_label}.revision_superseded_count_gte",
                result.superseded_count >= exp.revision_superseded_count_gte,
                exp.revision_superseded_count_gte,
                result.superseded_count,
            ))

    return checks


def aggregate_by_category(scores: list[ScenarioScore]) -> dict[str, CategoryScore]:
    """Group ScenarioScores by category and compute per-category average scores.

    Categories where all scenarios are skipped receive score=None and skipped=True.
    """
    categories: dict[str, list[ScenarioScore]] = {}
    for s in scores:
        categories.setdefault(s.category, []).append(s)

    result: dict[str, CategoryScore] = {}
    for cat, cat_scores in categories.items():
        non_skipped = [s for s in cat_scores if not s.skipped]
        if not non_skipped:
            result[cat] = CategoryScore(cat, None, True, cat_scores)
        else:
            avg = sum(s.score() for s in non_skipped) / len(non_skipped)
            result[cat] = CategoryScore(cat, avg, False, cat_scores)

    return result
