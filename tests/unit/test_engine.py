"""Tests for TruthState computation engine — pure functions."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from mnemebrain_core.engine import (
    apply_conflict_policy,
    compute_confidence,
    compute_truth_state,
    effective_weight,
    rank_score,
)
from mnemebrain_core.models import (
    Belief,
    BeliefType,
    ConflictPolicy,
    Evidence,
    Polarity,
    TruthState,
)


def _evidence(
    polarity: Polarity = Polarity.SUPPORTS,
    weight: float = 0.8,
    reliability: float = 0.9,
    valid: bool = True,
    days_old: int = 0,
    time_validity: datetime | None = None,
) -> Evidence:
    """Helper to create Evidence with sensible defaults."""
    return Evidence(
        belief_id=uuid4(),
        source_ref=f"test_{uuid4().hex[:6]}",
        content="test evidence",
        polarity=polarity,
        reliability=reliability,
        weight=weight,
        valid=valid,
        time_validity=time_validity,
        timestamp=datetime.now(timezone.utc) - timedelta(days=days_old),
    )


class TestEffectiveWeight:
    def test_fresh_evidence(self):
        e = _evidence(weight=0.8, reliability=0.9, days_old=0)
        w = effective_weight(e, BeliefType.INFERENCE)
        assert w == pytest.approx(0.8 * 0.9, abs=0.01)

    def test_invalid_evidence_returns_zero(self):
        e = _evidence(valid=False)
        assert effective_weight(e, BeliefType.INFERENCE) == 0.0

    def test_expired_evidence_returns_zero(self):
        e = _evidence(time_validity=datetime.now(timezone.utc) - timedelta(hours=1))
        assert effective_weight(e, BeliefType.INFERENCE) == 0.0

    def test_fact_decays_slowly(self):
        e = _evidence(weight=0.8, reliability=1.0, days_old=365)
        w = effective_weight(e, BeliefType.FACT)
        # After one half-life (365 days), weight should be ~half
        assert w == pytest.approx(0.4, abs=0.05)

    def test_prediction_decays_fast(self):
        e = _evidence(weight=0.8, reliability=1.0, days_old=3)
        w = effective_weight(e, BeliefType.PREDICTION)
        # After one half-life (3 days), weight should be ~half
        assert w == pytest.approx(0.4, abs=0.05)

    def test_inference_decay_30_days(self):
        e = _evidence(weight=0.8, reliability=1.0, days_old=30)
        w = effective_weight(e, BeliefType.INFERENCE)
        assert w == pytest.approx(0.4, abs=0.05)


class TestComputeTruthState:
    def test_no_evidence_returns_neither(self):
        assert compute_truth_state([], BeliefType.INFERENCE) == TruthState.NEITHER

    def test_strong_support_returns_true(self):
        evidence = [_evidence(polarity=Polarity.SUPPORTS, weight=0.8, reliability=0.9)]
        assert compute_truth_state(evidence, BeliefType.INFERENCE) == TruthState.TRUE

    def test_strong_attack_returns_false(self):
        evidence = [_evidence(polarity=Polarity.ATTACKS, weight=0.8, reliability=0.9)]
        assert compute_truth_state(evidence, BeliefType.INFERENCE) == TruthState.FALSE

    def test_both_strong_returns_both(self):
        evidence = [
            _evidence(polarity=Polarity.SUPPORTS, weight=0.8, reliability=0.9),
            _evidence(polarity=Polarity.ATTACKS, weight=0.7, reliability=0.85),
        ]
        assert compute_truth_state(evidence, BeliefType.INFERENCE) == TruthState.BOTH

    def test_weak_evidence_returns_neither(self):
        evidence = [_evidence(polarity=Polarity.SUPPORTS, weight=0.1, reliability=0.2)]
        assert compute_truth_state(evidence, BeliefType.INFERENCE) == TruthState.NEITHER

    def test_invalid_evidence_ignored(self):
        evidence = [
            _evidence(
                polarity=Polarity.SUPPORTS, weight=0.8, reliability=0.9, valid=False
            ),
        ]
        assert compute_truth_state(evidence, BeliefType.INFERENCE) == TruthState.NEITHER

    def test_decayed_prediction_becomes_neither(self):
        evidence = [
            _evidence(
                polarity=Polarity.SUPPORTS, weight=0.5, reliability=0.7, days_old=30
            ),
        ]
        # Prediction half-life is 3 days — after 30 days, ~10 half-lives, weight ~ 0
        assert (
            compute_truth_state(evidence, BeliefType.PREDICTION) == TruthState.NEITHER
        )


class TestComputeConfidence:
    def test_no_evidence_returns_0_5(self):
        assert compute_confidence([], BeliefType.INFERENCE) == 0.5

    def test_strong_support_above_0_5(self):
        evidence = [_evidence(polarity=Polarity.SUPPORTS, weight=0.9, reliability=0.9)]
        c = compute_confidence(evidence, BeliefType.INFERENCE)
        assert c > 0.5

    def test_strong_attack_below_0_5(self):
        evidence = [_evidence(polarity=Polarity.ATTACKS, weight=0.9, reliability=0.9)]
        c = compute_confidence(evidence, BeliefType.INFERENCE)
        assert c < 0.5

    def test_confidence_bounded_0_1(self):
        evidence = [
            _evidence(polarity=Polarity.SUPPORTS, weight=1.0, reliability=1.0)
            for _ in range(20)
        ]
        c = compute_confidence(evidence, BeliefType.INFERENCE)
        assert 0.0 <= c <= 1.0

    def test_balanced_evidence_near_0_5(self):
        evidence = [
            _evidence(polarity=Polarity.SUPPORTS, weight=0.7, reliability=0.8),
            _evidence(polarity=Polarity.ATTACKS, weight=0.7, reliability=0.8),
        ]
        c = compute_confidence(evidence, BeliefType.INFERENCE)
        assert c == pytest.approx(0.5, abs=0.1)


# ---------------------------------------------------------------------------
# rank_score
# ---------------------------------------------------------------------------


class TestRankScore:
    def test_default_alpha(self):
        # alpha=0.7: score = 0.7*sim + 0.3*conf
        score = rank_score(similarity=1.0, confidence=1.0)
        assert score == pytest.approx(1.0)

    def test_pure_similarity(self):
        # confidence=0 → score = alpha * similarity
        score = rank_score(similarity=0.8, confidence=0.0, alpha=0.7)
        assert score == pytest.approx(0.7 * 0.8)

    def test_pure_confidence(self):
        # similarity=0 → score = (1-alpha) * confidence
        score = rank_score(similarity=0.0, confidence=0.6, alpha=0.7)
        assert score == pytest.approx(0.3 * 0.6)

    def test_custom_alpha(self):
        score = rank_score(similarity=0.4, confidence=0.6, alpha=0.5)
        assert score == pytest.approx(0.5 * 0.4 + 0.5 * 0.6)

    def test_alpha_zero_uses_only_confidence(self):
        score = rank_score(similarity=0.9, confidence=0.3, alpha=0.0)
        assert score == pytest.approx(0.3)

    def test_alpha_one_uses_only_similarity(self):
        score = rank_score(similarity=0.9, confidence=0.3, alpha=1.0)
        assert score == pytest.approx(0.9)

    def test_intermediate_values(self):
        score = rank_score(similarity=0.6, confidence=0.8, alpha=0.7)
        expected = 0.7 * 0.6 + 0.3 * 0.8
        assert score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# apply_conflict_policy
# ---------------------------------------------------------------------------


def _belief(truth_state: TruthState) -> Belief:
    """Create a minimal Belief with the given TruthState."""
    b = Belief(claim="test belief", truth_state=truth_state)
    return b


class TestApplyConflictPolicy:
    def test_surface_returns_all(self):
        beliefs = [
            (_belief(TruthState.TRUE), 0.9),
            (_belief(TruthState.BOTH), 0.7),
            (_belief(TruthState.FALSE), 0.3),
        ]
        result = apply_conflict_policy(beliefs, ConflictPolicy.SURFACE)
        assert result == beliefs

    def test_surface_empty_input(self):
        result = apply_conflict_policy([], ConflictPolicy.SURFACE)
        assert result == []

    def test_conservative_removes_both(self):
        b_true = _belief(TruthState.TRUE)
        b_both = _belief(TruthState.BOTH)
        b_false = _belief(TruthState.FALSE)
        beliefs = [(b_true, 0.9), (b_both, 0.7), (b_false, 0.3)]

        result = apply_conflict_policy(beliefs, ConflictPolicy.CONSERVATIVE)

        assert len(result) == 2
        ids = [id(b) for b, _ in result]
        assert id(b_true) in ids
        assert id(b_false) in ids
        assert id(b_both) not in ids

    def test_conservative_all_both_returns_empty(self):
        beliefs = [(_belief(TruthState.BOTH), 0.8), (_belief(TruthState.BOTH), 0.6)]
        result = apply_conflict_policy(beliefs, ConflictPolicy.CONSERVATIVE)
        assert result == []

    def test_conservative_no_both_returns_all(self):
        beliefs = [
            (_belief(TruthState.TRUE), 0.9),
            (_belief(TruthState.NEITHER), 0.5),
        ]
        result = apply_conflict_policy(beliefs, ConflictPolicy.CONSERVATIVE)
        assert len(result) == 2

    def test_optimistic_falls_through_to_return_all(self):
        """OPTIMISTIC has no special handling; falls through to default 'return beliefs'."""
        beliefs = [
            (_belief(TruthState.TRUE), 0.9),
            (_belief(TruthState.BOTH), 0.7),
        ]
        result = apply_conflict_policy(beliefs, ConflictPolicy.OPTIMISTIC)
        assert result == beliefs

    def test_optimistic_empty_input(self):
        result = apply_conflict_policy([], ConflictPolicy.OPTIMISTIC)
        assert result == []

    def test_scores_preserved(self):
        """Scores in the tuples must survive unchanged through all policies."""
        b = _belief(TruthState.TRUE)
        beliefs = [(b, 0.123456)]
        for policy in ConflictPolicy:
            result = apply_conflict_policy(beliefs, policy)
            if result:
                assert result[0][1] == pytest.approx(0.123456)
