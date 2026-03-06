"""Truth engine — pure functions for TruthState and confidence computation.

All computation is stateless. Nothing is mutated. Evidence is read, never modified.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from mnemebrain_core.models import (
    ATTACK_THRESHOLD,
    DECAY_HALFLIFE,
    SUPPORT_THRESHOLD,
    BeliefType,
    Evidence,
    Polarity,
    TruthState,
)


def effective_weight(evidence: Evidence, belief_type: BeliefType) -> float:
    """Apply time decay to evidence weight at compute time — not stored.

    Returns 0.0 for invalid or expired evidence.
    Applies exponential decay based on belief type half-life.
    """
    if not evidence.valid:
        return 0.0

    now = datetime.now(timezone.utc)

    if evidence.time_validity and now > evidence.time_validity:
        return 0.0

    halflife_days = DECAY_HALFLIFE[belief_type]
    age_days = (now - evidence.timestamp).total_seconds() / 86400.0
    decay = 0.5 ** (age_days / halflife_days)

    return evidence.weight * evidence.reliability * decay


def compute_truth_state(
    evidence: list[Evidence], belief_type: BeliefType
) -> TruthState:
    """Compute TruthState from evidence ledger — pure function.

    Uses Belnap's four-valued logic:
    - TRUE: sufficient support, insufficient attack
    - FALSE: sufficient attack, insufficient support
    - BOTH: sufficient support AND attack — explicit contradiction
    - NEITHER: insufficient evidence in either direction
    """
    support = sum(
        effective_weight(e, belief_type)
        for e in evidence
        if e.polarity == Polarity.SUPPORTS
    )
    attack = sum(
        effective_weight(e, belief_type)
        for e in evidence
        if e.polarity == Polarity.ATTACKS
    )

    has_support = support >= SUPPORT_THRESHOLD
    has_attack = attack >= ATTACK_THRESHOLD

    if has_support and has_attack:
        return TruthState.BOTH
    if has_support:
        return TruthState.TRUE
    if has_attack:
        return TruthState.FALSE
    return TruthState.NEITHER


def compute_confidence(evidence: list[Evidence], belief_type: BeliefType) -> float:
    """Compute confidence as derived ranking output — log-odds with sigmoid.

    Confidence is used only for ranking, not as ground truth.
    Returns 0.5 when no evidence exists (maximum uncertainty).
    """
    active = [e for e in evidence if e.valid]
    if not active:
        return 0.5

    log_odds = 0.0
    for e in active:
        ew = effective_weight(e, belief_type)
        delta = (ew - 0.5) * 2
        if e.polarity == Polarity.ATTACKS:
            delta = -delta
        log_odds += delta

    return 1.0 / (1.0 + math.exp(-log_odds))
