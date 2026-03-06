"""Core data models for MnemeBrain belief memory system."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class TruthState(str, Enum):
    """Four-valued logic (Belnap) for belief truth representation."""

    TRUE = "true"
    FALSE = "false"
    BOTH = "both"
    NEITHER = "neither"


class BeliefType(str, Enum):
    """Belief category — drives decay rate and confidence prior."""

    FACT = "fact"
    PREFERENCE = "preference"
    INFERENCE = "inference"
    PREDICTION = "prediction"


class Polarity(str, Enum):
    """Evidence polarity — replaces ATTACKS graph edges."""

    SUPPORTS = "supports"
    ATTACKS = "attacks"


# Decay half-lives in days per belief type
DECAY_HALFLIFE: dict[BeliefType, float] = {
    BeliefType.FACT: 365.0,
    BeliefType.PREFERENCE: 90.0,
    BeliefType.INFERENCE: 30.0,
    BeliefType.PREDICTION: 3.0,
}

# Thresholds for TruthState computation
SUPPORT_THRESHOLD: float = 0.3
ATTACK_THRESHOLD: float = 0.3


class Evidence(BaseModel):
    """A single piece of evidence in the append-only ledger."""

    id: UUID = Field(default_factory=uuid4)
    belief_id: UUID | None = None
    source_ref: str
    content: str
    polarity: Polarity
    reliability: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    time_validity: datetime | None = None
    scope: str | None = None
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    valid: bool = True

    model_config = {"frozen": False}


class Belief(BaseModel):
    """A belief with its evidence ledger and computed truth state."""

    id: UUID = Field(default_factory=uuid4)
    claim: str = ""
    belief_type: BeliefType = BeliefType.INFERENCE
    truth_state: TruthState = TruthState.NEITHER
    confidence: float = 0.5
    tags: list[str] = Field(default_factory=list)
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    source_agent: str = ""
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_revised: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    evidence: list[Evidence] = Field(default_factory=list)

    model_config = {"frozen": False}
