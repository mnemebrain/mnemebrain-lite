"""Tests for MnemeBrain data models."""
import math
from datetime import datetime, timezone
from uuid import UUID

import pytest

from mnemebrain.models import (
    Belief,
    BeliefType,
    Evidence,
    Polarity,
    TruthState,
)


class TestTruthState:
    def test_enum_values(self):
        assert TruthState.TRUE.value == "true"
        assert TruthState.FALSE.value == "false"
        assert TruthState.BOTH.value == "both"
        assert TruthState.NEITHER.value == "neither"

    def test_all_four_values_exist(self):
        assert len(TruthState) == 4


class TestBeliefType:
    def test_enum_values(self):
        assert BeliefType.FACT.value == "fact"
        assert BeliefType.PREFERENCE.value == "preference"
        assert BeliefType.INFERENCE.value == "inference"
        assert BeliefType.PREDICTION.value == "prediction"


class TestEvidence:
    def test_create_supporting_evidence(self):
        e = Evidence(
            belief_id=None,
            source_ref="msg_12",
            content="User said they love pasta",
            polarity=Polarity.SUPPORTS,
            reliability=0.9,
            weight=0.8,
        )
        assert e.polarity == Polarity.SUPPORTS
        assert e.valid is True
        assert isinstance(e.id, UUID)
        assert isinstance(e.timestamp, datetime)

    def test_create_attacking_evidence(self):
        e = Evidence(
            belief_id=None,
            source_ref="msg_47",
            content="User ordered a burger",
            polarity=Polarity.ATTACKS,
            reliability=0.85,
            weight=0.7,
        )
        assert e.polarity == Polarity.ATTACKS

    def test_weight_bounds(self):
        with pytest.raises(ValueError):
            Evidence(
                belief_id=None,
                source_ref="x",
                content="x",
                polarity=Polarity.SUPPORTS,
                reliability=0.5,
                weight=1.5,  # out of bounds
            )

    def test_reliability_bounds(self):
        with pytest.raises(ValueError):
            Evidence(
                belief_id=None,
                source_ref="x",
                content="x",
                polarity=Polarity.SUPPORTS,
                reliability=-0.1,  # out of bounds
                weight=0.5,
            )

    def test_optional_fields(self):
        e = Evidence(
            belief_id=None,
            source_ref="msg_1",
            content="test",
            polarity=Polarity.SUPPORTS,
            reliability=0.5,
            weight=0.5,
            scope="at work",
        )
        assert e.scope == "at work"
        assert e.time_validity is None


class TestBelief:
    def test_create_belief(self):
        b = Belief(claim="user is vegetarian")
        assert b.claim == "user is vegetarian"
        assert b.belief_type == BeliefType.INFERENCE  # default
        assert b.truth_state == TruthState.NEITHER  # default
        assert b.confidence == 0.5  # default
        assert b.evidence == []
        assert isinstance(b.id, UUID)

    def test_belief_with_type(self):
        b = Belief(claim="user's name is Alice", belief_type=BeliefType.FACT)
        assert b.belief_type == BeliefType.FACT

    def test_belief_with_tags(self):
        b = Belief(claim="user likes pasta", tags=["food", "preference"])
        assert "food" in b.tags

    def test_belief_serialization(self):
        b = Belief(claim="test claim")
        data = b.model_dump()
        assert data["claim"] == "test claim"
        assert data["truth_state"] == "neither"
        assert data["belief_type"] == "inference"
