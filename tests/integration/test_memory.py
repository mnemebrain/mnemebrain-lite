"""Integration tests for BeliefMemory — requires Kuzu DB and embeddings."""

import os
import shutil
import tempfile

import pytest

from mnemebrain_core.memory import BeliefMemory
from mnemebrain_core.models import BeliefType, TruthState
from mnemebrain_core.providers.base import EvidenceInput

try:
    import sentence_transformers  # noqa: F401

    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_EMBEDDINGS, reason="sentence-transformers not installed"),
]


@pytest.fixture
def memory():
    """Create a BeliefMemory with a temporary database."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_db")
    mem = BeliefMemory(db_path=db_path)
    yield mem
    mem.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
class TestBelieve:
    def test_store_and_retrieve_belief(self, memory: BeliefMemory):
        result = memory.believe(
            claim="user is vegetarian",
            evidence_items=[
                EvidenceInput(
                    source_ref="msg_12",
                    content="They said no meat please",
                    polarity="supports",
                    weight=0.8,
                    reliability=0.9,
                )
            ],
        )
        assert result.truth_state == TruthState.TRUE
        assert result.confidence > 0.5
        assert result.conflict is False

    def test_merge_similar_beliefs(self, memory: BeliefMemory):
        r1 = memory.believe(
            claim="user is vegetarian",
            evidence_items=[
                EvidenceInput(
                    source_ref="msg_1",
                    content="no meat",
                    polarity="supports",
                    weight=0.8,
                    reliability=0.9,
                )
            ],
        )
        # Same claim should merge
        r2 = memory.believe(
            claim="the user is a vegetarian",
            evidence_items=[
                EvidenceInput(
                    source_ref="msg_2",
                    content="confirmed vegetarian",
                    polarity="supports",
                    weight=0.7,
                    reliability=0.85,
                )
            ],
        )
        assert r1.id == r2.id  # Merged into same belief

    def test_belief_with_type_and_tags(self, memory: BeliefMemory):
        result = memory.believe(
            claim="user's name is Alice",
            evidence_items=[
                EvidenceInput(
                    source_ref="msg_5",
                    content="introduced as Alice",
                    polarity="supports",
                    weight=0.9,
                    reliability=0.95,
                )
            ],
            belief_type=BeliefType.FACT,
            tags=["identity"],
            source_agent="onboarding",
        )
        assert result.truth_state == TruthState.TRUE


@pytest.mark.integration
class TestExplain:
    def test_explain_existing_belief(self, memory: BeliefMemory):
        memory.believe(
            claim="user likes spicy food",
            evidence_items=[
                EvidenceInput(
                    source_ref="msg_10",
                    content="ordered extra hot sauce",
                    polarity="supports",
                    weight=0.8,
                    reliability=0.9,
                )
            ],
        )
        explanation = memory.explain("user likes spicy food")
        assert explanation is not None
        assert explanation.truth_state == TruthState.TRUE
        assert len(explanation.supporting) == 1
        assert len(explanation.attacking) == 0

    def test_explain_nonexistent_returns_none(self, memory: BeliefMemory):
        result = memory.explain("unknown belief")
        assert result is None


@pytest.mark.integration
class TestRevise:
    def test_revise_adds_evidence(self, memory: BeliefMemory):
        r1 = memory.believe(
            claim="user is vegetarian",
            evidence_items=[
                EvidenceInput(
                    source_ref="msg_1",
                    content="no meat",
                    polarity="supports",
                    weight=0.8,
                    reliability=0.9,
                )
            ],
        )
        r2 = memory.revise(
            belief_id=r1.id,
            new_evidence=EvidenceInput(
                source_ref="msg_50",
                content="confirmed vegetarian",
                polarity="supports",
                weight=0.9,
                reliability=0.95,
            ),
        )
        assert r2.id == r1.id
        assert r2.confidence > r1.confidence  # More supporting evidence

    def test_revise_with_contradicting_evidence(self, memory: BeliefMemory):
        r1 = memory.believe(
            claim="user is vegetarian",
            evidence_items=[
                EvidenceInput(
                    source_ref="msg_1",
                    content="no meat",
                    polarity="supports",
                    weight=0.8,
                    reliability=0.9,
                )
            ],
        )
        r2 = memory.revise(
            belief_id=r1.id,
            new_evidence=EvidenceInput(
                source_ref="msg_60",
                content="ordered a steak",
                polarity="attacks",
                weight=0.85,
                reliability=0.9,
            ),
        )
        assert r2.truth_state == TruthState.BOTH  # Contradiction
        assert r2.conflict is True

    def test_revise_nonexistent_raises(self, memory: BeliefMemory):
        from uuid import uuid4

        with pytest.raises(ValueError, match="not found"):
            memory.revise(
                belief_id=uuid4(),
                new_evidence=EvidenceInput(
                    source_ref="x",
                    content="x",
                    polarity="supports",
                ),
            )


@pytest.mark.integration
class TestRetract:
    def test_retract_removes_evidence_effect(self, memory: BeliefMemory):
        memory.believe(
            claim="user likes pizza",
            evidence_items=[
                EvidenceInput(
                    source_ref="msg_1",
                    content="ordered pizza",
                    polarity="supports",
                    weight=0.8,
                    reliability=0.9,
                )
            ],
        )
        explanation = memory.explain("user likes pizza")
        assert explanation is not None
        evidence_id = explanation.supporting[0].id

        retracted = memory.retract(evidence_id)
        assert len(retracted) >= 1

        # After retraction, the belief should have no active support
        explanation_after = memory.explain("user likes pizza")
        assert explanation_after is not None
        assert len(explanation_after.supporting) == 0
        assert len(explanation_after.expired) == 1

    def test_retract_nonexistent_returns_empty(self, memory: BeliefMemory):
        from uuid import uuid4

        result = memory.retract(uuid4())
        assert result == []
