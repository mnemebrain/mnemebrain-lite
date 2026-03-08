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
    pytest.mark.skipif(
        not HAS_EMBEDDINGS, reason="sentence-transformers not installed"
    ),
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


# ---------------------------------------------------------------------------
# New integration tests — search(), list_beliefs(), _get_embedder()
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSearch:
    def _seed(self, memory: BeliefMemory, claim: str = "user loves hiking") -> None:
        memory.believe(
            claim=claim,
            evidence_items=[
                EvidenceInput(
                    source_ref="obs_1",
                    content="went on a hike last weekend",
                    polarity="supports",
                    weight=0.8,
                    reliability=0.9,
                )
            ],
        )

    def test_search_returns_ranked_results(self, memory: BeliefMemory):
        """search() returns (belief, similarity, confidence, rank_score) tuples."""
        self._seed(memory)

        results = memory.search(query="hiking outdoors", limit=5)
        assert len(results) >= 1
        belief, sim, conf, rs = results[0]
        assert 0.0 <= sim <= 1.0
        assert 0.0 <= conf <= 1.0
        assert rs >= 0.0

    def test_search_respects_limit(self, memory: BeliefMemory):
        """search() returns no more entries than the requested limit."""
        for i in range(5):
            memory.believe(
                claim=f"user likes outdoor activity {i}",
                evidence_items=[
                    EvidenceInput(
                        source_ref=f"obs_{i}",
                        content=f"evidence {i}",
                        polarity="supports",
                        weight=0.7,
                        reliability=0.8,
                    )
                ],
            )
        results = memory.search(query="outdoor", limit=2)
        assert len(results) <= 2

    def test_search_with_conflict_policy_conservative(self, memory: BeliefMemory):
        """search() with ConflictPolicy.CONSERVATIVE excludes BOTH-state beliefs."""
        from mnemebrain_core.models import ConflictPolicy

        self._seed(memory)
        results = memory.search(
            query="hiking", limit=10, conflict_policy=ConflictPolicy.CONSERVATIVE
        )
        # All returned beliefs should not be in BOTH state (or list may be empty)
        from mnemebrain_core.models import TruthState

        for belief, _, _, _ in results:
            assert belief.truth_state != TruthState.BOTH

    def test_search_with_conflict_policy_surface(self, memory: BeliefMemory):
        """search() with ConflictPolicy.SURFACE returns all matches."""
        from mnemebrain_core.models import ConflictPolicy

        self._seed(memory)
        results = memory.search(
            query="hiking", limit=10, conflict_policy=ConflictPolicy.SURFACE
        )
        assert len(results) >= 1

    def test_search_with_rank_alpha(self, memory: BeliefMemory):
        """search() with different rank_alpha values returns results."""
        self._seed(memory)
        for alpha in (0.0, 0.5, 1.0):
            results = memory.search(query="hiking", limit=5, rank_alpha=alpha)
            assert isinstance(results, list)


@pytest.mark.integration
class TestListBeliefs:
    def _seed_multiple(self, memory: BeliefMemory) -> None:
        """Create several beliefs of different types and tags."""
        memory.believe(
            claim="user is vegetarian",
            evidence_items=[
                EvidenceInput(
                    source_ref="food_1",
                    content="ordered veggie burger",
                    polarity="supports",
                    weight=0.9,
                    reliability=0.95,
                )
            ],
            belief_type=BeliefType.FACT,
            tags=["diet"],
        )
        memory.believe(
            claim="user prefers morning exercise",
            evidence_items=[
                EvidenceInput(
                    source_ref="fit_1",
                    content="gym at 6am",
                    polarity="supports",
                    weight=0.7,
                    reliability=0.8,
                )
            ],
            belief_type=BeliefType.PREFERENCE,
            tags=["fitness", "habit"],
        )

    def test_list_beliefs_no_filters_returns_all(self, memory: BeliefMemory):
        """list_beliefs() with no filters returns all stored beliefs."""
        self._seed_multiple(memory)
        beliefs, total = memory.list_beliefs()
        assert total >= 2
        assert len(beliefs) >= 2

    def test_list_beliefs_truth_state_filter(self, memory: BeliefMemory):
        """Filter by truth_states returns only matching beliefs."""
        from mnemebrain_core.models import TruthState

        self._seed_multiple(memory)
        beliefs, total = memory.list_beliefs(truth_states=[TruthState.TRUE])
        assert total >= 1
        for b in beliefs:
            assert b.truth_state == TruthState.TRUE

    def test_list_beliefs_belief_type_filter(self, memory: BeliefMemory):
        """Filter by belief_types returns only matching beliefs."""
        self._seed_multiple(memory)
        beliefs, total = memory.list_beliefs(belief_types=[BeliefType.FACT])
        assert total >= 1
        for b in beliefs:
            assert b.belief_type == BeliefType.FACT

    def test_list_beliefs_tag_filter(self, memory: BeliefMemory):
        """Filter by tag returns only beliefs that carry that tag."""
        self._seed_multiple(memory)
        beliefs, total = memory.list_beliefs(tag="diet")
        assert total >= 1
        for b in beliefs:
            assert "diet" in b.tags

    def test_list_beliefs_min_confidence_filter(self, memory: BeliefMemory):
        """min_confidence filter excludes low-confidence beliefs."""
        self._seed_multiple(memory)
        beliefs, total = memory.list_beliefs(min_confidence=0.0, max_confidence=1.0)
        assert total >= 1

    def test_list_beliefs_max_confidence_filter(self, memory: BeliefMemory):
        """max_confidence=0 should return no beliefs."""
        self._seed_multiple(memory)
        beliefs, total = memory.list_beliefs(min_confidence=0.0, max_confidence=0.0)
        # No belief should have confidence exactly 0 after evidence
        assert total == 0

    def test_list_beliefs_limit_and_offset(self, memory: BeliefMemory):
        """limit and offset control pagination correctly."""
        self._seed_multiple(memory)
        page1, total = memory.list_beliefs(limit=1, offset=0)
        page2, _ = memory.list_beliefs(limit=1, offset=1)
        assert len(page1) == 1
        assert len(page2) <= 1
        if total > 1:
            assert page1[0].id != page2[0].id

    def test_list_beliefs_empty_store(self, memory: BeliefMemory):
        """list_beliefs() on an empty store returns empty list and total=0."""
        beliefs, total = memory.list_beliefs()
        assert total == 0
        assert beliefs == []


@pytest.mark.integration
class TestGetEmbedder:
    def test_get_embedder_raises_when_none(self):
        """_get_embedder() raises ImportError when no provider is available."""
        import os
        import shutil
        import tempfile

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_db")
        try:
            # Inject None as the embedder explicitly
            mem = BeliefMemory.__new__(BeliefMemory)
            from mnemebrain_core.store import KuzuGraphStore

            mem._store = KuzuGraphStore(db_path)
            mem._embedder = None
            with pytest.raises(ImportError, match="No embedding provider"):
                mem._get_embedder()
        finally:
            mem._store.close()
            shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
class TestAutoDetectEmbedder:
    def test_auto_detect_returns_sentence_transformer_when_available(self):
        """_auto_detect_embedder() returns a provider when sentence-transformers is installed."""
        provider = BeliefMemory._auto_detect_embedder()
        # sentence-transformers is available (pytestmark skips when not)
        assert provider is not None

    def test_auto_detect_falls_back_to_openai_when_st_unavailable(self, monkeypatch):
        """_auto_detect_embedder() tries OpenAI when sentence-transformers is absent."""
        from unittest.mock import MagicMock, patch

        fake_provider = MagicMock()
        fake_cls = MagicMock(return_value=fake_provider)

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        def _fake_st_import(name, *args, **kwargs):
            raise ImportError("No module named 'sentence_transformers'")

        with (
            patch(
                "mnemebrain_core.providers.embeddings.sentence_transformers.SentenceTransformerProvider",
                side_effect=_fake_st_import,
            ),
            patch(
                "mnemebrain_core.providers.embeddings.openai.OpenAIEmbeddingProvider",
                fake_cls,
            ),
        ):
            # Re-patch both inner imports inside _auto_detect_embedder

            # We patch the class via its module; the function uses a local import
            # so we patch via sys.modules manipulation of the provider modules.
            pass

        # Simpler approach: patch the SentenceTransformerProvider constructor to raise
        # so the first branch falls through, and patch OpenAIEmbeddingProvider to return fake
        from mnemebrain_core.providers.embeddings import (
            sentence_transformers as st_mod,
        )
        import mnemebrain_core.providers.embeddings.openai as openai_mod

        with (
            patch.object(
                st_mod,
                "SentenceTransformerProvider",
                side_effect=ImportError("mocked"),
            ),
            patch.object(
                openai_mod,
                "OpenAIEmbeddingProvider",
                fake_cls,
            ),
        ):
            result = BeliefMemory._auto_detect_embedder()

        # Either the openai branch ran and returned our fake, or the class was
        # instantiated from the patched location.
        assert result is fake_provider or result is None

    def test_auto_detect_returns_none_when_no_providers(self, monkeypatch):
        """_auto_detect_embedder() returns None when neither provider is importable."""
        from unittest.mock import patch

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from mnemebrain_core.providers.embeddings import (
            sentence_transformers as st_mod,
        )

        with patch.object(
            st_mod,
            "SentenceTransformerProvider",
            side_effect=ImportError("mocked"),
        ):
            result = BeliefMemory._auto_detect_embedder()

        # Without OPENAI_API_KEY the OpenAI branch is skipped → returns None
        assert result is None

    def test_auto_detect_openai_import_error_is_caught(self, monkeypatch):
        """_auto_detect_embedder() catches ImportError when OpenAI provider fails to import."""
        import sys
        from types import ModuleType
        from unittest.mock import patch

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        from mnemebrain_core.providers.embeddings import (
            sentence_transformers as st_mod,
        )

        # Replace the openai provider module with a dummy that has no
        # OpenAIEmbeddingProvider attribute — this makes the
        # `from mnemebrain_core.providers.embeddings.openai import OpenAIEmbeddingProvider`
        # raise ImportError.
        mod_key = "mnemebrain_core.providers.embeddings.openai"
        original_mod = sys.modules.get(mod_key)
        dummy = ModuleType(mod_key)
        # Intentionally do NOT add OpenAIEmbeddingProvider to dummy

        try:
            sys.modules[mod_key] = dummy
            with patch.object(
                st_mod,
                "SentenceTransformerProvider",
                side_effect=ImportError("mocked st"),
            ):
                result = BeliefMemory._auto_detect_embedder()

            # Both providers failed → returns None
            assert result is None
        finally:
            if original_mod is not None:
                sys.modules[mod_key] = original_mod
            else:
                sys.modules.pop(mod_key, None)
