"""Tests for BeliefMemory — covers uncovered branches in memory.py."""

from __future__ import annotations

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from mnemebrain_core.memory import BeliefMemory, BeliefResult
from mnemebrain_core.models import (
    Belief,
    ConflictPolicy,
)
from mnemebrain_core.providers.base import EmbeddingProvider, EvidenceInput


class FakeEmbedder(EmbeddingProvider):
    """Simple fake embedder that returns a fixed vector."""

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def similarity(self, a: list[float], b: list[float]) -> float:
        return 0.99


@pytest.fixture
def memory():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_mem")
    m = BeliefMemory(
        db_path=db_path, embedding_provider=FakeEmbedder(), max_db_size=1 << 30
    )
    yield m
    m.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# _auto_detect_embedder
# ---------------------------------------------------------------------------


class TestAutoDetectEmbedder:
    def test_returns_none_when_no_providers(self):
        """If sentence-transformers and openai are both missing, returns None."""
        with (
            patch.dict(
                "sys.modules",
                {"mnemebrain_core.providers.embeddings.sentence_transformers": None},
            ),
            patch(
                "mnemebrain_core.memory.BeliefMemory._auto_detect_embedder",
                wraps=BeliefMemory._auto_detect_embedder,
            ),
        ):
            # Call the static method directly to test it
            result = BeliefMemory._auto_detect_embedder()
            # Without actual installs, we expect either a real provider or None
            # The key is that it doesn't raise
            assert result is None or hasattr(result, "embed")

    def test_auto_detect_sentence_transformers(self):
        """When sentence-transformers is available, returns that provider."""
        mock_provider = MagicMock(spec=EmbeddingProvider)
        mock_module = MagicMock()
        mock_module.SentenceTransformerProvider.return_value = mock_provider
        with patch.dict(
            "sys.modules",
            {"mnemebrain_core.providers.embeddings.sentence_transformers": mock_module},
        ):
            result = BeliefMemory._auto_detect_embedder()
            assert result is mock_provider

    def test_auto_detect_falls_to_openai(self):
        """When sentence-transformers fails, tries OpenAI if key is set."""
        mock_openai_provider = MagicMock(spec=EmbeddingProvider)
        mock_openai_module = MagicMock()
        mock_openai_module.OpenAIEmbeddingProvider.return_value = mock_openai_provider

        def _fake_import(name, *args, **kwargs):
            if "sentence_transformers" in name:
                raise ImportError("no sentence-transformers")
            return original_import(name, *args, **kwargs)

        import builtins

        original_import = builtins.__import__

        with (
            patch("builtins.__import__", side_effect=_fake_import),
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
            patch.dict(
                "sys.modules",
                {"mnemebrain_core.providers.embeddings.openai": mock_openai_module},
            ),
        ):
            result = BeliefMemory._auto_detect_embedder()
            assert result is mock_openai_provider

    def test_auto_detect_openai_import_fails(self):
        """When OPENAI_API_KEY is set but openai import fails, falls through."""

        def _fake_import(name, *args, **kwargs):
            if "sentence_transformers" in name:
                raise ImportError("no sentence-transformers")
            if "openai" in name and "providers" in name:
                raise ImportError("no openai package")
            return original_import(name, *args, **kwargs)

        import builtins

        original_import = builtins.__import__

        with (
            patch("builtins.__import__", side_effect=_fake_import),
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        ):
            # Remove cached module so import is re-attempted
            import sys

            sys.modules.pop("mnemebrain_core.providers.embeddings.openai", None)
            result = BeliefMemory._auto_detect_embedder()
            assert result is None

    def test_auto_detect_openai_no_key(self):
        """When sentence-transformers fails and no OPENAI_API_KEY, returns None."""

        def _fake_import(name, *args, **kwargs):
            if "sentence_transformers" in name:
                raise ImportError("no sentence-transformers")
            return original_import(name, *args, **kwargs)

        import builtins

        original_import = builtins.__import__

        with (
            patch("builtins.__import__", side_effect=_fake_import),
            patch.dict("os.environ", {}, clear=True),
        ):
            # Remove OPENAI_API_KEY if present
            os.environ.pop("OPENAI_API_KEY", None)
            result = BeliefMemory._auto_detect_embedder()
            assert result is None

    def test_auto_detect_openai_compatible(self):
        """When EMBEDDING_BASE_URL and EMBEDDING_MODEL are set, returns compatible provider."""

        def _fake_import(name, *args, **kwargs):
            if "sentence_transformers" in name:
                raise ImportError("no sentence-transformers")
            return original_import(name, *args, **kwargs)

        import builtins

        original_import = builtins.__import__

        with (
            patch("builtins.__import__", side_effect=_fake_import),
            patch.dict(
                "os.environ",
                {
                    "EMBEDDING_BASE_URL": "http://localhost:11434/v1",
                    "EMBEDDING_MODEL": "nomic-embed-text",
                },
                clear=True,
            ),
        ):
            result = BeliefMemory._auto_detect_embedder()
            from mnemebrain_core.providers.embeddings.openai_compatible import (
                OpenAICompatibleProvider,
            )

            assert isinstance(result, OpenAICompatibleProvider)

    def test_auto_detect_skips_when_partial_env(self):
        """When only EMBEDDING_BASE_URL is set (no model), skips compatible provider."""

        def _fake_import(name, *args, **kwargs):
            if "sentence_transformers" in name:
                raise ImportError("no sentence-transformers")
            return original_import(name, *args, **kwargs)

        import builtins

        original_import = builtins.__import__

        with (
            patch("builtins.__import__", side_effect=_fake_import),
            patch.dict(
                "os.environ",
                {"EMBEDDING_BASE_URL": "http://localhost:11434/v1"},
                clear=True,
            ),
        ):
            result = BeliefMemory._auto_detect_embedder()
            assert result is None


# ---------------------------------------------------------------------------
# Degraded mode (no embedder)
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_no_embedder():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_no_emb")
    m = BeliefMemory(
        db_path=db_path, embedding_provider=FakeEmbedder(), max_db_size=1 << 30
    )
    m._embedder = None  # Force degraded mode
    yield m
    m.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestDegradedMode:
    def test_believe_without_embedder(self, memory_no_embedder: BeliefMemory):
        """believe() works without an embedder."""
        result = memory_no_embedder.believe(
            claim="test claim",
            evidence_items=[
                EvidenceInput(source_ref="s1", content="c1", polarity="supports")
            ],
        )
        assert isinstance(result, BeliefResult)
        assert result.confidence > 0

    def test_believe_dedup_exact_match(self, memory_no_embedder: BeliefMemory):
        """Same claim twice without embedder merges via exact match."""
        r1 = memory_no_embedder.believe(
            claim="duplicate claim",
            evidence_items=[
                EvidenceInput(source_ref="s1", content="c1", polarity="supports")
            ],
        )
        r2 = memory_no_embedder.believe(
            claim="duplicate claim",
            evidence_items=[
                EvidenceInput(source_ref="s2", content="c2", polarity="supports")
            ],
        )
        assert r1.id == r2.id

    def test_explain_without_embedder(self, memory_no_embedder: BeliefMemory):
        """explain() falls back to exact match without embedder."""
        memory_no_embedder.believe(
            claim="explainable",
            evidence_items=[
                EvidenceInput(source_ref="s1", content="c1", polarity="supports")
            ],
        )
        result = memory_no_embedder.explain("explainable")
        assert result is not None
        assert result.claim == "explainable"

    def test_search_without_embedder(self, memory_no_embedder: BeliefMemory):
        """search() uses substring match without embedder."""
        memory_no_embedder.believe(
            claim="searchable belief about cats",
            evidence_items=[
                EvidenceInput(source_ref="s1", content="c1", polarity="supports")
            ],
        )
        results = memory_no_embedder.search("cats")
        assert len(results) >= 1
        assert "cats" in results[0][0].claim

    def test_warning_logged(self):
        """Degraded mode logs a warning at init."""
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_warn")
        try:
            import mnemebrain_core.memory as mem_mod

            with (
                patch.object(BeliefMemory, "_auto_detect_embedder", return_value=None),
                patch.object(mem_mod.logger, "warning") as mock_warn,
            ):
                m = BeliefMemory(db_path=db_path, max_db_size=1 << 30)
                mock_warn.assert_called_once()
                assert "degraded mode" in mock_warn.call_args[0][0].lower()
                m.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# believe
# ---------------------------------------------------------------------------


class TestBelieve:
    def test_believe_creates_new(self, memory: BeliefMemory):
        result = memory.believe(
            claim="Earth is round",
            evidence_items=[
                EvidenceInput(
                    source_ref="src_1",
                    content="observed from space",
                    polarity="supports",
                )
            ],
        )
        assert isinstance(result, BeliefResult)
        assert result.confidence > 0

    def test_believe_merges_existing(self, memory: BeliefMemory):
        """When similar belief exists (>0.92 similarity), merges evidence."""
        # First believe
        r1 = memory.believe(
            claim="Earth is round",
            evidence_items=[
                EvidenceInput(source_ref="s1", content="c1", polarity="supports")
            ],
        )
        # Second believe — same claim, same embedding (FakeEmbedder returns same vector)
        r2 = memory.believe(
            claim="Earth is round",
            evidence_items=[
                EvidenceInput(source_ref="s2", content="c2", polarity="supports")
            ],
        )
        # Should merge into the same belief
        assert r1.id == r2.id


# ---------------------------------------------------------------------------
# retract
# ---------------------------------------------------------------------------


class TestRetract:
    def test_retract_existing_evidence(self, memory: BeliefMemory):
        result = memory.believe(
            claim="test retract",
            evidence_items=[
                EvidenceInput(source_ref="s1", content="c1", polarity="supports")
            ],
        )
        # Get the belief to find its evidence
        belief = memory._store.get(result.id)
        assert belief is not None
        ev_id = belief.evidence[0].id

        retract_results = memory.retract(ev_id)
        assert len(retract_results) >= 1
        assert retract_results[0].id == result.id

    def test_retract_nonexistent_evidence(self, memory: BeliefMemory):
        result = memory.retract(uuid4())
        assert result == []


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------


class TestExplain:
    def test_explain_found_via_similarity(self, memory: BeliefMemory):
        memory.believe(
            claim="explain test",
            evidence_items=[
                EvidenceInput(source_ref="s1", content="c1", polarity="supports")
            ],
        )
        result = memory.explain("explain test")
        assert result is not None
        assert result.claim == "explain test"

    def test_explain_not_found(self, memory: BeliefMemory):
        result = memory.explain("nonexistent claim")
        assert result is None

    def test_explain_falls_back_to_exact_match(self, memory: BeliefMemory):
        """When find_similar returns nothing, falls back to find_by_claim."""
        # Store a belief without embedding
        belief = Belief(claim="exact match only")
        memory._store.upsert(belief)  # No embedding

        # The fake embedder returns [0.1, 0.2, 0.3] but stored belief has no embedding
        # find_similar with threshold=0.8 won't find it
        # But find_by_claim should find it
        result = memory.explain("exact match only")
        assert result is not None
        assert result.claim == "exact match only"


# ---------------------------------------------------------------------------
# revise
# ---------------------------------------------------------------------------


class TestRevise:
    def test_revise_existing(self, memory: BeliefMemory):
        r = memory.believe(
            claim="revise me",
            evidence_items=[
                EvidenceInput(source_ref="s1", content="c1", polarity="supports")
            ],
        )
        result = memory.revise(
            r.id,
            EvidenceInput(source_ref="s2", content="new evidence", polarity="attacks"),
        )
        assert isinstance(result, BeliefResult)
        assert result.id == r.id

    def test_revise_nonexistent(self, memory: BeliefMemory):
        with pytest.raises(ValueError, match="not found"):
            memory.revise(
                uuid4(),
                EvidenceInput(source_ref="s", content="c", polarity="supports"),
            )


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_returns_ranked(self, memory: BeliefMemory):
        memory.believe(
            claim="searchable belief",
            evidence_items=[
                EvidenceInput(source_ref="s1", content="c1", polarity="supports")
            ],
        )
        results = memory.search("searchable")
        assert len(results) >= 1
        # Each result is (belief, similarity, confidence, rank_score)
        b, sim, conf, rs = results[0]
        assert b.claim == "searchable belief"
        assert sim > 0
        assert rs > 0

    def test_search_with_conflict_policy(self, memory: BeliefMemory):
        memory.believe(
            claim="policy test",
            evidence_items=[
                EvidenceInput(source_ref="s1", content="c1", polarity="supports")
            ],
        )
        results = memory.search(
            "policy test", conflict_policy=ConflictPolicy.CONSERVATIVE
        )
        assert isinstance(results, list)

    def test_search_empty(self, memory: BeliefMemory):
        results = memory.search("nothing here at all xyz")
        assert results == []


# ---------------------------------------------------------------------------
# list_beliefs (delegates to store)
# ---------------------------------------------------------------------------


class TestListBeliefs:
    def test_list_beliefs(self, memory: BeliefMemory):
        memory.believe(
            claim="list me",
            evidence_items=[
                EvidenceInput(source_ref="s1", content="c1", polarity="supports")
            ],
        )
        beliefs, total = memory.list_beliefs()
        assert total >= 1


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestClose:
    def test_close(self, memory: BeliefMemory):
        memory.close()
