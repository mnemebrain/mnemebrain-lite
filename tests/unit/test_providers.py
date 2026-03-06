"""Tests for embedding providers — covers similarity method."""

import pytest

from mnemebrain_core.providers.embeddings.sentence_transformers import (
    SentenceTransformerProvider,
)


@pytest.fixture(scope="module")
def provider():
    return SentenceTransformerProvider()


class TestSentenceTransformerProvider:
    def test_similarity_identical_vectors(self, provider: SentenceTransformerProvider):
        """similarity() on identical normalized vectors returns ~1.0 (lines 23-25)."""
        vec = provider.embed("hello world")
        sim = provider.similarity(vec, vec)
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_similarity_different_vectors(self, provider: SentenceTransformerProvider):
        """similarity() on different vectors returns < 1.0."""
        a = provider.embed("the sun is bright")
        b = provider.embed("database query optimization")
        sim = provider.similarity(a, b)
        assert sim < 0.9
