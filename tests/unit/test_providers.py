"""Tests for embedding providers — covers embed and similarity methods."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# --- SentenceTransformer provider tests ---

try:
    from mnemebrain_core.providers.embeddings.sentence_transformers import (
        SentenceTransformerProvider,
    )

    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False


@pytest.fixture(scope="module")
def st_provider():
    if not HAS_EMBEDDINGS:
        pytest.skip("sentence-transformers not installed")
    return SentenceTransformerProvider()


class TestSentenceTransformerProvider:
    def test_similarity_identical_vectors(self, st_provider):
        """similarity() on identical normalized vectors returns ~1.0."""
        vec = st_provider.embed("hello world")
        sim = st_provider.similarity(vec, vec)
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_similarity_different_vectors(self, st_provider):
        """similarity() on different vectors returns < 1.0."""
        a = st_provider.embed("the sun is bright")
        b = st_provider.embed("database query optimization")
        sim = st_provider.similarity(a, b)
        assert sim < 0.9

    def test_embed_returns_list_of_floats(self, st_provider):
        """embed() returns a list of floats."""
        vec = st_provider.embed("test")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec)

    def test_similarity_zero_norm(self, st_provider):
        """similarity() with zero vector — SentenceTransformer uses dot product."""
        vec = st_provider.embed("hello")
        zero = [0.0] * len(vec)
        sim = st_provider.similarity(vec, zero)
        assert sim == pytest.approx(0.0, abs=0.01)


# --- OpenAI provider tests (mocked, no API key needed) ---


@pytest.fixture()
def mock_openai():
    """Inject a fake openai module to test without the real package."""
    fake_openai = ModuleType("openai")

    embedding_vector = [0.1, 0.2, 0.3, 0.4, 0.5]

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.embeddings = MagicMock()
            mock_resp = MagicMock()
            mock_data = MagicMock()
            mock_data.embedding = embedding_vector
            mock_resp.data = [mock_data]
            self.embeddings.create.return_value = mock_resp

    fake_openai.OpenAI = FakeOpenAI
    sys.modules["openai"] = fake_openai

    # Force reimport of the provider module
    mod_name = "mnemebrain_core.providers.embeddings.openai"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    yield fake_openai, embedding_vector

    del sys.modules["openai"]
    if mod_name in sys.modules:
        del sys.modules[mod_name]


class TestOpenAIEmbeddingProvider:
    def test_embed_returns_expected_vector(self, mock_openai):
        """embed() returns the vector from the mocked API response."""
        _, expected_vec = mock_openai
        from mnemebrain_core.providers.embeddings.openai import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider()
        result = provider.embed("test text")
        assert result == expected_vec

    def test_embed_calls_api_with_correct_model(self, mock_openai):
        """embed() passes the configured model to the API."""
        from mnemebrain_core.providers.embeddings.openai import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")
        provider.embed("test")
        provider._client.embeddings.create.assert_called_once_with(
            input="test", model="text-embedding-3-large"
        )

    def test_similarity_identical_vectors(self, mock_openai):
        """similarity() returns 1.0 for identical vectors."""
        from mnemebrain_core.providers.embeddings.openai import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider()
        vec = [0.6, 0.8]
        assert provider.similarity(vec, vec) == pytest.approx(1.0, abs=0.001)

    def test_similarity_orthogonal_vectors(self, mock_openai):
        """similarity() returns ~0.0 for orthogonal vectors."""
        from mnemebrain_core.providers.embeddings.openai import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider()
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert provider.similarity(a, b) == pytest.approx(0.0, abs=0.001)

    def test_similarity_zero_vector_returns_zero(self, mock_openai):
        """similarity() returns 0.0 when one vector is all zeros."""
        from mnemebrain_core.providers.embeddings.openai import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider()
        vec = [0.5, 0.5, 0.5]
        zero = [0.0, 0.0, 0.0]
        assert provider.similarity(vec, zero) == 0.0
        assert provider.similarity(zero, vec) == 0.0

    def test_similarity_opposite_vectors(self, mock_openai):
        """similarity() returns -1.0 for opposite vectors."""
        from mnemebrain_core.providers.embeddings.openai import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider()
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert provider.similarity(a, b) == pytest.approx(-1.0, abs=0.001)

    def test_default_model(self, mock_openai):
        """Default model is text-embedding-3-small."""
        from mnemebrain_core.providers.embeddings.openai import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider()
        assert provider._model == "text-embedding-3-small"
