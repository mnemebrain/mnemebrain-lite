"""Tests for embedding providers — covers embed and similarity methods."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

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


# --- SentenceTransformer provider tests (mocked, no package needed) ---


class TestSentenceTransformerProviderMocked:
    """Tests that exercise the provider code with a mocked SentenceTransformer."""

    @pytest.fixture(autouse=True)
    def _mock_sentence_transformers(self):
        import numpy as np

        fake_st_mod = ModuleType("sentence_transformers")
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.6, 0.8, 0.0])

        fake_st_mod.SentenceTransformer = MagicMock(return_value=mock_model)
        sys.modules["sentence_transformers"] = fake_st_mod

        mod_name = "mnemebrain_core.providers.embeddings.sentence_transformers"
        if mod_name in sys.modules:
            del sys.modules[mod_name]

        yield

        del sys.modules["sentence_transformers"]
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    def _make_provider(self):
        from mnemebrain_core.providers.embeddings.sentence_transformers import (
            SentenceTransformerProvider,
        )

        return SentenceTransformerProvider()

    def test_embed_returns_list_of_floats(self):
        provider = self._make_provider()
        result = provider.embed("hello")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)
        assert result == [pytest.approx(0.6), pytest.approx(0.8), pytest.approx(0.0)]

    def test_similarity_dot_product(self):
        provider = self._make_provider()
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert provider.similarity(a, b) == pytest.approx(0.0, abs=0.001)

    def test_similarity_identical(self):
        provider = self._make_provider()
        vec = [0.6, 0.8]
        assert provider.similarity(vec, vec) == pytest.approx(1.0, abs=0.001)

    def test_default_model_name(self):
        from mnemebrain_core.providers.embeddings.sentence_transformers import (
            SentenceTransformerProvider,
        )
        import sentence_transformers

        SentenceTransformerProvider()
        sentence_transformers.SentenceTransformer.assert_called_with("all-MiniLM-L6-v2")

    def test_custom_model_name(self):
        from mnemebrain_core.providers.embeddings.sentence_transformers import (
            SentenceTransformerProvider,
        )
        import sentence_transformers

        SentenceTransformerProvider("custom-model")
        sentence_transformers.SentenceTransformer.assert_called_with("custom-model")


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


# --- OpenAI-compatible provider tests (mocked, no server needed) ---


class TestOpenAICompatibleProvider:
    def _make_provider(self, api_key=None):
        from mnemebrain_core.providers.embeddings.openai_compatible import (
            OpenAICompatibleProvider,
        )

        return OpenAICompatibleProvider(
            base_url="http://localhost:11434/v1",
            model="nomic-embed-text",
            api_key=api_key,
        )

    def test_embed_returns_vector(self):
        """embed() returns the vector from a mocked response."""
        provider = self._make_provider()
        expected = [0.1, 0.2, 0.3, 0.4]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": expected}]}
        with patch.object(provider._client, "post", return_value=mock_response):
            result = provider.embed("hello")
        assert result == expected

    def test_embed_sends_correct_payload(self):
        """embed() sends the right JSON body."""
        provider = self._make_provider()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1]}]}
        with patch.object(
            provider._client, "post", return_value=mock_response
        ) as mock_post:
            provider.embed("test text")
        mock_post.assert_called_once_with(
            "/embeddings",
            json={"input": "test text", "model": "nomic-embed-text"},
        )

    def test_embed_with_api_key(self):
        """When api_key is set, Authorization header is included."""
        provider = self._make_provider(api_key="sk-test-key")
        assert provider._client.headers["authorization"] == "Bearer sk-test-key"

    def test_embed_without_api_key(self):
        """When no api_key, no Authorization header."""
        provider = self._make_provider()
        assert "authorization" not in provider._client.headers

    def test_embed_error_response(self):
        """Non-200 response raises RuntimeError."""
        provider = self._make_provider()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        with patch.object(provider._client, "post", return_value=mock_response):
            with pytest.raises(RuntimeError, match="Embedding request failed"):
                provider.embed("hello")

    def test_similarity_identical(self):
        """similarity() on identical vectors returns 1.0."""
        provider = self._make_provider()
        vec = [0.6, 0.8]
        assert provider.similarity(vec, vec) == pytest.approx(1.0, abs=0.001)

    def test_similarity_zero_vector(self):
        """similarity() with zero vector returns 0.0."""
        provider = self._make_provider()
        vec = [0.5, 0.5]
        zero = [0.0, 0.0]
        assert provider.similarity(vec, zero) == 0.0
        assert provider.similarity(zero, vec) == 0.0
