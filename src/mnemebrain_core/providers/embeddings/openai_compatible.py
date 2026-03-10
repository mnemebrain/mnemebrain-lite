"""OpenAI-compatible embedding provider — works with Ollama, LM Studio, vLLM, etc."""

from __future__ import annotations

import numpy as np

from mnemebrain_core.providers.base import EmbeddingProvider


class OpenAICompatibleProvider(EmbeddingProvider):
    """Embedding provider using any OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
    ) -> None:
        import httpx

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(base_url=base_url, headers=headers, timeout=30.0)
        self._model = model

    def embed(self, text: str) -> list[float]:
        """Embed text via the /embeddings endpoint."""
        response = self._client.post(
            "/embeddings",
            json={"input": text, "model": self._model},
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Embedding request failed ({response.status_code}): {response.text}"
            )
        return response.json()["data"][0]["embedding"]

    def similarity(self, a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        a_arr, b_arr = np.array(a), np.array(b)
        norm_a, norm_b = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))
