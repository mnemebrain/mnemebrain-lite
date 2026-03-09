"""OpenAI embedding provider — uses text-embedding-3-small."""

from __future__ import annotations

import numpy as np

from mnemebrain_core.providers.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using OpenAI's API."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        import openai

        self._client = openai.OpenAI()
        self._model = model

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(input=text, model=self._model)
        return response.data[0].embedding

    def similarity(self, a: list[float], b: list[float]) -> float:
        a_arr, b_arr = np.array(a), np.array(b)
        norm_a, norm_b = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))
