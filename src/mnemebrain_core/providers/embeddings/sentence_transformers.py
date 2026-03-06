"""Sentence-transformers embedding provider — local, free, no API key."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from mnemebrain_core.providers.base import EmbeddingProvider

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SentenceTransformerProvider(EmbeddingProvider):
    """Local embedding using sentence-transformers."""

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def similarity(self, a: list[float], b: list[float]) -> float:
        va = np.array(a)
        vb = np.array(b)
        return float(np.dot(va, vb))
