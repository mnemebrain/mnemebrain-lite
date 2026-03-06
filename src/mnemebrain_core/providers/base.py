"""Abstract provider interfaces for embeddings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EvidenceInput:
    """Input for creating evidence."""

    source_ref: str
    content: str
    polarity: str  # "supports" or "attacks"
    weight: float = 0.7
    reliability: float = 0.8
    scope: str | None = None


class EmbeddingProvider(ABC):
    """Abstract interface for text embedding."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed text into a vector."""
        ...

    @abstractmethod
    def similarity(self, a: list[float], b: list[float]) -> float:
        """Compute similarity between two vectors. Returns 0.0-1.0."""
        ...
