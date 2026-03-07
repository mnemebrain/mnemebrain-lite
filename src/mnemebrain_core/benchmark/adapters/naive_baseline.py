"""Naive baseline memory system -- flat vector store, no belief logic."""
from __future__ import annotations

from uuid import uuid4

from mnemebrain_core.benchmark.interface import (
    Capability,
    MemorySystem,
    QueryResult,
    StoreResult,
)


class NaiveBaseline(MemorySystem):
    """Flat in-memory vector store with cosine-similarity deduplication.

    Supports STORE and QUERY only. Retract, explain, and temporal decay
    raise NotImplementedError as the capability set advertises.
    """

    def __init__(self, embedder: object, threshold: float = 0.92) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._beliefs: list[dict] = []

    def name(self) -> str:
        """Return the stable identifier for this adapter."""
        return "naive_baseline"

    def capabilities(self) -> set[Capability]:
        """Return the set of capabilities this adapter supports."""
        return {Capability.STORE, Capability.QUERY}

    def store(self, claim: str, evidence: list[dict]) -> StoreResult:
        """Store a claim, merging if a near-duplicate already exists.

        A near-duplicate is defined as any stored belief whose embedding
        has cosine similarity >= threshold with the new claim's embedding.
        """
        embedding = self._embedder.embed(claim)
        for b in self._beliefs:
            sim = self._embedder.similarity(embedding, b["embedding"])
            if sim >= self._threshold:
                return StoreResult(
                    belief_id=b["id"],
                    merged=True,
                    contradiction_detected=False,
                    truth_state=None,
                    confidence=None,
                )
        belief_id = str(uuid4())
        self._beliefs.append({"id": belief_id, "claim": claim, "embedding": embedding})
        return StoreResult(
            belief_id=belief_id,
            merged=False,
            contradiction_detected=False,
            truth_state=None,
            confidence=None,
        )

    def query(self, claim: str) -> list[QueryResult]:
        """Return all stored beliefs with cosine similarity >= 0.5 to claim."""
        if not self._beliefs:
            return []
        embedding = self._embedder.embed(claim)
        results = []
        for b in self._beliefs:
            sim = self._embedder.similarity(embedding, b["embedding"])
            if sim >= 0.5:
                results.append(QueryResult(
                    belief_id=b["id"],
                    claim=b["claim"],
                    confidence=None,
                    truth_state=None,
                ))
        return results

    def reset(self) -> None:
        """Clear all stored beliefs."""
        self._beliefs.clear()
