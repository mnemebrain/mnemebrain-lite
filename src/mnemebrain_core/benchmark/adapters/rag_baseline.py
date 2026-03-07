"""RAG (Retrieval-Augmented Generation) baseline adapter.

Simulates a typical RAG memory system:
- Vector store with semantic similarity retrieval
- Last-write-wins: new info on same topic OVERWRITES old entry
- No contradiction detection (overwrites silently)
- No evidence tracking, no temporal decay
- No sandbox, no attack edges
"""
from __future__ import annotations

from uuid import uuid4

from mnemebrain_core.benchmark.interface import (
    Capability,
    MemorySystem,
    QueryResult,
    StoreResult,
)


class RAGBaseline(MemorySystem):
    """Vector-store RAG memory with overwrite-on-conflict semantics.

    Uses embedding similarity for retrieval and deduplication.
    When a new claim is similar enough to an existing one, the old
    entry is REPLACED (last-write-wins), not merged or flagged.
    """

    def __init__(self, embedder: object, threshold: float = 0.75) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._entries: list[dict] = []

    def name(self) -> str:
        return "rag_baseline"

    def capabilities(self) -> set[Capability]:
        return {Capability.STORE, Capability.QUERY}

    def store(self, claim: str, evidence: list[dict]) -> StoreResult:
        """Store claim. If similar entry exists, overwrite it (last-write-wins)."""
        embedding = self._embedder.embed(claim)
        for i, entry in enumerate(self._entries):
            sim = self._embedder.similarity(embedding, entry["embedding"])
            if sim >= self._threshold:
                # Overwrite: RAG systems replace old info with new
                self._entries[i] = {
                    "id": entry["id"],
                    "claim": claim,
                    "embedding": embedding,
                }
                return StoreResult(
                    belief_id=entry["id"],
                    merged=True,
                    contradiction_detected=False,
                    truth_state=None,
                    confidence=None,
                )
        entry_id = str(uuid4())
        self._entries.append({
            "id": entry_id,
            "claim": claim,
            "embedding": embedding,
        })
        return StoreResult(
            belief_id=entry_id,
            merged=False,
            contradiction_detected=False,
            truth_state=None,
            confidence=None,
        )

    def query(self, claim: str) -> list[QueryResult]:
        """Retrieve entries by embedding similarity."""
        if not self._entries:
            return []
        embedding = self._embedder.embed(claim)
        results = []
        for entry in self._entries:
            sim = self._embedder.similarity(embedding, entry["embedding"])
            if sim >= 0.5:
                results.append(QueryResult(
                    belief_id=entry["id"],
                    claim=entry["claim"],
                    confidence=None,
                    truth_state=None,
                ))
        return results

    def reset(self) -> None:
        self._entries.clear()
