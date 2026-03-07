"""LangChain-style ConversationBufferMemory adapter.

Simulates how LangChain's ConversationBufferMemory works:
- Append-only text buffer (no structure)
- Keyword/substring matching for retrieval
- No contradiction detection, no evidence tracking
- Last-write-wins for conflicting information
- No temporal decay, no revision, no sandbox
"""
from __future__ import annotations

from uuid import uuid4

from mnemebrain_core.benchmark.interface import (
    Capability,
    MemorySystem,
    QueryResult,
    StoreResult,
)


class LangChainBufferBaseline(MemorySystem):
    """Append-only text buffer simulating LangChain ConversationBufferMemory.

    Stores claims as plain text entries. Query uses substring matching
    and simple word overlap scoring. No belief logic whatsoever.
    """

    def __init__(self) -> None:
        self._entries: list[dict] = []

    def name(self) -> str:
        return "langchain_buffer"

    def capabilities(self) -> set[Capability]:
        return {Capability.STORE, Capability.QUERY}

    def store(self, claim: str, evidence: list[dict]) -> StoreResult:
        """Append claim to buffer. No dedup, no conflict detection."""
        entry_id = str(uuid4())
        self._entries.append({"id": entry_id, "claim": claim})
        return StoreResult(
            belief_id=entry_id,
            merged=False,
            contradiction_detected=False,
            truth_state=None,
            confidence=None,
        )

    def query(self, claim: str) -> list[QueryResult]:
        """Simple word-overlap retrieval (simulates buffer search)."""
        if not self._entries:
            return []
        query_words = set(claim.lower().split())
        results = []
        for entry in self._entries:
            entry_words = set(entry["claim"].lower().split())
            overlap = len(query_words & entry_words)
            if overlap >= 2 or any(w in entry["claim"].lower() for w in query_words if len(w) > 4):
                results.append(QueryResult(
                    belief_id=entry["id"],
                    claim=entry["claim"],
                    confidence=None,
                    truth_state=None,
                ))
        return results

    def reset(self) -> None:
        self._entries.clear()
