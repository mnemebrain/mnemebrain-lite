"""BeliefMemory — the 4 core operations for belief management."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from mnemebrain_core.engine import compute_confidence, compute_truth_state
from mnemebrain_core.models import (
    Belief,
    BeliefType,
    ConflictPolicy,
    Evidence,
    Polarity,
    TruthState,
)
from mnemebrain_core.providers.base import EmbeddingProvider, EvidenceInput
from mnemebrain_core.store import KuzuGraphStore

logger = logging.getLogger(__name__)


@dataclass
class BeliefResult:
    """Result of a belief operation."""

    id: UUID
    truth_state: TruthState
    confidence: float
    conflict: bool


@dataclass
class ExplanationResult:
    """Result of explain() — full justification chain."""

    claim: str
    truth_state: TruthState
    confidence: float
    supporting: list[Evidence]
    attacking: list[Evidence]
    expired: list[Evidence]


class BeliefMemory:
    """Core belief memory system — 4 operations: believe, retract, explain, revise."""

    def __init__(
        self,
        db_path: str,
        embedding_provider: EmbeddingProvider | None = None,
        *,
        max_db_size: int = 0,
    ) -> None:
        self._store = KuzuGraphStore(db_path, max_db_size=max_db_size)
        self._embedder = embedding_provider
        if self._embedder is None:
            self._embedder = self._auto_detect_embedder()
        if self._embedder is None:
            logger.warning(
                "No embedding provider available. Running in degraded mode: "
                "believe/explain/search will use text matching instead of "
                "semantic similarity. Install sentence-transformers, set "
                "EMBEDDING_BASE_URL+EMBEDDING_MODEL, or set OPENAI_API_KEY "
                "to enable embeddings."
            )

    @staticmethod
    def _auto_detect_embedder() -> EmbeddingProvider | None:
        """Try available embedding providers in order of preference."""
        import os

        # 1. Local sentence-transformers (no API key needed)
        try:
            from mnemebrain_core.providers.embeddings.sentence_transformers import (
                SentenceTransformerProvider,
            )

            return SentenceTransformerProvider()
        except ImportError:
            pass
        # 2. OpenAI-compatible server (Ollama, LM Studio, vLLM, etc.)
        base_url = os.environ.get("EMBEDDING_BASE_URL")
        model = os.environ.get("EMBEDDING_MODEL")
        if base_url and model:
            from mnemebrain_core.providers.embeddings.openai_compatible import (
                OpenAICompatibleProvider,
            )

            return OpenAICompatibleProvider(
                base_url=base_url,
                model=model,
                api_key=os.environ.get("EMBEDDING_API_KEY"),
            )
        # 3. OpenAI API (requires OPENAI_API_KEY)
        try:
            if os.environ.get("OPENAI_API_KEY"):
                from mnemebrain_core.providers.embeddings.openai import (
                    OpenAIEmbeddingProvider,
                )

                return OpenAIEmbeddingProvider()
        except ImportError:
            pass
        return None  # Degraded mode — text matching only

    def believe(
        self,
        claim: str,
        evidence_items: list[EvidenceInput],
        belief_type: BeliefType = BeliefType.INFERENCE,
        tags: list[str] | None = None,
        source_agent: str = "",
    ) -> BeliefResult:
        """Store a new belief with evidence. Merges if similar belief exists."""
        embedding: list[float]
        if self._embedder is not None:
            embedding = self._embedder.embed(claim)
            existing = self._store.find_similar(embedding, threshold=0.92)
        else:
            embedding = []
            exact = self._store.find_by_claim(claim)
            existing = [(exact, 1.0)] if exact else []

        if existing:
            belief = existing[0][0]
        else:
            belief = Belief(
                claim=claim,
                belief_type=belief_type,
                tags=tags or [],
                source_agent=source_agent,
            )

        for item in evidence_items:
            ev = Evidence(
                belief_id=belief.id,
                source_ref=item.source_ref,
                content=item.content,
                polarity=Polarity(item.polarity),
                reliability=item.reliability,
                weight=item.weight,
                scope=item.scope,
            )
            belief.evidence.append(ev)

        belief.truth_state = compute_truth_state(belief.evidence, belief.belief_type)
        belief.confidence = compute_confidence(belief.evidence, belief.belief_type)
        belief.last_revised = datetime.now(timezone.utc)

        self._store.upsert(belief, embedding=embedding)

        return BeliefResult(
            id=belief.id,
            truth_state=belief.truth_state,
            confidence=belief.confidence,
            conflict=belief.truth_state == TruthState.BOTH,
        )

    def retract(self, evidence_id: UUID) -> list[BeliefResult]:
        """Mark evidence invalid and recompute affected beliefs."""
        evidence = self._store.get_evidence(evidence_id)
        if evidence is None:
            return []

        evidence.valid = False
        self._store.update_evidence(evidence)
        affected = self._store.find_beliefs_using(evidence_id)

        results = []
        for belief in affected:
            # Update the evidence in the belief's ledger
            for ev in belief.evidence:
                if ev.id == evidence_id:
                    ev.valid = False

            belief.truth_state = compute_truth_state(
                belief.evidence, belief.belief_type
            )
            belief.confidence = compute_confidence(belief.evidence, belief.belief_type)
            belief.last_revised = datetime.now(timezone.utc)

            # Re-embed to preserve searchability (upsert without embedding
            # would overwrite with empty list, making belief unfindable).
            embedding = self._embedder.embed(belief.claim) if self._embedder else None
            self._store.upsert(belief, embedding=embedding)

            results.append(
                BeliefResult(
                    id=belief.id,
                    truth_state=belief.truth_state,
                    confidence=belief.confidence,
                    conflict=belief.truth_state == TruthState.BOTH,
                )
            )
        return results

    def explain(self, claim: str) -> ExplanationResult | None:
        """Return full justification chain for a belief."""
        if self._embedder is not None:
            embedding = self._embedder.embed(claim)
            matches = self._store.find_similar(embedding, threshold=0.8)
        else:
            matches = []

        if not matches:
            exact = self._store.find_by_claim(claim)
            if exact is None:
                return None
            matches = [(exact, 1.0)]

        belief = matches[0][0]

        # Recompute truth_state and confidence with current-time decay
        # (the persisted values may be stale if time has elapsed).
        truth_state = compute_truth_state(belief.evidence, belief.belief_type)
        confidence = compute_confidence(belief.evidence, belief.belief_type)

        active = [e for e in belief.evidence if e.valid]
        expired = [e for e in belief.evidence if not e.valid]

        return ExplanationResult(
            claim=belief.claim,
            truth_state=truth_state,
            confidence=confidence,
            supporting=[e for e in active if e.polarity == Polarity.SUPPORTS],
            attacking=[e for e in active if e.polarity == Polarity.ATTACKS],
            expired=expired,
        )

    def revise(self, belief_id: UUID, new_evidence: EvidenceInput) -> BeliefResult:
        """Add new evidence to an existing belief and recompute."""
        belief = self._store.get(belief_id)
        if belief is None:
            raise ValueError(f"Belief {belief_id} not found")

        ev = Evidence(
            belief_id=belief_id,
            source_ref=new_evidence.source_ref,
            content=new_evidence.content,
            polarity=Polarity(new_evidence.polarity),
            reliability=new_evidence.reliability,
            weight=new_evidence.weight,
            scope=new_evidence.scope,
        )
        belief.evidence.append(ev)

        belief.truth_state = compute_truth_state(belief.evidence, belief.belief_type)
        belief.confidence = compute_confidence(belief.evidence, belief.belief_type)
        belief.last_revised = datetime.now(timezone.utc)

        # Re-embed to preserve searchability.
        embedding = self._embedder.embed(belief.claim) if self._embedder else None
        self._store.upsert(belief, embedding=embedding)

        return BeliefResult(
            id=belief.id,
            truth_state=belief.truth_state,
            confidence=belief.confidence,
            conflict=belief.truth_state == TruthState.BOTH,
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        rank_alpha: float = 0.7,
        conflict_policy: ConflictPolicy = ConflictPolicy.SURFACE,
    ) -> list[tuple[Belief, float, float, float]]:
        """Search beliefs with ranking.

        Returns (belief, similarity, confidence, rank_score) tuples.
        """
        from mnemebrain_core.engine import apply_conflict_policy, rank_score

        if self._embedder is not None:
            embedding = self._embedder.embed(query)
            raw_matches = self._store.find_similar(embedding, threshold=0.3)
        else:
            raw_matches = self._store.find_by_text(query, limit=limit)

        scored = [
            (
                belief,
                sim,
                belief.confidence,
                rank_score(sim, belief.confidence, rank_alpha),
            )
            for belief, sim in raw_matches
        ]

        filtered_pairs = apply_conflict_policy(
            [(b, rs) for b, _, _, rs in scored],
            conflict_policy,
        )
        filtered_ids = {id(b) for b, _ in filtered_pairs}
        scored = [item for item in scored if id(item[0]) in filtered_ids]

        scored.sort(key=lambda x: x[3], reverse=True)
        return scored[:limit]

    def list_beliefs(
        self,
        truth_states: list[TruthState] | None = None,
        belief_types: list[BeliefType] | None = None,
        tag: str | None = None,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Belief], int]:
        """List beliefs with filtering."""
        return self._store.list_beliefs_filtered(
            truth_states=truth_states,
            belief_types=belief_types,
            tag=tag,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            limit=limit,
            offset=offset,
        )

    def close(self) -> None:
        """Close the underlying store."""
        self._store.close()
