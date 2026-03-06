"""BeliefMemory — the 4 core operations for belief management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from mnemebrain_core.engine import compute_confidence, compute_truth_state
from mnemebrain_core.models import Belief, BeliefType, Evidence, Polarity, TruthState
from mnemebrain_core.providers.base import EmbeddingProvider, EvidenceInput
from mnemebrain_core.store import KuzuGraphStore


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
    ) -> None:
        self._store = KuzuGraphStore(db_path)
        self._embedder = embedding_provider
        if self._embedder is None:
            try:
                from mnemebrain_core.providers.embeddings.sentence_transformers import (
                    SentenceTransformerProvider,
                )

                self._embedder = SentenceTransformerProvider()
            except ImportError:
                self._embedder = None  # Will fail at use-time with clear message

    def _get_embedder(self) -> EmbeddingProvider:
        if self._embedder is None:
            raise ImportError(
                "No embedding provider available. "
                "Install with: pip install mnemebrain-lite[embeddings]"
            )
        return self._embedder

    def believe(
        self,
        claim: str,
        evidence_items: list[EvidenceInput],
        belief_type: BeliefType = BeliefType.INFERENCE,
        tags: list[str] | None = None,
        source_agent: str = "",
    ) -> BeliefResult:
        """Store a new belief with evidence. Merges if similar belief exists."""
        embedding = self._get_embedder().embed(claim)
        existing = self._store.find_similar(embedding, threshold=0.92)

        if existing:
            belief = existing[0]
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
            self._store.upsert(belief)

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
        embedding = self._get_embedder().embed(claim)
        matches = self._store.find_similar(embedding, threshold=0.8)

        if not matches:
            exact = self._store.find_by_claim(claim)
            if exact is None:
                return None
            matches = [exact]

        belief = matches[0]
        active = [e for e in belief.evidence if e.valid]
        expired = [e for e in belief.evidence if not e.valid]

        return ExplanationResult(
            claim=belief.claim,
            truth_state=belief.truth_state,
            confidence=belief.confidence,
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

        self._store.upsert(belief)

        return BeliefResult(
            id=belief.id,
            truth_state=belief.truth_state,
            confidence=belief.confidence,
            conflict=belief.truth_state == TruthState.BOTH,
        )

    def close(self) -> None:
        """Close the underlying store."""
        self._store.close()
