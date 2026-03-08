"""Phase 5.2 — Inter-triple relations: typed edges between knowledge-graph triples.

Provides typed, weighted, directional edges between belief triples that
capture how triples relate to each other — e.g. triple A attacks triple B,
or triple C is an exception to triple D.

Stage 2 materializes the three core relation types: attacks, supports,
depends_on.  Stage 3 relation types (narrows, overrides, exception_to,
derived_from) are defined but reserved for future use.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class TripleRelationType(str, Enum):
    """Typed relation between two triples."""

    # Stage 2 — core relations
    ATTACKS = "attacks"
    SUPPORTS = "supports"
    DEPENDS_ON = "depends_on"

    # Stage 3 — extended relations (reserved)
    NARROWS = "narrows"
    OVERRIDES = "overrides"
    EXCEPTION_TO = "exception_to"
    DERIVED_FROM = "derived_from"


#: Stage 2 relation types that are currently active.
STAGE_2_RELATIONS: frozenset[TripleRelationType] = frozenset(
    {
        TripleRelationType.ATTACKS,
        TripleRelationType.SUPPORTS,
        TripleRelationType.DEPENDS_ON,
    }
)

#: Stage 3 relation types — reserved for future use.
STAGE_3_RELATIONS: frozenset[TripleRelationType] = frozenset(
    {
        TripleRelationType.NARROWS,
        TripleRelationType.OVERRIDES,
        TripleRelationType.EXCEPTION_TO,
        TripleRelationType.DERIVED_FROM,
    }
)


class TripleRelation(BaseModel):
    """A typed, weighted, directional edge between two triples.

    Attributes:
        source_triple_id: The triple asserting the relation.
        target_triple_id: The triple being related to.
        relation_type: The semantic type of this relation.
        weight: Strength of the relation (0.0–1.0).
        provenance: How this relation was established
            (e.g. "nli_conflict", "user_stated", "consolidation").
        id: Unique identifier for this relation edge.
        created_at: Timestamp of creation.
        active: Soft-delete flag.
    """

    id: UUID = Field(default_factory=uuid4)
    source_triple_id: UUID
    target_triple_id: UUID
    relation_type: TripleRelationType
    weight: float = Field(ge=0.0, le=1.0, default=1.0)
    provenance: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    active: bool = True

    model_config = {"frozen": False}

    @model_validator(mode="after")
    def _source_and_target_must_differ(self) -> TripleRelation:
        if self.source_triple_id == self.target_triple_id:
            raise ValueError("source_triple_id and target_triple_id must differ")
        return self


class RelationIndex:
    """In-memory index for TripleRelation edges.

    Provides O(1) lookup by relation ID and secondary indices by
    source/target triple ID for efficient graph traversal.
    """

    def __init__(self) -> None:
        self._by_id: dict[UUID, TripleRelation] = {}
        self._by_source: dict[UUID, list[UUID]] = defaultdict(list)
        self._by_target: dict[UUID, list[UUID]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(self, relation: TripleRelation) -> None:
        """Store a single relation edge."""
        self._by_id[relation.id] = relation
        self._by_source[relation.source_triple_id].append(relation.id)
        self._by_target[relation.target_triple_id].append(relation.id)

    def add_many(self, relations: list[TripleRelation]) -> None:
        """Store multiple relation edges."""
        for rel in relations:
            self.add(rel)

    def deactivate(self, relation_id: UUID) -> bool:
        """Soft-delete a relation by ID.

        Returns True if the relation was found and deactivated.
        """
        rel = self._by_id.get(relation_id)
        if rel is None or not rel.active:
            return False
        rel.active = False
        return True

    def deactivate_by_triple(self, triple_id: UUID) -> int:
        """Deactivate all relations where triple_id is source or target.

        Returns the number of relations deactivated.
        """
        count = 0
        seen: set[UUID] = set()
        for rel_id in self._by_source.get(triple_id, []):
            rel = self._by_id.get(rel_id)
            if rel is not None and rel.active:
                rel.active = False
                count += 1
                seen.add(rel_id)
        for rel_id in self._by_target.get(triple_id, []):
            if rel_id in seen:
                continue
            rel = self._by_id.get(rel_id)
            if rel is not None and rel.active:
                rel.active = False
                count += 1
        return count

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, relation_id: UUID) -> TripleRelation | None:
        """Retrieve a relation by ID, or None if not found."""
        return self._by_id.get(relation_id)

    def find_from(
        self,
        source_triple_id: UUID,
        *,
        relation_type: TripleRelationType | None = None,
        active_only: bool = True,
    ) -> list[TripleRelation]:
        """Return relations originating from source_triple_id."""
        results: list[TripleRelation] = []
        for rel_id in self._by_source.get(source_triple_id, []):
            rel = self._by_id.get(rel_id)
            if rel is None:
                continue
            if active_only and not rel.active:
                continue
            if relation_type is not None and rel.relation_type != relation_type:
                continue
            results.append(rel)
        return results

    def find_to(
        self,
        target_triple_id: UUID,
        *,
        relation_type: TripleRelationType | None = None,
        active_only: bool = True,
    ) -> list[TripleRelation]:
        """Return relations targeting target_triple_id."""
        results: list[TripleRelation] = []
        for rel_id in self._by_target.get(target_triple_id, []):
            rel = self._by_id.get(rel_id)
            if rel is None:
                continue
            if active_only and not rel.active:
                continue
            if relation_type is not None and rel.relation_type != relation_type:
                continue
            results.append(rel)
        return results

    def find_between(
        self,
        triple_a: UUID,
        triple_b: UUID,
        *,
        active_only: bool = True,
    ) -> list[TripleRelation]:
        """Return all relations between two triples (in either direction)."""
        results: list[TripleRelation] = []
        for rel_id in self._by_source.get(triple_a, []):
            rel = self._by_id.get(rel_id)
            if rel is None:
                continue
            if active_only and not rel.active:
                continue
            if rel.target_triple_id == triple_b:
                results.append(rel)
        for rel_id in self._by_source.get(triple_b, []):
            rel = self._by_id.get(rel_id)
            if rel is None:
                continue
            if active_only and not rel.active:
                continue
            if rel.target_triple_id == triple_a:
                results.append(rel)
        return results

    def count_by_type(
        self,
        *,
        active_only: bool = True,
    ) -> dict[TripleRelationType, int]:
        """Return a frequency map of relation types."""
        counts: dict[TripleRelationType, int] = defaultdict(int)
        for rel in self._by_id.values():
            if active_only and not rel.active:
                continue
            counts[rel.relation_type] += 1
        return dict(counts)
