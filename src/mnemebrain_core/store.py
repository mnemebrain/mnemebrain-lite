"""Kuzu embedded graph store for beliefs and evidence."""

from __future__ import annotations

import json
from uuid import UUID

import kuzu
import numpy as np

from mnemebrain_core.models import Belief, BeliefType, Evidence, TruthState


class KuzuGraphStore:
    """Persistent graph store backed by Kuzu embedded DB."""

    def __init__(self, db_path: str, *, max_db_size: int = 0) -> None:
        kwargs: dict = {}
        if max_db_size > 0:
            kwargs["max_db_size"] = max_db_size
        self._db = kuzu.Database(db_path, **kwargs)
        self._conn = kuzu.Connection(self._db)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create node/rel tables if they don't exist."""
        self._conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS Belief("
            "id STRING, data STRING, embedding DOUBLE[], "
            "PRIMARY KEY(id))"
        )
        self._conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS EvidenceNode("
            "id STRING, belief_id STRING, data STRING, "
            "PRIMARY KEY(id))"
        )
        self._conn.execute(
            "CREATE REL TABLE IF NOT EXISTS HAS_EVIDENCE(FROM Belief TO EvidenceNode)"
        )

    def upsert(self, belief: Belief, embedding: list[float] | None = None) -> None:
        """Insert or update a belief and its evidence."""
        belief_data = belief.model_dump(mode="json")
        evidence_list = belief_data.pop("evidence", [])
        data_json = json.dumps(belief_data)
        bid = str(belief.id)

        emb = embedding if embedding else []

        self._conn.execute(
            "MERGE (b:Belief {id: $id}) SET b.data = $data, b.embedding = $embedding",
            parameters={"id": bid, "data": data_json, "embedding": emb},
        )

        for ev_data in evidence_list:
            ev_id = str(ev_data["id"])
            ev_json = json.dumps(ev_data)
            self._conn.execute(
                "MERGE (e:EvidenceNode {id: $id}) "
                "SET e.belief_id = $belief_id, e.data = $data",
                parameters={"id": ev_id, "belief_id": bid, "data": ev_json},
            )
            # Create relationship (MERGE is idempotent)
            self._conn.execute(
                "MATCH (b:Belief {id: $bid}), (e:EvidenceNode {id: $eid}) "
                "MERGE (b)-[:HAS_EVIDENCE]->(e)",
                parameters={"bid": bid, "eid": ev_id},
            )

    def get(self, belief_id: UUID) -> Belief | None:
        """Retrieve a belief by ID."""
        result = self._conn.execute(
            "MATCH (b:Belief {id: $id}) RETURN b.data",
            parameters={"id": str(belief_id)},
        )
        if not result.has_next():
            return None
        row = result.get_next()
        belief_data = json.loads(row[0])

        # Load evidence
        ev_result = self._conn.execute(
            "MATCH (b:Belief {id: $id})-[:HAS_EVIDENCE]->(e:EvidenceNode) "
            "RETURN e.data",
            parameters={"id": str(belief_id)},
        )
        evidence_list = []
        while ev_result.has_next():
            ev_row = ev_result.get_next()
            evidence_list.append(json.loads(ev_row[0]))

        belief_data["evidence"] = evidence_list
        return Belief.model_validate(belief_data)

    def get_evidence(self, evidence_id: UUID) -> Evidence | None:
        """Retrieve a single evidence item by ID."""
        result = self._conn.execute(
            "MATCH (e:EvidenceNode {id: $id}) RETURN e.data",
            parameters={"id": str(evidence_id)},
        )
        if not result.has_next():
            return None
        row = result.get_next()
        return Evidence.model_validate(json.loads(row[0]))

    def update_evidence(self, evidence: Evidence) -> None:
        """Update an evidence node's data in the store."""
        ev_json = json.dumps(evidence.model_dump(mode="json"))
        self._conn.execute(
            "MATCH (e:EvidenceNode {id: $id}) SET e.data = $data",
            parameters={"id": str(evidence.id), "data": ev_json},
        )

    def find_beliefs_using(self, evidence_id: UUID) -> list[Belief]:
        """Find all beliefs that reference a given evidence item."""
        result = self._conn.execute(
            "MATCH (b:Belief)-[:HAS_EVIDENCE]->(e:EvidenceNode {id: $eid}) RETURN b.id",
            parameters={"eid": str(evidence_id)},
        )
        beliefs = []
        while result.has_next():
            row = result.get_next()
            belief = self.get(UUID(row[0]))
            if belief:
                beliefs.append(belief)
        return beliefs

    def find_similar(
        self, embedding: list[float], threshold: float = 0.92
    ) -> list[tuple[Belief, float]]:
        """Find beliefs with similar embeddings. Returns (belief, similarity) pairs."""
        result = self._conn.execute(
            "MATCH (b:Belief) WHERE size(b.embedding) > 0 RETURN b.id, b.embedding"
        )
        matches: list[tuple[Belief, float]] = []
        query_vec = np.array(embedding)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        while result.has_next():
            row = result.get_next()
            stored_emb = np.array(row[1])
            if stored_emb.shape != query_vec.shape:
                continue  # skip embeddings from a different provider
            stored_norm = np.linalg.norm(stored_emb)
            if stored_norm == 0:
                continue
            sim = float(np.dot(query_vec, stored_emb) / (query_norm * stored_norm))
            if sim >= threshold:
                belief = self.get(UUID(row[0]))
                if belief:
                    matches.append((belief, sim))
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def list_beliefs(self) -> list[Belief]:
        """List all beliefs in the store."""
        result = self._conn.execute("MATCH (b:Belief) RETURN b.id")
        beliefs = []
        while result.has_next():
            row = result.get_next()
            belief = self.get(UUID(row[0]))
            if belief:
                beliefs.append(belief)
        return beliefs

    def list_beliefs_filtered(
        self,
        truth_states: list[TruthState] | None = None,
        belief_types: list[BeliefType] | None = None,
        tag: str | None = None,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Belief], int]:
        """List beliefs with filtering. Returns (beliefs, total_count)."""
        all_beliefs = self.list_beliefs()

        filtered = all_beliefs
        if truth_states:
            filtered = [b for b in filtered if b.truth_state in truth_states]
        if belief_types:
            filtered = [b for b in filtered if b.belief_type in belief_types]
        if tag:
            filtered = [b for b in filtered if tag in b.tags]
        filtered = [
            b for b in filtered if min_confidence <= b.confidence <= max_confidence
        ]

        filtered.sort(key=lambda b: b.confidence, reverse=True)
        total = len(filtered)
        return filtered[offset : offset + limit], total

    def find_by_text(self, query: str, limit: int = 10) -> list[tuple[Belief, float]]:
        """Find beliefs by case-insensitive substring match on claim text.

        Returns (belief, score) pairs sorted by relevance score.
        """
        result = self._conn.execute("MATCH (b:Belief) RETURN b.id, b.data")
        matches: list[tuple[Belief, float]] = []
        query_lower = query.lower()

        while result.has_next():
            row = result.get_next()
            data = json.loads(row[1])
            claim = data.get("claim", "")
            if query_lower in claim.lower():
                score = len(query) / len(claim) if claim else 0.0
                belief = self.get(UUID(row[0]))
                if belief:
                    matches.append((belief, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:limit]

    def find_by_claim(self, claim: str) -> Belief | None:
        """Find a belief by exact claim match."""
        result = self._conn.execute("MATCH (b:Belief) RETURN b.id, b.data")
        while result.has_next():
            row = result.get_next()
            data = json.loads(row[1])
            if data.get("claim") == claim:
                return self.get(UUID(row[0]))
        return None

    def close(self) -> None:
        """Close the database connection."""
        # Kuzu handles cleanup automatically
        pass
