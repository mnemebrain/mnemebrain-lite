"""Kuzu embedded graph store for beliefs and evidence."""

from __future__ import annotations

import json
from uuid import UUID

import kuzu
import numpy as np

from mnemebrain_core.models import Belief, Evidence


class KuzuGraphStore:
    """Persistent graph store backed by Kuzu embedded DB."""

    def __init__(self, db_path: str) -> None:
        self._db = kuzu.Database(db_path)
        self._conn = kuzu.Connection(self._db)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create node/rel tables if they don't exist."""
        try:
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
                "CREATE REL TABLE IF NOT EXISTS HAS_EVIDENCE("
                "FROM Belief TO EvidenceNode)"
            )
        except RuntimeError:  # pragma: no cover
            pass  # Tables already exist

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
            # Create relationship
            try:
                self._conn.execute(
                    "MATCH (b:Belief {id: $bid}), (e:EvidenceNode {id: $eid}) "
                    "MERGE (b)-[:HAS_EVIDENCE]->(e)",
                    parameters={"bid": bid, "eid": ev_id},
                )
            except RuntimeError:  # pragma: no cover
                pass  # Relationship already exists

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
    ) -> list[Belief]:
        """Find beliefs with similar embeddings using cosine similarity."""
        result = self._conn.execute(
            "MATCH (b:Belief) WHERE size(b.embedding) > 0 RETURN b.id, b.embedding"
        )
        matches = []
        query_vec = np.array(embedding)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        while result.has_next():
            row = result.get_next()
            stored_emb = np.array(row[1])
            stored_norm = np.linalg.norm(stored_emb)
            if stored_norm == 0:
                continue
            sim = float(np.dot(query_vec, stored_emb) / (query_norm * stored_norm))
            if sim >= threshold:
                belief = self.get(UUID(row[0]))
                if belief:
                    matches.append(belief)
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
