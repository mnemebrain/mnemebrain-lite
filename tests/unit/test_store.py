"""Tests for KuzuGraphStore — covers uncovered branches."""
import json
import os
import shutil
import tempfile
from uuid import uuid4

import pytest

from mnemebrain_core.models import Belief, BeliefType, Evidence, Polarity
from mnemebrain_core.store import KuzuGraphStore


@pytest.fixture
def store():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_store")
    s = KuzuGraphStore(db_path)
    yield s
    s.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestKuzuGraphStore:
    def test_init_schema_idempotent(self):
        """Calling _init_schema twice doesn't raise (covers except pass on line 38-39)."""
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_idem")
        s1 = KuzuGraphStore(db_path)
        # Second init on same DB — tables already exist, hits except pass
        s2 = KuzuGraphStore(db_path)
        s1.close()
        s2.close()
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_upsert_with_evidence_relationship_exists(self, store: KuzuGraphStore):
        """Upserting same belief+evidence twice covers the except pass on line 71-72."""
        ev = Evidence(
            belief_id=None,
            source_ref="msg_1",
            content="test",
            polarity=Polarity.SUPPORTS,
            reliability=0.8,
            weight=0.7,
        )
        belief = Belief(claim="test claim", evidence=[ev])
        store.upsert(belief, embedding=[0.1, 0.2, 0.3])
        # Upsert again — relationship already exists
        store.upsert(belief, embedding=[0.1, 0.2, 0.3])
        retrieved = store.get(belief.id)
        assert retrieved is not None

    def test_find_similar_zero_query_norm(self, store: KuzuGraphStore):
        """Zero-norm query embedding returns empty (line 137)."""
        result = store.find_similar([0.0, 0.0, 0.0])
        assert result == []

    def test_find_similar_zero_stored_norm(self, store: KuzuGraphStore):
        """Stored belief with zero-norm embedding is skipped (line 143-144)."""
        belief = Belief(claim="zero emb")
        store.upsert(belief, embedding=[0.0, 0.0, 0.0])
        result = store.find_similar([1.0, 0.0, 0.0])
        assert result == []

    def test_list_beliefs(self, store: KuzuGraphStore):
        """list_beliefs returns all stored beliefs (lines 154-161)."""
        b1 = Belief(claim="claim 1")
        b2 = Belief(claim="claim 2")
        store.upsert(b1)
        store.upsert(b2)
        beliefs = store.list_beliefs()
        assert len(beliefs) == 2
        claims = {b.claim for b in beliefs}
        assert claims == {"claim 1", "claim 2"}

    def test_find_by_claim_found(self, store: KuzuGraphStore):
        """find_by_claim returns matching belief."""
        belief = Belief(claim="unique claim xyz")
        store.upsert(belief)
        found = store.find_by_claim("unique claim xyz")
        assert found is not None
        assert found.id == belief.id

    def test_find_by_claim_not_found(self, store: KuzuGraphStore):
        """find_by_claim returns None when no match."""
        belief = Belief(claim="something")
        store.upsert(belief)
        found = store.find_by_claim("nonexistent")
        assert found is None

    def test_get_nonexistent(self, store: KuzuGraphStore):
        """get() returns None for nonexistent belief."""
        assert store.get(uuid4()) is None

    def test_get_evidence_nonexistent(self, store: KuzuGraphStore):
        """get_evidence() returns None for nonexistent evidence."""
        assert store.get_evidence(uuid4()) is None
