"""Tests for KuzuGraphStore — covers uncovered branches."""

import os
import shutil
import tempfile
from uuid import uuid4

import pytest

from mnemebrain_core.models import Belief, Evidence, Polarity
from mnemebrain_core.store import KuzuGraphStore


@pytest.fixture
def store():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_store")
    s = KuzuGraphStore(db_path, max_db_size=1 << 30)
    yield s
    s.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestKuzuGraphStore:
    def test_init_schema_idempotent(self):
        """Calling _init_schema twice doesn't raise (covers except pass on line 38-39)."""
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_idem")
        s1 = KuzuGraphStore(db_path, max_db_size=1 << 30)
        # Second init on same DB — tables already exist, hits except pass
        s2 = KuzuGraphStore(db_path, max_db_size=1 << 30)
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

    # ------------------------------------------------------------------
    # find_similar — shape mismatch skip (line 148-149)
    # ------------------------------------------------------------------

    def test_find_similar_shape_mismatch_skipped(self, store: KuzuGraphStore):
        """Stored embedding with different dim than query is skipped (line 149)."""
        belief = Belief(claim="3-dim belief")
        store.upsert(belief, embedding=[0.1, 0.2, 0.3])  # 3-dim stored

        # Query with 5-dim — shapes differ, the stored belief must be skipped
        result = store.find_similar([0.1, 0.2, 0.3, 0.4, 0.5])
        assert result == []

    # ------------------------------------------------------------------
    # list_beliefs_filtered (lines 183-198)
    # ------------------------------------------------------------------

    def _make_belief(
        self,
        claim: str,
        truth_state=None,
        belief_type=None,
        tags=None,
        confidence: float = 0.5,
    ) -> Belief:

        kwargs: dict = {"claim": claim, "confidence": confidence}
        if truth_state is not None:
            kwargs["truth_state"] = truth_state
        if belief_type is not None:
            kwargs["belief_type"] = belief_type
        if tags is not None:
            kwargs["tags"] = tags
        return Belief(**kwargs)

    def test_list_beliefs_filtered_truth_state(self, store: KuzuGraphStore):
        """Filter by truth_state returns only matching beliefs."""
        from mnemebrain_core.models import TruthState

        b_true = self._make_belief("true belief", truth_state=TruthState.TRUE)
        b_false = self._make_belief("false belief", truth_state=TruthState.FALSE)
        store.upsert(b_true)
        store.upsert(b_false)

        results, total = store.list_beliefs_filtered(truth_states=[TruthState.TRUE])
        assert total == 1
        assert results[0].claim == "true belief"

    def test_list_beliefs_filtered_belief_type(self, store: KuzuGraphStore):
        """Filter by belief_type returns only matching beliefs."""
        from mnemebrain_core.models import BeliefType

        b_fact = self._make_belief("fact belief", belief_type=BeliefType.FACT)
        b_pred = self._make_belief("pred belief", belief_type=BeliefType.PREDICTION)
        store.upsert(b_fact)
        store.upsert(b_pred)

        results, total = store.list_beliefs_filtered(
            belief_types=[BeliefType.PREDICTION]
        )
        assert total == 1
        assert results[0].claim == "pred belief"

    def test_list_beliefs_filtered_tag(self, store: KuzuGraphStore):
        """Filter by tag returns only beliefs containing that tag."""
        b_tagged = self._make_belief("tagged", tags=["science", "health"])
        b_other = self._make_belief("untagged", tags=["sports"])
        store.upsert(b_tagged)
        store.upsert(b_other)

        results, total = store.list_beliefs_filtered(tag="science")
        assert total == 1
        assert results[0].claim == "tagged"

    def test_list_beliefs_filtered_confidence_range(self, store: KuzuGraphStore):
        """Filter by confidence range excludes out-of-range beliefs."""
        b_low = self._make_belief("low conf", confidence=0.2)
        b_high = self._make_belief("high conf", confidence=0.9)
        store.upsert(b_low)
        store.upsert(b_high)

        results, total = store.list_beliefs_filtered(
            min_confidence=0.5, max_confidence=1.0
        )
        assert total == 1
        assert results[0].claim == "high conf"

    def test_list_beliefs_filtered_limit_and_offset(self, store: KuzuGraphStore):
        """limit and offset slice the result set."""
        for i in range(5):
            store.upsert(self._make_belief(f"belief {i}", confidence=float(i) / 10))

        results, total = store.list_beliefs_filtered(limit=2, offset=1)
        assert total == 5
        assert len(results) == 2

    def test_list_beliefs_filtered_no_filters(self, store: KuzuGraphStore):
        """No filters returns all beliefs sorted by confidence descending."""
        b1 = self._make_belief("low", confidence=0.3)
        b2 = self._make_belief("high", confidence=0.8)
        store.upsert(b1)
        store.upsert(b2)

        results, total = store.list_beliefs_filtered()
        assert total == 2
        # Sorted by confidence descending
        assert results[0].confidence >= results[1].confidence

    # ------------------------------------------------------------------
    # update_evidence (line 207)
    # ------------------------------------------------------------------

    def test_update_evidence(self, store: KuzuGraphStore):
        """update_evidence persists new data and get_evidence reflects it."""
        ev = Evidence(
            belief_id=None,
            source_ref="src_1",
            content="original content",
            polarity=Polarity.SUPPORTS,
            reliability=0.7,
            weight=0.6,
        )
        belief = Belief(claim="belief with evidence", evidence=[ev])
        store.upsert(belief)

        # Mutate the evidence object and update
        ev.content = "updated content"
        ev.reliability = 0.95
        store.update_evidence(ev)

        retrieved_ev = store.get_evidence(ev.id)
        assert retrieved_ev is not None
        assert retrieved_ev.content == "updated content"
        assert retrieved_ev.reliability == 0.95

    # ------------------------------------------------------------------
    # close (lines 210-213)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # find_beliefs_using (lines 118-130)
    # ------------------------------------------------------------------

    def test_find_beliefs_using(self, store: KuzuGraphStore):
        """find_beliefs_using returns beliefs linked to a given evidence."""
        ev = Evidence(
            belief_id=None,
            source_ref="msg_fb",
            content="finding evidence",
            polarity=Polarity.SUPPORTS,
            reliability=0.9,
            weight=0.8,
        )
        belief = Belief(claim="belief using evidence", evidence=[ev])
        store.upsert(belief, embedding=[0.5, 0.5, 0.5])

        found = store.find_beliefs_using(ev.id)
        assert len(found) == 1
        assert found[0].id == belief.id

    def test_find_beliefs_using_nonexistent(self, store: KuzuGraphStore):
        """find_beliefs_using returns empty for nonexistent evidence."""
        found = store.find_beliefs_using(uuid4())
        assert found == []

    def test_close_is_callable(self, store: KuzuGraphStore):
        """close() is a no-op but must be reachable for coverage."""
        # Should not raise
        store.close()
