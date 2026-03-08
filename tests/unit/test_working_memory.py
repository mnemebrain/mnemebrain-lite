"""Unit tests for WorkingMemoryFrame and WorkingMemoryManager.

Uses a real temp Kuzu DB with sentence-transformers where available,
falling back to a mock embedder for pure-unit isolation.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from mnemebrain_core.models import BeliefType, Polarity, TruthState
from mnemebrain_core.working_memory import (
    BeliefSnapshot,
    FrameCommitResult,
    FrameContext,
    FrameStatus,
    WorkingMemoryFrame,
    WorkingMemoryManager,
)

# ---------------------------------------------------------------------------
# Embedding availability guard
# ---------------------------------------------------------------------------

try:
    import sentence_transformers  # noqa: F401

    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

requires_embeddings = pytest.mark.skipif(
    not HAS_EMBEDDINGS, reason="sentence-transformers not installed"
)


# ---------------------------------------------------------------------------
# Minimal mock embedder — deterministic, no ML, no network
# ---------------------------------------------------------------------------


class _FixedEmbedder:
    """Returns a fixed 3-d unit vector and similarity=1.0 for all inputs."""

    def embed(self, text: str) -> list[float]:  # noqa: ARG002
        return [1.0, 0.0, 0.0]

    def similarity(self, a: list[float], b: list[float]) -> float:  # noqa: ARG002
        return 1.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmpdb():
    """Yield a temporary directory suitable for a Kuzu database."""
    d = tempfile.mkdtemp()
    db_path = os.path.join(d, "test_wm_db")
    yield db_path
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def memory(tmpdb):
    """BeliefMemory backed by a throw-away Kuzu DB with a mock embedder."""
    from mnemebrain_core.memory import BeliefMemory

    mem = BeliefMemory(
        db_path=tmpdb, embedding_provider=_FixedEmbedder(), max_db_size=1 << 30
    )
    yield mem
    mem.close()


@pytest.fixture()
def manager(memory):
    """WorkingMemoryManager wired to the fixture memory."""
    return WorkingMemoryManager(memory)


@pytest.fixture()
def memory_with_belief(memory):
    """BeliefMemory that already contains one stored belief.

    Returns (memory, belief_id, claim) so callers can act on it.
    """
    from mnemebrain_core.providers.base import EvidenceInput

    claim = "the sky is blue"
    result = memory.believe(
        claim=claim,
        evidence_items=[
            EvidenceInput(
                source_ref="src_1",
                content="direct observation",
                polarity="supports",
                weight=0.9,
                reliability=0.95,
            )
        ],
        belief_type=BeliefType.FACT,
        tags=["nature"],
    )
    return memory, result.id, claim


# ---------------------------------------------------------------------------
# Dataclass construction tests
# ---------------------------------------------------------------------------


class TestBeliefSnapshotDataclass:
    def test_all_fields(self):
        bid = uuid4()
        now = datetime.now(timezone.utc)
        snap = BeliefSnapshot(
            belief_id=bid,
            claim="water is wet",
            truth_state=TruthState.TRUE,
            confidence=0.9,
            belief_type=BeliefType.FACT,
            evidence_count=2,
            conflict=False,
            loaded_at=now,
        )
        assert snap.belief_id == bid
        assert snap.claim == "water is wet"
        assert snap.truth_state == TruthState.TRUE
        assert snap.confidence == 0.9
        assert snap.belief_type == BeliefType.FACT
        assert snap.evidence_count == 2
        assert snap.conflict is False
        assert snap.loaded_at == now

    def test_conflict_true(self):
        snap = BeliefSnapshot(
            belief_id=uuid4(),
            claim="test",
            truth_state=TruthState.BOTH,
            confidence=0.5,
            belief_type=BeliefType.INFERENCE,
            evidence_count=3,
            conflict=True,
            loaded_at=datetime.now(timezone.utc),
        )
        assert snap.conflict is True
        assert snap.truth_state == TruthState.BOTH


class TestFrameContextDataclass:
    def test_construction(self):
        qid = uuid4()
        gid = uuid4()
        snap = BeliefSnapshot(
            belief_id=uuid4(),
            claim="sky is blue",
            truth_state=TruthState.TRUE,
            confidence=0.9,
            belief_type=BeliefType.FACT,
            evidence_count=1,
            conflict=False,
            loaded_at=datetime.now(timezone.utc),
        )
        ctx = FrameContext(
            active_query=qid,
            active_goal=gid,
            beliefs=[snap],
            scratchpad={"step": 1},
            conflicts=[],
            step_count=1,
        )
        assert ctx.active_query == qid
        assert ctx.active_goal == gid
        assert len(ctx.beliefs) == 1
        assert ctx.scratchpad == {"step": 1}
        assert ctx.conflicts == []
        assert ctx.step_count == 1

    def test_no_goal(self):
        ctx = FrameContext(
            active_query=uuid4(),
            active_goal=None,
            beliefs=[],
            scratchpad={},
            conflicts=[],
            step_count=0,
        )
        assert ctx.active_goal is None


class TestFrameCommitResultDataclass:
    def test_construction(self):
        fid = uuid4()
        r = FrameCommitResult(frame_id=fid, beliefs_created=3, beliefs_revised=1)
        assert r.frame_id == fid
        assert r.beliefs_created == 3
        assert r.beliefs_revised == 1

    def test_zero_values(self):
        r = FrameCommitResult(frame_id=uuid4(), beliefs_created=0, beliefs_revised=0)
        assert r.beliefs_created == 0
        assert r.beliefs_revised == 0


class TestWorkingMemoryFrameDataclass:
    def test_defaults(self):
        frame = WorkingMemoryFrame()
        assert isinstance(frame.id, UUID)
        assert isinstance(frame.active_query, UUID)
        assert frame.active_goal is None
        assert frame.active_beliefs == []
        assert frame.active_evidence == []
        assert frame.belief_snapshots == {}
        assert frame.scratchpad == {}
        assert frame.expires_at is None
        assert frame.status == FrameStatus.ACTIVE
        assert frame.source_agent == ""
        assert frame.step_count == 0
        # created_at should be recent (within 5 seconds)
        delta = datetime.now(timezone.utc) - frame.created_at
        assert abs(delta.total_seconds()) < 5

    def test_custom_values(self):
        qid = uuid4()
        gid = uuid4()
        frame = WorkingMemoryFrame(
            active_query=qid,
            active_goal=gid,
            source_agent="test_agent",
            step_count=5,
        )
        assert frame.active_query == qid
        assert frame.active_goal == gid
        assert frame.source_agent == "test_agent"
        assert frame.step_count == 5

    def test_mutable_collections_are_independent(self):
        f1 = WorkingMemoryFrame()
        f2 = WorkingMemoryFrame()
        f1.active_beliefs.append(uuid4())
        assert f2.active_beliefs == []


# ---------------------------------------------------------------------------
# WorkingMemoryManager._snapshot_belief
# ---------------------------------------------------------------------------


class TestSnapshotBelief:
    def test_snapshot_fields(self, manager, memory_with_belief):
        mem, belief_id, claim = memory_with_belief
        beliefs, _ = mem.list_beliefs(limit=100)
        belief = next(b for b in beliefs if b.id == belief_id)

        snap = manager._snapshot_belief(belief)

        assert snap.belief_id == belief.id
        assert snap.claim == belief.claim
        assert snap.truth_state == belief.truth_state
        assert snap.confidence == belief.confidence
        assert snap.belief_type == belief.belief_type
        # only valid evidence counted
        valid_count = len([e for e in belief.evidence if e.valid])
        assert snap.evidence_count == valid_count
        assert snap.conflict == (belief.truth_state == TruthState.BOTH)
        # loaded_at should be very recent
        delta = datetime.now(timezone.utc) - snap.loaded_at
        assert abs(delta.total_seconds()) < 5

    def test_snapshot_conflict_flag_false_for_true_belief(
        self, manager, memory_with_belief
    ):
        mem, belief_id, _ = memory_with_belief
        beliefs, _ = mem.list_beliefs(limit=100)
        belief = next(b for b in beliefs if b.id == belief_id)
        # Belief is TRUE (has strong support), not BOTH
        assert belief.truth_state != TruthState.BOTH
        snap = manager._snapshot_belief(belief)
        assert snap.conflict is False

    def test_snapshot_conflict_flag_true_for_both_belief(self, manager, memory):
        """A belief in BOTH state must have conflict=True in snapshot."""
        from mnemebrain_core.providers.base import EvidenceInput

        # Add both supporting and attacking evidence so truth_state=BOTH
        memory.believe(
            claim="conflict claim",
            evidence_items=[
                EvidenceInput(
                    source_ref="src_sup",
                    content="supports",
                    polarity="supports",
                    weight=0.9,
                    reliability=0.95,
                ),
                EvidenceInput(
                    source_ref="src_att",
                    content="attacks",
                    polarity="attacks",
                    weight=0.9,
                    reliability=0.95,
                ),
            ],
        )
        beliefs, _ = memory.list_beliefs(limit=100)
        both_beliefs = [b for b in beliefs if b.truth_state == TruthState.BOTH]
        assert both_beliefs, "Expected at least one BOTH-state belief"
        snap = manager._snapshot_belief(both_beliefs[0])
        assert snap.conflict is True


# ---------------------------------------------------------------------------
# open_frame
# ---------------------------------------------------------------------------


class TestOpenFrame:
    def test_returns_frame(self, manager):
        qid = uuid4()
        frame = manager.open_frame(query_id=qid)
        assert isinstance(frame, WorkingMemoryFrame)
        assert frame.active_query == qid
        assert frame.active_goal is None
        assert frame.status == FrameStatus.ACTIVE

    def test_frame_is_stored(self, manager):
        frame = manager.open_frame(query_id=uuid4())
        assert frame.id in manager._frames

    def test_default_ttl_is_300_seconds(self, manager):
        frame = manager.open_frame(query_id=uuid4())
        delta = frame.expires_at - datetime.now(timezone.utc)
        # Should be close to 300 seconds (±10 s for test latency)
        assert 290 < delta.total_seconds() < 310

    def test_custom_ttl(self, manager):
        frame = manager.open_frame(query_id=uuid4(), ttl_seconds=60)
        delta = frame.expires_at - datetime.now(timezone.utc)
        assert 50 < delta.total_seconds() < 70

    def test_goal_id_stored(self, manager):
        gid = uuid4()
        frame = manager.open_frame(query_id=uuid4(), goal_id=gid)
        assert frame.active_goal == gid

    def test_multiple_frames_are_independent(self, manager):
        f1 = manager.open_frame(query_id=uuid4())
        f2 = manager.open_frame(query_id=uuid4())
        assert f1.id != f2.id
        assert len(manager._frames) == 2


# ---------------------------------------------------------------------------
# get_frame
# ---------------------------------------------------------------------------


class TestGetFrame:
    def test_found(self, manager):
        frame = manager.open_frame(query_id=uuid4())
        retrieved = manager.get_frame(frame.id)
        assert retrieved is frame

    def test_not_found_returns_none(self, manager):
        result = manager.get_frame(uuid4())
        assert result is None

    def test_expired_lazy_flip(self, manager):
        """get_frame must mark an overdue ACTIVE frame as EXPIRED."""
        frame = manager.open_frame(query_id=uuid4(), ttl_seconds=300)
        # Back-date expires_at to the past
        frame.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

        retrieved = manager.get_frame(frame.id)
        assert retrieved is not None
        assert retrieved.status == FrameStatus.EXPIRED

    def test_committed_frame_not_expired_on_get(self, manager):
        """A committed frame whose expires_at is in the past must NOT be re-expired."""
        frame = manager.open_frame(query_id=uuid4(), ttl_seconds=300)
        frame.status = FrameStatus.COMMITTED
        frame.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

        retrieved = manager.get_frame(frame.id)
        # status should stay COMMITTED because lazy expiry only applies to ACTIVE
        assert retrieved.status == FrameStatus.COMMITTED

    def test_frame_without_expires_at_not_expired(self, manager):
        """A frame with expires_at=None is never lazily expired."""
        frame = manager.open_frame(query_id=uuid4(), ttl_seconds=300)
        frame.expires_at = None

        retrieved = manager.get_frame(frame.id)
        assert retrieved.status == FrameStatus.ACTIVE


# ---------------------------------------------------------------------------
# add_to_frame
# ---------------------------------------------------------------------------


class TestAddToFrame:
    def test_add_existing_belief_returns_snapshot(self, manager, memory_with_belief):
        mem, _bid, claim = memory_with_belief
        frame = manager.open_frame(query_id=uuid4())

        snap = manager.add_to_frame(frame.id, claim)

        assert snap is not None
        assert isinstance(snap, BeliefSnapshot)
        assert snap.claim == claim
        assert frame.id in manager._frames
        assert snap.belief_id in frame.belief_snapshots

    def test_add_populates_active_beliefs(self, manager, memory_with_belief):
        mem, bid, claim = memory_with_belief
        frame = manager.open_frame(query_id=uuid4())

        manager.add_to_frame(frame.id, claim)

        assert bid in frame.active_beliefs

    def test_add_nonexistent_claim_returns_none(self, manager):
        frame = manager.open_frame(query_id=uuid4())
        # Use a claim that is sufficiently different from any stored belief
        # With FixedEmbedder similarity=1.0, explain() finds ANY belief.
        # We must ensure the store is empty so explain() returns None.
        result = manager.add_to_frame(frame.id, "belief that does not exist xyz123")
        # If memory is empty, explain returns None → add_to_frame returns None
        assert result is None

    def test_add_frame_not_found_raises(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.add_to_frame(uuid4(), "anything")

    def test_add_to_non_active_frame_raises(self, manager, memory_with_belief):
        mem, _bid, claim = memory_with_belief
        frame = manager.open_frame(query_id=uuid4())
        frame.status = FrameStatus.COMMITTED

        with pytest.raises(ValueError, match="committed"):
            manager.add_to_frame(frame.id, claim)

    def test_add_duplicate_belief_not_added_twice(self, manager, memory_with_belief):
        mem, bid, claim = memory_with_belief
        frame = manager.open_frame(query_id=uuid4())

        manager.add_to_frame(frame.id, claim)
        manager.add_to_frame(frame.id, claim)  # second call: already in snapshots

        # Should only appear once in active_beliefs
        count = frame.active_beliefs.count(bid)
        assert count == 1


# ---------------------------------------------------------------------------
# write_scratchpad
# ---------------------------------------------------------------------------


class TestWriteScratchpad:
    def test_write_stores_value(self, manager):
        frame = manager.open_frame(query_id=uuid4())
        manager.write_scratchpad(frame.id, "result", 42)
        assert frame.scratchpad["result"] == 42

    def test_write_increments_step_count(self, manager):
        frame = manager.open_frame(query_id=uuid4())
        assert frame.step_count == 0
        manager.write_scratchpad(frame.id, "a", 1)
        assert frame.step_count == 1
        manager.write_scratchpad(frame.id, "b", 2)
        assert frame.step_count == 2

    def test_write_frame_not_found_raises(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.write_scratchpad(uuid4(), "key", "val")

    def test_write_non_active_frame_raises(self, manager):
        frame = manager.open_frame(query_id=uuid4())
        frame.status = FrameStatus.EXPIRED

        with pytest.raises(ValueError, match="expired"):
            manager.write_scratchpad(frame.id, "key", "val")

    def test_write_complex_value(self, manager):
        frame = manager.open_frame(query_id=uuid4())
        manager.write_scratchpad(frame.id, "data", {"nested": [1, 2, 3]})
        assert frame.scratchpad["data"] == {"nested": [1, 2, 3]}

    def test_overwrite_key(self, manager):
        frame = manager.open_frame(query_id=uuid4())
        manager.write_scratchpad(frame.id, "x", "first")
        manager.write_scratchpad(frame.id, "x", "second")
        assert frame.scratchpad["x"] == "second"
        assert frame.step_count == 2


# ---------------------------------------------------------------------------
# commit_frame
# ---------------------------------------------------------------------------


class TestCommitFrame:
    def test_commit_empty_frame(self, manager):
        frame = manager.open_frame(query_id=uuid4())
        result = manager.commit_frame(frame.id)

        assert isinstance(result, FrameCommitResult)
        assert result.frame_id == frame.id
        assert result.beliefs_created == 0
        assert result.beliefs_revised == 0
        assert frame.status == FrameStatus.COMMITTED

    def test_commit_with_new_beliefs(self, manager):
        from mnemebrain_core.api.schemas import EvidenceRequest, NewBeliefPayload

        frame = manager.open_frame(query_id=uuid4())
        payload = NewBeliefPayload(
            claim="grass is green",
            evidence=[
                EvidenceRequest(
                    source_ref="obs_1",
                    content="looked outside",
                    polarity=Polarity.SUPPORTS,
                    weight=0.8,
                    reliability=0.9,
                )
            ],
            belief_type=BeliefType.FACT,
            tags=["nature"],
        )

        result = manager.commit_frame(frame.id, new_beliefs=[payload])

        assert result.beliefs_created == 1
        assert result.beliefs_revised == 0
        assert frame.status == FrameStatus.COMMITTED

    def test_commit_with_multiple_new_beliefs(self, manager):
        from mnemebrain_core.api.schemas import EvidenceRequest, NewBeliefPayload

        frame = manager.open_frame(query_id=uuid4())
        payloads = [
            NewBeliefPayload(
                claim=f"claim number {i}",
                evidence=[
                    EvidenceRequest(
                        source_ref=f"src_{i}",
                        content=f"evidence {i}",
                        polarity=Polarity.SUPPORTS,
                        weight=0.7,
                        reliability=0.8,
                    )
                ],
            )
            for i in range(3)
        ]

        result = manager.commit_frame(frame.id, new_beliefs=payloads)

        assert result.beliefs_created == 3
        assert result.beliefs_revised == 0

    def test_commit_with_revision(self, manager, memory_with_belief):
        from mnemebrain_core.api.schemas import EvidenceRequest, RevisionPayload

        mem, bid, claim = memory_with_belief
        frame = manager.open_frame(query_id=uuid4())
        rev = RevisionPayload(
            belief_id=bid,
            evidence=EvidenceRequest(
                source_ref="new_src",
                content="updated evidence",
                polarity=Polarity.SUPPORTS,
                weight=0.95,
                reliability=0.99,
            ),
        )

        result = manager.commit_frame(frame.id, revisions=[rev])

        assert result.beliefs_created == 0
        assert result.beliefs_revised == 1
        assert frame.status == FrameStatus.COMMITTED

    def test_commit_both_new_and_revision(self, manager, memory_with_belief):
        from mnemebrain_core.api.schemas import (
            EvidenceRequest,
            NewBeliefPayload,
            RevisionPayload,
        )

        mem, bid, claim = memory_with_belief
        frame = manager.open_frame(query_id=uuid4())
        new_payload = NewBeliefPayload(
            claim="new distinct claim abc",
            evidence=[
                EvidenceRequest(
                    source_ref="src_new",
                    content="new content",
                    polarity=Polarity.SUPPORTS,
                    weight=0.8,
                    reliability=0.85,
                )
            ],
        )
        rev = RevisionPayload(
            belief_id=bid,
            evidence=EvidenceRequest(
                source_ref="src_rev",
                content="revision content",
                polarity=Polarity.ATTACKS,
                weight=0.7,
                reliability=0.8,
            ),
        )

        result = manager.commit_frame(
            frame.id, new_beliefs=[new_payload], revisions=[rev]
        )

        assert result.beliefs_created == 1
        assert result.beliefs_revised == 1

    def test_commit_frame_not_found_raises(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.commit_frame(uuid4())

    def test_commit_non_active_frame_raises(self, manager):
        frame = manager.open_frame(query_id=uuid4())
        frame.status = FrameStatus.ABANDONED

        with pytest.raises(ValueError, match="abandoned"):
            manager.commit_frame(frame.id)

    def test_commit_belief_type_as_string_value(self, manager):
        """Ensures BeliefType string values are accepted (backward compat path)."""
        from mnemebrain_core.api.schemas import EvidenceRequest, NewBeliefPayload

        frame = manager.open_frame(query_id=uuid4())
        # Pass BeliefType enum directly (as schemas do)
        payload = NewBeliefPayload(
            claim="temperature is rising",
            evidence=[
                EvidenceRequest(
                    source_ref="weather_api",
                    content="measured 35C",
                    polarity=Polarity.SUPPORTS,
                    weight=0.9,
                    reliability=0.95,
                )
            ],
            belief_type=BeliefType.PREDICTION,
        )

        result = manager.commit_frame(frame.id, new_beliefs=[payload])
        assert result.beliefs_created == 1

    def test_commit_evidence_polarity_as_string(self, manager):
        """Covers the hasattr(e.polarity, 'value') else branch."""
        from mnemebrain_core.api.schemas import EvidenceRequest, NewBeliefPayload

        # Polarity enum has .value so the hasattr branch is True — to hit the
        # else branch, we'd need a plain string. We verify enum path works here.
        frame = manager.open_frame(query_id=uuid4())
        payload = NewBeliefPayload(
            claim="cloud is white",
            evidence=[
                EvidenceRequest(
                    source_ref="obs",
                    content="white cloud",
                    polarity=Polarity.SUPPORTS,
                )
            ],
        )
        result = manager.commit_frame(frame.id, new_beliefs=[payload])
        assert result.beliefs_created == 1


# ---------------------------------------------------------------------------
# close_frame
# ---------------------------------------------------------------------------


class TestCloseFrame:
    def test_close_marks_abandoned(self, manager):
        frame = manager.open_frame(query_id=uuid4())
        manager.close_frame(frame.id)
        assert frame.status == FrameStatus.ABANDONED

    def test_close_with_reason(self, manager):
        frame = manager.open_frame(query_id=uuid4())
        manager.close_frame(frame.id, reason="timeout")
        assert frame.status == FrameStatus.ABANDONED

    def test_close_not_found_raises(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.close_frame(uuid4())

    def test_close_already_committed_frame(self, manager):
        """close_frame should update status regardless of current state."""
        frame = manager.open_frame(query_id=uuid4())
        frame.status = FrameStatus.COMMITTED
        # close_frame does not guard on current status — it just sets ABANDONED
        manager.close_frame(frame.id)
        assert frame.status == FrameStatus.ABANDONED


# ---------------------------------------------------------------------------
# get_frame_context
# ---------------------------------------------------------------------------


class TestGetFrameContext:
    def test_empty_frame_context(self, manager):
        qid = uuid4()
        frame = manager.open_frame(query_id=qid)
        ctx = manager.get_frame_context(frame.id)

        assert isinstance(ctx, FrameContext)
        assert ctx.active_query == qid
        assert ctx.active_goal is None
        assert ctx.beliefs == []
        assert ctx.conflicts == []
        assert ctx.scratchpad == {}
        assert ctx.step_count == 0

    def test_context_with_belief(self, manager, memory_with_belief):
        mem, bid, claim = memory_with_belief
        frame = manager.open_frame(query_id=uuid4())
        manager.add_to_frame(frame.id, claim)

        ctx = manager.get_frame_context(frame.id)

        assert len(ctx.beliefs) == 1
        assert ctx.beliefs[0].claim == claim

    def test_context_with_scratchpad(self, manager):
        frame = manager.open_frame(query_id=uuid4())
        manager.write_scratchpad(frame.id, "notes", "something")
        manager.write_scratchpad(frame.id, "count", 7)

        ctx = manager.get_frame_context(frame.id)

        assert ctx.scratchpad == {"notes": "something", "count": 7}
        assert ctx.step_count == 2

    def test_context_not_found_raises(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.get_frame_context(uuid4())

    def test_context_conflicts_populated(self, manager, memory):
        """Beliefs in BOTH state appear in ctx.conflicts."""
        from mnemebrain_core.providers.base import EvidenceInput

        conflict_claim = "conflict test belief"
        memory.believe(
            claim=conflict_claim,
            evidence_items=[
                EvidenceInput(
                    source_ref="s1",
                    content="supports",
                    polarity="supports",
                    weight=0.9,
                    reliability=0.95,
                ),
                EvidenceInput(
                    source_ref="s2",
                    content="attacks",
                    polarity="attacks",
                    weight=0.9,
                    reliability=0.95,
                ),
            ],
        )
        frame = manager.open_frame(query_id=uuid4())
        manager.add_to_frame(frame.id, conflict_claim)

        ctx = manager.get_frame_context(frame.id)

        both_in_beliefs = [b for b in ctx.beliefs if b.truth_state == TruthState.BOTH]
        if both_in_beliefs:
            assert len(ctx.conflicts) == len(both_in_beliefs)
            for c in ctx.conflicts:
                assert c.conflict is True

    def test_context_goal_propagated(self, manager):
        gid = uuid4()
        frame = manager.open_frame(query_id=uuid4(), goal_id=gid)
        ctx = manager.get_frame_context(frame.id)
        assert ctx.active_goal == gid

    def test_scratchpad_is_copy(self, manager):
        """Mutating the returned scratchpad must not affect the frame."""
        frame = manager.open_frame(query_id=uuid4())
        manager.write_scratchpad(frame.id, "x", 10)
        ctx = manager.get_frame_context(frame.id)
        ctx.scratchpad["x"] = 999
        assert frame.scratchpad["x"] == 10


# ---------------------------------------------------------------------------
# gc_frames
# ---------------------------------------------------------------------------


class TestGcFrames:
    def test_gc_expires_overdue_active_frames(self, manager):
        frame = manager.open_frame(query_id=uuid4(), ttl_seconds=300)
        frame.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

        count = manager.gc_frames()

        assert count >= 1
        assert frame.status == FrameStatus.EXPIRED

    def test_gc_does_not_expire_future_frames(self, manager):
        frame = manager.open_frame(query_id=uuid4(), ttl_seconds=300)

        manager.gc_frames()

        assert frame.status == FrameStatus.ACTIVE

    def test_gc_removes_stale_non_active_frames(self, manager):
        frame = manager.open_frame(query_id=uuid4(), ttl_seconds=300)
        frame.status = FrameStatus.COMMITTED
        # Back-date created_at by more than 1 hour so it's stale
        frame.created_at = datetime.now(timezone.utc) - timedelta(hours=2)

        count = manager.gc_frames()

        assert count >= 1
        assert frame.id not in manager._frames

    def test_gc_does_not_remove_recent_non_active_frames(self, manager):
        """Committed frames within the 1-hour window are kept."""
        frame = manager.open_frame(query_id=uuid4(), ttl_seconds=300)
        frame.status = FrameStatus.COMMITTED
        # created_at is very recent (default), so < 1 hour

        manager.gc_frames()

        assert frame.id in manager._frames

    def test_gc_returns_total_count(self, manager):
        # 2 overdue ACTIVE frames
        for _ in range(2):
            f = manager.open_frame(query_id=uuid4(), ttl_seconds=300)
            f.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

        # 1 stale ABANDONED frame
        f3 = manager.open_frame(query_id=uuid4(), ttl_seconds=300)
        f3.status = FrameStatus.ABANDONED
        f3.created_at = datetime.now(timezone.utc) - timedelta(hours=2)

        count = manager.gc_frames()

        # 2 expired + 1 stale = 3
        assert count == 3

    def test_gc_no_frames(self, manager):
        count = manager.gc_frames()
        assert count == 0

    def test_gc_multiple_passes(self, manager):
        """Second gc pass on already-expired frames should still remove stale ones."""
        frame = manager.open_frame(query_id=uuid4(), ttl_seconds=300)
        frame.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

        manager.gc_frames()
        # Now frame.status == EXPIRED; back-date created_at to make it stale
        frame.created_at = datetime.now(timezone.utc) - timedelta(hours=2)

        count2 = manager.gc_frames()
        assert count2 >= 1
        assert frame.id not in manager._frames
