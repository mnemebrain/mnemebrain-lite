"""WorkingMemoryFrame — active context buffer for multi-step reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from mnemebrain_core.models import BeliefType, TruthState

if TYPE_CHECKING:
    from mnemebrain_core.memory import BeliefMemory


class FrameStatus(str, Enum):
    ACTIVE = "active"
    COMMITTED = "committed"
    EXPIRED = "expired"
    ABANDONED = "abandoned"


@dataclass
class BeliefSnapshot:
    """Point-in-time snapshot of a belief loaded into working memory."""

    belief_id: UUID
    claim: str
    truth_state: TruthState
    confidence: float
    belief_type: BeliefType
    evidence_count: int
    conflict: bool
    loaded_at: datetime


@dataclass
class FrameContext:
    """Active context of a frame for LLM prompt injection."""

    active_query: UUID
    active_goal: UUID | None
    beliefs: list[BeliefSnapshot]
    scratchpad: dict[str, Any]
    conflicts: list[BeliefSnapshot]
    step_count: int


@dataclass
class FrameCommitResult:
    """Result of committing a frame back to the belief graph."""

    frame_id: UUID
    beliefs_created: int
    beliefs_revised: int


@dataclass
class WorkingMemoryFrame:
    id: UUID = field(default_factory=uuid4)
    active_query: UUID = field(default_factory=uuid4)
    active_goal: UUID | None = None
    active_beliefs: list[UUID] = field(default_factory=list)
    active_evidence: list[UUID] = field(default_factory=list)
    belief_snapshots: dict[UUID, BeliefSnapshot] = field(default_factory=dict)
    scratchpad: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    status: FrameStatus = FrameStatus.ACTIVE
    source_agent: str = ""
    step_count: int = 0


class WorkingMemoryManager:
    """Manages working memory frames for multi-step reasoning episodes."""

    def __init__(self, memory: BeliefMemory) -> None:
        self._memory = memory
        self._frames: dict[UUID, WorkingMemoryFrame] = {}

    def _snapshot_belief(self, belief) -> BeliefSnapshot:
        """Create a point-in-time snapshot of a belief."""
        return BeliefSnapshot(
            belief_id=belief.id,
            claim=belief.claim,
            truth_state=belief.truth_state,
            confidence=belief.confidence,
            belief_type=belief.belief_type,
            evidence_count=len([e for e in belief.evidence if e.valid]),
            conflict=belief.truth_state == TruthState.BOTH,
            loaded_at=datetime.now(timezone.utc),
        )

    def open_frame(
        self,
        query_id: UUID,
        goal_id: UUID | None = None,
        top_k: int = 20,
        ttl_seconds: int = 300,
    ) -> WorkingMemoryFrame:
        """Open a new working memory frame for multi-step reasoning.

        The frame is a pure context cache. Beliefs are loaded into
        ``active_beliefs`` via :meth:`add_to_frame` after opening.
        """
        frame = WorkingMemoryFrame(
            active_query=query_id,
            active_goal=goal_id,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
        )

        self._frames[frame.id] = frame
        return frame

    def get_frame(self, frame_id: UUID) -> WorkingMemoryFrame | None:
        """Get a frame by ID. Returns None if not found or expired."""
        frame = self._frames.get(frame_id)
        if frame is None:
            return None
        # Lazy expiry check
        if (
            frame.expires_at
            and datetime.now(timezone.utc) > frame.expires_at
            and frame.status == FrameStatus.ACTIVE
        ):
            frame.status = FrameStatus.EXPIRED
        return frame

    def add_to_frame(
        self,
        frame_id: UUID,
        claim: str,
    ) -> BeliefSnapshot | None:
        """Load an additional belief into the active frame."""
        frame = self._frames.get(frame_id)
        if frame is None:
            raise ValueError(f"Frame {frame_id} not found")
        if frame.status != FrameStatus.ACTIVE:
            raise ValueError(f"Frame {frame_id} is {frame.status.value}, not active")

        result = self._memory.explain(claim)
        if result is None:
            return None

        # Find the belief to get its full object
        beliefs, _ = self._memory.list_beliefs(limit=1000)
        for b in beliefs:
            if b.claim == result.claim:
                # Return existing snapshot if already in frame
                if b.id in frame.belief_snapshots:
                    return frame.belief_snapshots[b.id]
                frame.active_beliefs.append(b.id)
                snapshot = self._snapshot_belief(b)
                frame.belief_snapshots[b.id] = snapshot
                return snapshot
        return None

    def write_scratchpad(
        self,
        frame_id: UUID,
        key: str,
        value: Any,
    ) -> None:
        """Store an intermediate result in the frame's scratchpad."""
        frame = self._frames.get(frame_id)
        if frame is None:
            raise ValueError(f"Frame {frame_id} not found")
        if frame.status != FrameStatus.ACTIVE:
            raise ValueError(f"Frame {frame_id} is {frame.status.value}, not active")

        frame.scratchpad[key] = value
        frame.step_count += 1

    @staticmethod
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        """Attribute access for objects, key access for dicts."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _build_evidence_input(self, raw: Any) -> Any:
        """Build an EvidenceInput from a raw dict or schema object."""
        from mnemebrain_core.providers.base import EvidenceInput

        raw_pol = self._get(raw, "polarity", "supports")
        return EvidenceInput(
            source_ref=self._get(raw, "source_ref", ""),
            content=self._get(raw, "content", ""),
            polarity=raw_pol.value if hasattr(raw_pol, "value") else raw_pol,
            weight=self._get(raw, "weight", 0.8),
            reliability=self._get(raw, "reliability", 0.7),
            scope=self._get(raw, "scope", None),
        )

    def _apply_new_beliefs(self, payloads: list[Any], source_agent: str | None) -> int:
        """Process new belief payloads and return count created."""
        count = 0
        for payload in payloads:
            raw_evidence = self._get(payload, "evidence", [])
            evidence_items = [self._build_evidence_input(e) for e in raw_evidence]
            raw_bt = self._get(payload, "belief_type", "inference")
            self._memory.believe(
                claim=self._get(payload, "claim"),
                evidence_items=evidence_items,
                belief_type=raw_bt if hasattr(raw_bt, "value") else BeliefType(raw_bt),
                tags=self._get(payload, "tags", []),
                source_agent=source_agent or "",
            )
            count += 1
        return count

    def _apply_revisions(self, revisions: list[Any]) -> int:
        """Process revision payloads and return count revised."""
        count = 0
        for rev in revisions:
            raw_ev = self._get(rev, "evidence", {})
            ev = self._build_evidence_input(raw_ev)
            bid = self._get(rev, "belief_id")
            if isinstance(bid, str):
                bid = UUID(bid)
            self._memory.revise(bid, ev)
            count += 1
        return count

    def commit_frame(
        self,
        frame_id: UUID,
        new_beliefs: list[Any] | None = None,
        revisions: list[Any] | None = None,
    ) -> FrameCommitResult:
        """Commit frame results back to the belief graph.

        Accepts typed NewBeliefPayload/RevisionPayload objects from the API
        schemas, or raw dicts for backwards compatibility.
        """
        frame = self._frames.get(frame_id)
        if frame is None:
            raise ValueError(f"Frame {frame_id} not found")
        if frame.status != FrameStatus.ACTIVE:
            raise ValueError(f"Frame {frame_id} is {frame.status.value}, not active")

        beliefs_created = self._apply_new_beliefs(new_beliefs or [], frame.source_agent)
        beliefs_revised = self._apply_revisions(revisions or [])

        frame.status = FrameStatus.COMMITTED
        return FrameCommitResult(
            frame_id=frame.id,
            beliefs_created=beliefs_created,
            beliefs_revised=beliefs_revised,
        )

    def close_frame(self, frame_id: UUID, reason: str = "") -> None:
        """Close a frame without committing."""
        frame = self._frames.get(frame_id)
        if frame is None:
            raise ValueError(f"Frame {frame_id} not found")
        frame.status = FrameStatus.ABANDONED

    def get_frame_context(self, frame_id: UUID) -> FrameContext:
        """Return the full active context of a frame."""
        frame = self._frames.get(frame_id)
        if frame is None:
            raise ValueError(f"Frame {frame_id} not found")

        beliefs = [
            frame.belief_snapshots[bid]
            for bid in frame.active_beliefs
            if bid in frame.belief_snapshots
        ]
        conflicts = [s for s in beliefs if s.conflict]

        return FrameContext(
            active_query=frame.active_query,
            active_goal=frame.active_goal,
            beliefs=beliefs,
            scratchpad=dict(frame.scratchpad),
            conflicts=conflicts,
            step_count=frame.step_count,
        )

    def gc_frames(self) -> int:
        """Garbage collect expired frames."""
        now = datetime.now(timezone.utc)
        expired_ids = [
            fid
            for fid, frame in self._frames.items()
            if frame.expires_at
            and now > frame.expires_at
            and frame.status == FrameStatus.ACTIVE
        ]
        for fid in expired_ids:
            self._frames[fid].status = FrameStatus.EXPIRED

        stale_ids = [
            fid
            for fid, frame in self._frames.items()
            if frame.status != FrameStatus.ACTIVE
            and (now - frame.created_at).total_seconds() > 3600
        ]
        for fid in stale_ids:
            del self._frames[fid]

        return len(expired_ids) + len(stale_ids)
