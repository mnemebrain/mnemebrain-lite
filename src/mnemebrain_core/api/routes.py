"""FastAPI route handlers."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException

from mnemebrain_core.api.schemas import (
    AddToFrameRequest,
    BeliefListItemResponse,
    BeliefListResponse,
    BeliefResponse,
    BeliefSnapshotResponse,
    BelieveRequest,
    CommitFrameRequest,
    CommitFrameResponse,
    EvidenceResponse,
    ExplanationResponse,
    FrameContextResponse,
    OpenFrameRequest,
    OpenFrameResponse,
    RetractRequest,
    ReviseRequest,
    ScratchpadRequest,
    SearchResponse,
    SearchResultResponse,
)
from mnemebrain_core.memory import BeliefMemory
from mnemebrain_core.models import BeliefType, ConflictPolicy, TruthState
from mnemebrain_core.providers.base import EvidenceInput
from mnemebrain_core.working_memory import WorkingMemoryManager

router = APIRouter()

# Set by app.py at startup
_memory: BeliefMemory | None = None
_wm_manager: WorkingMemoryManager | None = None


def set_memory(memory: BeliefMemory) -> None:
    global _memory, _wm_manager
    _memory = memory
    _wm_manager = WorkingMemoryManager(memory)


def get_memory() -> BeliefMemory:
    if _memory is None:
        raise RuntimeError("BeliefMemory not initialized")
    return _memory


def get_wm_manager() -> WorkingMemoryManager:
    if _wm_manager is None:
        raise RuntimeError("WorkingMemoryManager not initialized")
    return _wm_manager


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/believe", response_model=BeliefResponse)
async def believe(req: BelieveRequest):
    mem = get_memory()
    result = mem.believe(
        claim=req.claim,
        evidence_items=[
            EvidenceInput(
                source_ref=e.source_ref,
                content=e.content,
                polarity=e.polarity,
                weight=e.weight,
                reliability=e.reliability,
                scope=e.scope,
            )
            for e in req.evidence
        ],
        belief_type=BeliefType(req.belief_type),
        tags=req.tags,
        source_agent=req.source_agent,
    )
    return BeliefResponse(
        id=str(result.id),
        truth_state=result.truth_state.value,
        confidence=result.confidence,
        conflict=result.conflict,
    )


@router.post("/retract", response_model=list[BeliefResponse])
async def retract(req: RetractRequest):
    mem = get_memory()
    results = mem.retract(UUID(req.evidence_id))
    return [
        BeliefResponse(
            id=str(r.id),
            truth_state=r.truth_state.value,
            confidence=r.confidence,
            conflict=r.conflict,
        )
        for r in results
    ]


@router.get("/explain", response_model=ExplanationResponse)
async def explain(claim: str):
    mem = get_memory()
    result = mem.explain(claim)
    if result is None:
        raise HTTPException(status_code=404, detail="Belief not found")
    return ExplanationResponse(
        claim=result.claim,
        truth_state=result.truth_state.value,
        confidence=result.confidence,
        supporting=[
            EvidenceResponse(
                id=str(e.id),
                source_ref=e.source_ref,
                content=e.content,
                polarity=e.polarity.value,
                weight=e.weight,
                reliability=e.reliability,
                scope=e.scope,
            )
            for e in result.supporting
        ],
        attacking=[
            EvidenceResponse(
                id=str(e.id),
                source_ref=e.source_ref,
                content=e.content,
                polarity=e.polarity.value,
                weight=e.weight,
                reliability=e.reliability,
                scope=e.scope,
            )
            for e in result.attacking
        ],
        expired=[
            EvidenceResponse(
                id=str(e.id),
                source_ref=e.source_ref,
                content=e.content,
                polarity=e.polarity.value,
                weight=e.weight,
                reliability=e.reliability,
                scope=e.scope,
            )
            for e in result.expired
        ],
    )


@router.get("/search", response_model=SearchResponse)
async def search(
    query: str,
    limit: int = 10,
    alpha: float = 0.7,
    conflict_policy: str = "surface",
):
    mem = get_memory()
    try:
        policy = ConflictPolicy(conflict_policy)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid conflict_policy: {conflict_policy}",
        ) from exc
    results = mem.search(query=query, limit=limit, rank_alpha=alpha, conflict_policy=policy)
    return SearchResponse(
        results=[
            SearchResultResponse(
                belief_id=str(b.id),
                claim=b.claim,
                truth_state=b.truth_state.value,
                confidence=conf,
                similarity=sim,
                rank_score=rs,
            )
            for b, sim, conf, rs in results
        ]
    )


@router.get("/beliefs", response_model=BeliefListResponse)
async def list_beliefs(
    truth_state: str | None = None,
    belief_type: str | None = None,
    tag: str | None = None,
    min_confidence: float = 0.0,
    max_confidence: float = 1.0,
    limit: int = 50,
    offset: int = 0,
):
    mem = get_memory()

    try:
        truth_states = (
            [TruthState(s) for s in truth_state.split(",")]
            if truth_state else None
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"Invalid truth_state: {truth_state}",
        ) from exc
    try:
        belief_types = (
            [BeliefType(s) for s in belief_type.split(",")]
            if belief_type else None
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"Invalid belief_type: {belief_type}",
        ) from exc

    beliefs, total = mem.list_beliefs(
        truth_states=truth_states,
        belief_types=belief_types,
        tag=tag,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        limit=limit,
        offset=offset,
    )

    return BeliefListResponse(
        beliefs=[
            BeliefListItemResponse(
                id=str(b.id),
                claim=b.claim,
                belief_type=b.belief_type.value,
                truth_state=b.truth_state.value,
                confidence=b.confidence,
                tag_count=len(b.tags),
                evidence_count=len([e for e in b.evidence if e.valid]),
                created_at=b.created_at.isoformat(),
                last_revised=b.last_revised.isoformat(),
            )
            for b in beliefs
        ],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.post("/revise", response_model=BeliefResponse)
async def revise(req: ReviseRequest):
    mem = get_memory()
    result = mem.revise(
        belief_id=UUID(req.belief_id),
        new_evidence=EvidenceInput(
            source_ref=req.evidence.source_ref,
            content=req.evidence.content,
            polarity=req.evidence.polarity,
            weight=req.evidence.weight,
            reliability=req.evidence.reliability,
            scope=req.evidence.scope,
        ),
    )
    return BeliefResponse(
        id=str(result.id),
        truth_state=result.truth_state.value,
        confidence=result.confidence,
        conflict=result.conflict,
    )


# --- Phase 2: WorkingMemoryFrame endpoints ---


def _snapshot_to_response(snap) -> BeliefSnapshotResponse:
    return BeliefSnapshotResponse(
        belief_id=str(snap.belief_id),
        claim=snap.claim,
        truth_state=snap.truth_state.value,
        confidence=snap.confidence,
        belief_type=snap.belief_type.value,
        evidence_count=snap.evidence_count,
        conflict=snap.conflict,
    )


@router.post("/frame/open", response_model=OpenFrameResponse)
async def open_frame(req: OpenFrameRequest):
    wm = get_wm_manager()
    frame = wm.open_frame(
        query_id=UUID(req.query_id),
        goal_id=UUID(req.goal_id) if req.goal_id else None,
        top_k=req.top_k,
        ttl_seconds=req.ttl_seconds,
    )
    snapshots = [
        frame.belief_snapshots[bid]
        for bid in frame.active_beliefs
        if bid in frame.belief_snapshots
    ]
    conflicts = [s for s in snapshots if s.conflict]
    return OpenFrameResponse(
        frame_id=str(frame.id),
        beliefs_loaded=len(snapshots),
        conflicts=len(conflicts),
        snapshots=[_snapshot_to_response(s) for s in snapshots],
    )


@router.post("/frame/{frame_id}/add", response_model=BeliefSnapshotResponse)
async def add_to_frame(frame_id: str, req: AddToFrameRequest):
    wm = get_wm_manager()
    try:
        snapshot = wm.add_to_frame(UUID(frame_id), req.claim)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Belief not found for claim")
    return _snapshot_to_response(snapshot)


@router.post("/frame/{frame_id}/scratchpad", status_code=204)
async def write_scratchpad(frame_id: str, req: ScratchpadRequest):
    wm = get_wm_manager()
    try:
        wm.write_scratchpad(UUID(frame_id), req.key, req.value)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/frame/{frame_id}/context", response_model=FrameContextResponse)
async def get_frame_context(frame_id: str):
    wm = get_wm_manager()
    try:
        ctx = wm.get_frame_context(UUID(frame_id))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return FrameContextResponse(
        active_query=str(ctx.active_query),
        active_goal=str(ctx.active_goal) if ctx.active_goal else None,
        beliefs=[_snapshot_to_response(s) for s in ctx.beliefs],
        scratchpad=ctx.scratchpad,
        conflicts=[_snapshot_to_response(s) for s in ctx.conflicts],
        step_count=ctx.step_count,
    )


@router.post("/frame/{frame_id}/commit", response_model=CommitFrameResponse)
async def commit_frame(frame_id: str, req: CommitFrameRequest):
    wm = get_wm_manager()
    try:
        result = wm.commit_frame(
            UUID(frame_id),
            new_beliefs=req.new_beliefs or None,
            revisions=req.revisions or None,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return CommitFrameResponse(
        frame_id=str(result.frame_id),
        beliefs_created=result.beliefs_created,
        beliefs_revised=result.beliefs_revised,
    )


@router.delete("/frame/{frame_id}", status_code=204)
async def close_frame(frame_id: str):
    wm = get_wm_manager()
    try:
        wm.close_frame(UUID(frame_id))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
