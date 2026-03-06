"""FastAPI route handlers."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException

from mnemebrain_core.api.schemas import (
    BeliefResponse,
    BelieveRequest,
    EvidenceResponse,
    ExplanationResponse,
    RetractRequest,
    ReviseRequest,
)
from mnemebrain_core.memory import BeliefMemory
from mnemebrain_core.models import BeliefType
from mnemebrain_core.providers.base import EvidenceInput

router = APIRouter()

# Set by app.py at startup
_memory: BeliefMemory | None = None


def set_memory(memory: BeliefMemory) -> None:
    global _memory
    _memory = memory


def get_memory() -> BeliefMemory:
    if _memory is None:
        raise RuntimeError("BeliefMemory not initialized")
    return _memory


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
