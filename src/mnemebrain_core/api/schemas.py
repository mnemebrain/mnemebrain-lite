"""Request/response schemas for the REST API."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from mnemebrain_core.models import BeliefType, Polarity


class EvidenceRequest(BaseModel):
    source_ref: str
    content: str
    polarity: Polarity
    weight: float = Field(ge=0.0, le=1.0, default=0.7)
    reliability: float = Field(ge=0.0, le=1.0, default=0.8)
    scope: str | None = None


class BelieveRequest(BaseModel):
    claim: str = Field(min_length=1)
    evidence: list[EvidenceRequest]
    belief_type: BeliefType = BeliefType.INFERENCE
    tags: list[str] = Field(default_factory=list)
    source_agent: str = ""


class RetractRequest(BaseModel):
    evidence_id: UUID


class ReviseRequest(BaseModel):
    belief_id: UUID
    evidence: EvidenceRequest


class BeliefResponse(BaseModel):
    id: str
    truth_state: str
    confidence: float
    conflict: bool


class EvidenceResponse(BaseModel):
    id: str
    source_ref: str
    content: str
    polarity: str
    weight: float
    reliability: float
    scope: str | None = None


class ExplanationResponse(BaseModel):
    claim: str
    truth_state: str
    confidence: float
    supporting: list[EvidenceResponse]
    attacking: list[EvidenceResponse]
    expired: list[EvidenceResponse]


class SearchResultResponse(BaseModel):
    belief_id: str
    claim: str
    truth_state: str
    confidence: float
    similarity: float
    rank_score: float


class SearchResponse(BaseModel):
    results: list[SearchResultResponse]


class BeliefListItemResponse(BaseModel):
    id: str
    claim: str
    belief_type: str
    truth_state: str
    confidence: float
    tag_count: int
    evidence_count: int
    created_at: str
    last_revised: str


class BeliefListResponse(BaseModel):
    beliefs: list[BeliefListItemResponse]
    total: int
    offset: int
    limit: int


# --- Phase 2: WorkingMemoryFrame schemas ---


class OpenFrameRequest(BaseModel):
    query_id: UUID
    goal_id: UUID | None = None
    top_k: int = Field(default=20, ge=1, le=1000)
    ttl_seconds: int = Field(default=300, ge=10, le=3600)


class BeliefSnapshotResponse(BaseModel):
    belief_id: str
    claim: str
    truth_state: str
    confidence: float
    belief_type: str
    evidence_count: int
    conflict: bool


class OpenFrameResponse(BaseModel):
    frame_id: str
    beliefs_loaded: int
    conflicts: int
    snapshots: list[BeliefSnapshotResponse]


class AddToFrameRequest(BaseModel):
    claim: str


class ScratchpadRequest(BaseModel):
    key: str
    value: Any


class FrameContextResponse(BaseModel):
    active_query: str
    active_goal: str | None
    beliefs: list[BeliefSnapshotResponse]
    scratchpad: dict[str, Any]
    conflicts: list[BeliefSnapshotResponse]
    step_count: int


class NewBeliefPayload(BaseModel):
    claim: str = Field(min_length=1)
    evidence: list[EvidenceRequest] = Field(default_factory=list)
    belief_type: BeliefType = BeliefType.INFERENCE
    tags: list[str] = Field(default_factory=list)


class RevisionPayload(BaseModel):
    belief_id: UUID
    evidence: EvidenceRequest


class CommitFrameRequest(BaseModel):
    new_beliefs: list[NewBeliefPayload] = Field(default_factory=list)
    revisions: list[RevisionPayload] = Field(default_factory=list)


class CommitFrameResponse(BaseModel):
    frame_id: str
    beliefs_created: int
    beliefs_revised: int
