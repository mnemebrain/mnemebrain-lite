"""Request/response schemas for the REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvidenceRequest(BaseModel):
    source_ref: str
    content: str
    polarity: str  # "supports" or "attacks"
    weight: float = Field(ge=0.0, le=1.0, default=0.7)
    reliability: float = Field(ge=0.0, le=1.0, default=0.8)
    scope: str | None = None


class BelieveRequest(BaseModel):
    claim: str
    evidence: list[EvidenceRequest]
    belief_type: str = "inference"
    tags: list[str] = Field(default_factory=list)
    source_agent: str = ""


class RetractRequest(BaseModel):
    evidence_id: str


class ReviseRequest(BaseModel):
    belief_id: str
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
    query_id: str
    goal_id: str | None = None
    top_k: int = 20
    ttl_seconds: int = 300


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


class CommitFrameRequest(BaseModel):
    new_beliefs: list[dict[str, Any]] = Field(default_factory=list)
    revisions: list[dict[str, Any]] = Field(default_factory=list)


class CommitFrameResponse(BaseModel):
    frame_id: str
    beliefs_created: int
    beliefs_revised: int
