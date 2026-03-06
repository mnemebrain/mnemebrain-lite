"""Request/response schemas for the REST API."""

from __future__ import annotations


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
