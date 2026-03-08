"""Tests for all route handlers — covers uncovered branches in routes.py."""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import State

from mnemebrain_core.api.routes import get_memory, get_wm_manager
from mnemebrain_core.memory import BeliefMemory, BeliefResult, ExplanationResult
from mnemebrain_core.models import (
    Belief,
    BeliefType,
    Evidence,
    Polarity,
    TruthState,
)
from mnemebrain_core.working_memory import (
    BeliefSnapshot,
    FrameCommitResult,
    FrameContext,
    WorkingMemoryFrame,
    WorkingMemoryManager,
)


class TestGetMemoryUninitialized:
    def test_raises_when_not_initialized(self):
        request = MagicMock()
        request.app.state = State()
        with pytest.raises(RuntimeError, match="not initialized"):
            get_memory(request)


class TestGetWmManagerUninitialized:
    def test_raises_when_not_initialized(self):
        request = MagicMock()
        request.app.state = State()
        with pytest.raises(RuntimeError, match="not initialized"):
            get_wm_manager(request)


# ---------------------------------------------------------------------------
# Helpers to build a TestClient with mocked BeliefMemory/WorkingMemoryManager
# ---------------------------------------------------------------------------


def _build_test_client(mock_memory=None, mock_wm=None):
    """Create a FastAPI TestClient with mocked state."""
    from mnemebrain_core.api.app import create_app

    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_db")
    app = create_app(db_path=db_path)

    if mock_memory is None:
        mock_memory = MagicMock(spec=BeliefMemory)
    if mock_wm is None:
        mock_wm = MagicMock(spec=WorkingMemoryManager)

    app.state.memory = mock_memory
    app.state.wm_manager = mock_wm

    client = TestClient(app, raise_server_exceptions=False)
    client._tmpdir = tmpdir  # for cleanup
    return client, mock_memory, mock_wm, tmpdir


def _cleanup(tmpdir):
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealthRoute:
    def test_health(self):
        client, _, _, tmpdir = _build_test_client()
        try:
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}
        finally:
            _cleanup(tmpdir)


# ---------------------------------------------------------------------------
# POST /believe
# ---------------------------------------------------------------------------


class TestBelieveRoute:
    def test_believe_success(self):
        mock_mem = MagicMock(spec=BeliefMemory)
        bid = uuid4()
        mock_mem.believe.return_value = BeliefResult(
            id=bid, truth_state=TruthState.TRUE, confidence=0.85, conflict=False
        )
        client, _, _, tmpdir = _build_test_client(mock_memory=mock_mem)
        try:
            resp = client.post(
                "/believe",
                json={
                    "claim": "Python is great",
                    "evidence": [
                        {
                            "source_ref": "msg_1",
                            "content": "everyone says so",
                            "polarity": "supports",
                        }
                    ],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["id"] == str(bid)
            assert data["truth_state"] == "true"
            assert data["confidence"] == 0.85
            assert data["conflict"] is False
        finally:
            _cleanup(tmpdir)


# ---------------------------------------------------------------------------
# POST /retract
# ---------------------------------------------------------------------------


class TestRetractRoute:
    def test_retract_success(self):
        mock_mem = MagicMock(spec=BeliefMemory)
        bid = uuid4()
        mock_mem.retract.return_value = [
            BeliefResult(
                id=bid, truth_state=TruthState.NEITHER, confidence=0.5, conflict=False
            )
        ]
        client, _, _, tmpdir = _build_test_client(mock_memory=mock_mem)
        try:
            eid = uuid4()
            resp = client.post("/retract", json={"evidence_id": str(eid)})
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 1
            assert data[0]["truth_state"] == "neither"
        finally:
            _cleanup(tmpdir)

    def test_retract_empty(self):
        mock_mem = MagicMock(spec=BeliefMemory)
        mock_mem.retract.return_value = []
        client, _, _, tmpdir = _build_test_client(mock_memory=mock_mem)
        try:
            resp = client.post("/retract", json={"evidence_id": str(uuid4())})
            assert resp.status_code == 200
            assert resp.json() == []
        finally:
            _cleanup(tmpdir)


# ---------------------------------------------------------------------------
# GET /explain
# ---------------------------------------------------------------------------


class TestExplainRoute:
    def test_explain_found(self):
        mock_mem = MagicMock(spec=BeliefMemory)
        ev_sup = Evidence(
            belief_id=uuid4(),
            source_ref="src_1",
            content="supporting evidence",
            polarity=Polarity.SUPPORTS,
            reliability=0.9,
            weight=0.8,
        )
        ev_att = Evidence(
            belief_id=uuid4(),
            source_ref="src_2",
            content="attacking evidence",
            polarity=Polarity.ATTACKS,
            reliability=0.7,
            weight=0.6,
        )
        mock_mem.explain.return_value = ExplanationResult(
            claim="test claim",
            truth_state=TruthState.TRUE,
            confidence=0.9,
            supporting=[ev_sup],
            attacking=[ev_att],
            expired=[],
        )
        client, _, _, tmpdir = _build_test_client(mock_memory=mock_mem)
        try:
            resp = client.get("/explain", params={"claim": "test claim"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["claim"] == "test claim"
            assert len(data["supporting"]) == 1
            assert len(data["attacking"]) == 1
            assert data["expired"] == []
        finally:
            _cleanup(tmpdir)

    def test_explain_not_found(self):
        mock_mem = MagicMock(spec=BeliefMemory)
        mock_mem.explain.return_value = None
        client, _, _, tmpdir = _build_test_client(mock_memory=mock_mem)
        try:
            resp = client.get("/explain", params={"claim": "nonexistent"})
            assert resp.status_code == 404
        finally:
            _cleanup(tmpdir)


# ---------------------------------------------------------------------------
# GET /search
# ---------------------------------------------------------------------------


class TestSearchRoute:
    def test_search_success(self):
        mock_mem = MagicMock(spec=BeliefMemory)
        b = Belief(claim="test")
        mock_mem.search.return_value = [(b, 0.95, 0.8, 0.865)]
        client, _, _, tmpdir = _build_test_client(mock_memory=mock_mem)
        try:
            resp = client.get("/search", params={"query": "test"})
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["results"]) == 1
            assert data["results"][0]["claim"] == "test"
        finally:
            _cleanup(tmpdir)

    def test_search_invalid_conflict_policy(self):
        mock_mem = MagicMock(spec=BeliefMemory)
        client, _, _, tmpdir = _build_test_client(mock_memory=mock_mem)
        try:
            resp = client.get(
                "/search",
                params={"query": "test", "conflict_policy": "invalid_policy"},
            )
            assert resp.status_code == 422
        finally:
            _cleanup(tmpdir)


# ---------------------------------------------------------------------------
# GET /beliefs
# ---------------------------------------------------------------------------


class TestListBeliefsRoute:
    def test_list_beliefs_no_filters(self):
        mock_mem = MagicMock(spec=BeliefMemory)
        b = Belief(claim="test belief", tags=["tag1"])
        mock_mem.list_beliefs.return_value = ([b], 1)
        client, _, _, tmpdir = _build_test_client(mock_memory=mock_mem)
        try:
            resp = client.get("/beliefs")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 1
            assert data["beliefs"][0]["claim"] == "test belief"
            assert data["beliefs"][0]["tag_count"] == 1
        finally:
            _cleanup(tmpdir)

    def test_list_beliefs_with_truth_state_filter(self):
        mock_mem = MagicMock(spec=BeliefMemory)
        mock_mem.list_beliefs.return_value = ([], 0)
        client, _, _, tmpdir = _build_test_client(mock_memory=mock_mem)
        try:
            resp = client.get("/beliefs", params={"truth_state": "true"})
            assert resp.status_code == 200
        finally:
            _cleanup(tmpdir)

    def test_list_beliefs_invalid_truth_state(self):
        mock_mem = MagicMock(spec=BeliefMemory)
        client, _, _, tmpdir = _build_test_client(mock_memory=mock_mem)
        try:
            resp = client.get("/beliefs", params={"truth_state": "INVALID"})
            assert resp.status_code == 422
        finally:
            _cleanup(tmpdir)

    def test_list_beliefs_with_belief_type_filter(self):
        mock_mem = MagicMock(spec=BeliefMemory)
        mock_mem.list_beliefs.return_value = ([], 0)
        client, _, _, tmpdir = _build_test_client(mock_memory=mock_mem)
        try:
            resp = client.get("/beliefs", params={"belief_type": "fact"})
            assert resp.status_code == 200
        finally:
            _cleanup(tmpdir)

    def test_list_beliefs_invalid_belief_type(self):
        mock_mem = MagicMock(spec=BeliefMemory)
        client, _, _, tmpdir = _build_test_client(mock_memory=mock_mem)
        try:
            resp = client.get("/beliefs", params={"belief_type": "INVALID"})
            assert resp.status_code == 422
        finally:
            _cleanup(tmpdir)


# ---------------------------------------------------------------------------
# POST /revise
# ---------------------------------------------------------------------------


class TestReviseRoute:
    def test_revise_success(self):
        mock_mem = MagicMock(spec=BeliefMemory)
        bid = uuid4()
        mock_mem.revise.return_value = BeliefResult(
            id=bid, truth_state=TruthState.TRUE, confidence=0.9, conflict=False
        )
        client, _, _, tmpdir = _build_test_client(mock_memory=mock_mem)
        try:
            resp = client.post(
                "/revise",
                json={
                    "belief_id": str(bid),
                    "evidence": {
                        "source_ref": "src_1",
                        "content": "new evidence",
                        "polarity": "supports",
                    },
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["id"] == str(bid)
            assert data["truth_state"] == "true"
        finally:
            _cleanup(tmpdir)


# ---------------------------------------------------------------------------
# Working memory frame endpoints
# ---------------------------------------------------------------------------


def _make_snapshot(bid=None, claim="test", conflict=False):
    return BeliefSnapshot(
        belief_id=bid or uuid4(),
        claim=claim,
        truth_state=TruthState.TRUE,
        confidence=0.8,
        belief_type=BeliefType.INFERENCE,
        evidence_count=1,
        conflict=conflict,
        loaded_at=datetime.now(timezone.utc),
    )


class TestOpenFrameRoute:
    def test_open_frame(self):
        mock_wm = MagicMock(spec=WorkingMemoryManager)
        frame_id = uuid4()
        bid = uuid4()
        snap = _make_snapshot(bid=bid)
        frame = WorkingMemoryFrame(
            id=frame_id,
            active_beliefs=[bid],
            belief_snapshots={bid: snap},
        )
        mock_wm.open_frame.return_value = frame
        client, _, _, tmpdir = _build_test_client(mock_wm=mock_wm)
        try:
            resp = client.post(
                "/frame/open",
                json={"query_id": str(uuid4())},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["frame_id"] == str(frame_id)
            assert data["beliefs_loaded"] == 1
        finally:
            _cleanup(tmpdir)


class TestAddToFrameRoute:
    def test_add_to_frame_success(self):
        mock_wm = MagicMock(spec=WorkingMemoryManager)
        snap = _make_snapshot()
        mock_wm.add_to_frame.return_value = snap
        client, _, _, tmpdir = _build_test_client(mock_wm=mock_wm)
        try:
            resp = client.post(
                f"/frame/{uuid4()}/add",
                json={"claim": "test claim"},
            )
            assert resp.status_code == 200
        finally:
            _cleanup(tmpdir)

    def test_add_to_frame_not_found(self):
        mock_wm = MagicMock(spec=WorkingMemoryManager)
        mock_wm.add_to_frame.side_effect = ValueError("Frame not found")
        client, _, _, tmpdir = _build_test_client(mock_wm=mock_wm)
        try:
            resp = client.post(
                f"/frame/{uuid4()}/add",
                json={"claim": "test"},
            )
            assert resp.status_code == 404
        finally:
            _cleanup(tmpdir)

    def test_add_to_frame_belief_not_found(self):
        mock_wm = MagicMock(spec=WorkingMemoryManager)
        mock_wm.add_to_frame.return_value = None
        client, _, _, tmpdir = _build_test_client(mock_wm=mock_wm)
        try:
            resp = client.post(
                f"/frame/{uuid4()}/add",
                json={"claim": "nonexistent"},
            )
            assert resp.status_code == 404
            assert "Belief not found" in resp.json()["detail"]
        finally:
            _cleanup(tmpdir)


class TestScratchpadRoute:
    def test_write_scratchpad(self):
        mock_wm = MagicMock(spec=WorkingMemoryManager)
        client, _, _, tmpdir = _build_test_client(mock_wm=mock_wm)
        try:
            resp = client.post(
                f"/frame/{uuid4()}/scratchpad",
                json={"key": "step1", "value": "result"},
            )
            assert resp.status_code == 204
        finally:
            _cleanup(tmpdir)

    def test_write_scratchpad_not_found(self):
        mock_wm = MagicMock(spec=WorkingMemoryManager)
        mock_wm.write_scratchpad.side_effect = ValueError("Frame not found")
        client, _, _, tmpdir = _build_test_client(mock_wm=mock_wm)
        try:
            resp = client.post(
                f"/frame/{uuid4()}/scratchpad",
                json={"key": "k", "value": "v"},
            )
            assert resp.status_code == 404
        finally:
            _cleanup(tmpdir)


class TestGetFrameContextRoute:
    def test_get_context(self):
        mock_wm = MagicMock(spec=WorkingMemoryManager)
        snap = _make_snapshot()
        mock_wm.get_frame_context.return_value = FrameContext(
            active_query=uuid4(),
            active_goal=uuid4(),
            beliefs=[snap],
            scratchpad={"key": "val"},
            conflicts=[],
            step_count=2,
        )
        client, _, _, tmpdir = _build_test_client(mock_wm=mock_wm)
        try:
            resp = client.get(f"/frame/{uuid4()}/context")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["beliefs"]) == 1
            assert data["step_count"] == 2
        finally:
            _cleanup(tmpdir)

    def test_get_context_not_found(self):
        mock_wm = MagicMock(spec=WorkingMemoryManager)
        mock_wm.get_frame_context.side_effect = ValueError("Frame not found")
        client, _, _, tmpdir = _build_test_client(mock_wm=mock_wm)
        try:
            resp = client.get(f"/frame/{uuid4()}/context")
            assert resp.status_code == 404
        finally:
            _cleanup(tmpdir)


class TestCommitFrameRoute:
    def test_commit_frame(self):
        mock_wm = MagicMock(spec=WorkingMemoryManager)
        fid = uuid4()
        mock_wm.commit_frame.return_value = FrameCommitResult(
            frame_id=fid, beliefs_created=1, beliefs_revised=0
        )
        client, _, _, tmpdir = _build_test_client(mock_wm=mock_wm)
        try:
            resp = client.post(
                f"/frame/{fid}/commit",
                json={},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["beliefs_created"] == 1
        finally:
            _cleanup(tmpdir)

    def test_commit_frame_not_found(self):
        mock_wm = MagicMock(spec=WorkingMemoryManager)
        mock_wm.commit_frame.side_effect = ValueError("Frame not found")
        client, _, _, tmpdir = _build_test_client(mock_wm=mock_wm)
        try:
            resp = client.post(f"/frame/{uuid4()}/commit", json={})
            assert resp.status_code == 404
        finally:
            _cleanup(tmpdir)


class TestCloseFrameRoute:
    def test_close_frame(self):
        mock_wm = MagicMock(spec=WorkingMemoryManager)
        client, _, _, tmpdir = _build_test_client(mock_wm=mock_wm)
        try:
            resp = client.delete(f"/frame/{uuid4()}")
            assert resp.status_code == 204
        finally:
            _cleanup(tmpdir)

    def test_close_frame_not_found(self):
        mock_wm = MagicMock(spec=WorkingMemoryManager)
        mock_wm.close_frame.side_effect = ValueError("Frame not found")
        client, _, _, tmpdir = _build_test_client(mock_wm=mock_wm)
        try:
            resp = client.delete(f"/frame/{uuid4()}")
            assert resp.status_code == 404
        finally:
            _cleanup(tmpdir)
