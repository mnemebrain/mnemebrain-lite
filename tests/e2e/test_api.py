"""End-to-end tests for the REST API."""

import os
import shutil
import tempfile

import pytest
from httpx import ASGITransport, AsyncClient

from mnemebrain_core.api.app import create_app

import importlib.util

HAS_EMBEDDINGS = importlib.util.find_spec("sentence_transformers") is not None

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not HAS_EMBEDDINGS, reason="sentence-transformers not installed"
    ),
]


@pytest.fixture
def app():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_db")
    application = create_app(db_path=db_path)
    # ASGITransport does not trigger lifespan events,
    # so we manually initialize app.state for testing.
    from mnemebrain_core.memory import BeliefMemory
    from mnemebrain_core.working_memory import WorkingMemoryManager

    memory = BeliefMemory(db_path=db_path, max_db_size=1 << 30)
    wm_manager = WorkingMemoryManager(memory)
    application.state.memory = memory
    application.state.wm_manager = wm_manager
    yield application
    memory.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.e2e
class TestHealthEndpoint:
    async def test_health(self, client: AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


@pytest.mark.e2e
class TestBelieveEndpoint:
    async def test_believe_creates_belief(self, client: AsyncClient):
        resp = await client.post(
            "/believe",
            json={
                "claim": "user is vegetarian",
                "evidence": [
                    {
                        "source_ref": "msg_12",
                        "content": "They said no meat please",
                        "polarity": "supports",
                        "weight": 0.8,
                        "reliability": 0.9,
                    }
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["truth_state"] == "true"
        assert data["confidence"] > 0.5
        assert data["conflict"] is False
        assert "id" in data

    async def test_believe_with_all_fields(self, client: AsyncClient):
        resp = await client.post(
            "/believe",
            json={
                "claim": "user's name is Alice",
                "evidence": [
                    {
                        "source_ref": "msg_1",
                        "content": "introduced as Alice",
                        "polarity": "supports",
                        "weight": 0.9,
                        "reliability": 0.95,
                    }
                ],
                "belief_type": "fact",
                "tags": ["identity"],
                "source_agent": "onboarding",
            },
        )
        assert resp.status_code == 200


@pytest.mark.e2e
class TestExplainEndpoint:
    async def test_explain_existing(self, client: AsyncClient):
        # First create a belief
        await client.post(
            "/believe",
            json={
                "claim": "user likes spicy food",
                "evidence": [
                    {
                        "source_ref": "msg_10",
                        "content": "ordered extra hot sauce",
                        "polarity": "supports",
                        "weight": 0.8,
                        "reliability": 0.9,
                    }
                ],
            },
        )
        # Then explain it
        resp = await client.get("/explain", params={"claim": "user likes spicy food"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["truth_state"] == "true"
        assert len(data["supporting"]) == 1
        assert len(data["attacking"]) == 0

    async def test_explain_not_found(self, client: AsyncClient):
        resp = await client.get("/explain", params={"claim": "unknown belief"})
        assert resp.status_code == 404


@pytest.mark.e2e
class TestReviseEndpoint:
    async def test_revise_adds_evidence(self, client: AsyncClient):
        # Create belief
        resp = await client.post(
            "/believe",
            json={
                "claim": "user is vegetarian",
                "evidence": [
                    {
                        "source_ref": "msg_1",
                        "content": "no meat",
                        "polarity": "supports",
                        "weight": 0.8,
                        "reliability": 0.9,
                    }
                ],
            },
        )
        belief_id = resp.json()["id"]

        # Revise with new evidence
        resp = await client.post(
            "/revise",
            json={
                "belief_id": belief_id,
                "evidence": {
                    "source_ref": "msg_50",
                    "content": "confirmed vegetarian",
                    "polarity": "supports",
                    "weight": 0.9,
                    "reliability": 0.95,
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == belief_id


@pytest.mark.e2e
class TestRetractEndpoint:
    async def test_retract_evidence(self, client: AsyncClient):
        # Create belief
        await client.post(
            "/believe",
            json={
                "claim": "user likes pizza",
                "evidence": [
                    {
                        "source_ref": "msg_1",
                        "content": "ordered pizza",
                        "polarity": "supports",
                        "weight": 0.8,
                        "reliability": 0.9,
                    }
                ],
            },
        )
        # Get evidence ID
        explain_resp = await client.get(
            "/explain", params={"claim": "user likes pizza"}
        )
        evidence_id = explain_resp.json()["supporting"][0]["id"]

        # Retract
        resp = await client.post("/retract", json={"evidence_id": evidence_id})
        assert resp.status_code == 200


@pytest.mark.e2e
class TestFullWorkflow:
    async def test_believe_revise_explain_retract(self, client: AsyncClient):
        """Full lifecycle: believe → revise → explain → retract."""
        # 1. Create belief
        r1 = await client.post(
            "/believe",
            json={
                "claim": "user prefers morning meetings",
                "evidence": [
                    {
                        "source_ref": "cal_1",
                        "content": "scheduled 3 morning meetings",
                        "polarity": "supports",
                        "weight": 0.7,
                        "reliability": 0.8,
                    }
                ],
                "belief_type": "preference",
            },
        )
        assert r1.status_code == 200
        belief_id = r1.json()["id"]

        # 2. Revise with contradicting evidence
        r2 = await client.post(
            "/revise",
            json={
                "belief_id": belief_id,
                "evidence": {
                    "source_ref": "cal_20",
                    "content": "complained about early meetings",
                    "polarity": "attacks",
                    "weight": 0.8,
                    "reliability": 0.9,
                },
            },
        )
        assert r2.status_code == 200
        assert r2.json()["conflict"] is True  # BOTH state

        # 3. Explain
        r3 = await client.get(
            "/explain", params={"claim": "user prefers morning meetings"}
        )
        assert r3.status_code == 200
        data = r3.json()
        assert len(data["supporting"]) == 1
        assert len(data["attacking"]) == 1

        # 4. Retract the attacking evidence
        attacking_id = data["attacking"][0]["id"]
        r4 = await client.post("/retract", json={"evidence_id": attacking_id})
        assert r4.status_code == 200

        # 5. Verify conflict resolved
        r5 = await client.get(
            "/explain", params={"claim": "user prefers morning meetings"}
        )
        assert r5.status_code == 200
        final = r5.json()
        assert len(final["attacking"]) == 0
        assert len(final["expired"]) == 1
        assert final["truth_state"] == "true"


# ---------------------------------------------------------------------------
# New tests — search, list_beliefs, frame endpoints
# ---------------------------------------------------------------------------

_BELIEVE_PAYLOAD = {
    "claim": "user drinks coffee every morning",
    "evidence": [
        {
            "source_ref": "obs_1",
            "content": "seen with coffee cup",
            "polarity": "supports",
            "weight": 0.8,
            "reliability": 0.9,
        }
    ],
    "belief_type": "preference",
    "tags": ["habit"],
}


@pytest.mark.e2e
class TestSearchEndpoint:
    async def test_search_basic(self, client: AsyncClient):
        """Search returns ranked results for a matching query."""
        await client.post("/believe", json=_BELIEVE_PAYLOAD)
        resp = await client.get("/search", params={"query": "coffee morning"})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) >= 1
        result = data["results"][0]
        assert "belief_id" in result
        assert "claim" in result
        assert "truth_state" in result
        assert "confidence" in result
        assert "similarity" in result
        assert "rank_score" in result

    async def test_search_custom_alpha_and_limit(self, client: AsyncClient):
        """Search respects alpha and limit query parameters."""
        await client.post("/believe", json=_BELIEVE_PAYLOAD)
        resp = await client.get(
            "/search",
            params={"query": "coffee", "alpha": 0.5, "limit": 1},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) <= 1

    async def test_search_with_conflict_policy_conservative(self, client: AsyncClient):
        """Search with conflict_policy=conservative is accepted."""
        await client.post("/believe", json=_BELIEVE_PAYLOAD)
        resp = await client.get(
            "/search",
            params={"query": "coffee", "conflict_policy": "conservative"},
        )
        assert resp.status_code == 200

    async def test_search_invalid_conflict_policy(self, client: AsyncClient):
        """Search with an unknown conflict_policy returns 422."""
        resp = await client.get(
            "/search",
            params={"query": "coffee", "conflict_policy": "unknown_policy"},
        )
        assert resp.status_code == 422
        assert "conflict_policy" in resp.json()["detail"]


@pytest.mark.e2e
class TestListBeliefsEndpoint:
    async def _seed(self, client: AsyncClient) -> str:
        """Create a belief and return its id."""
        resp = await client.post("/believe", json=_BELIEVE_PAYLOAD)
        assert resp.status_code == 200
        return resp.json()["id"]

    async def test_list_beliefs_no_filters(self, client: AsyncClient):
        """GET /beliefs returns all beliefs with pagination metadata."""
        await self._seed(client)
        resp = await client.get("/beliefs")
        assert resp.status_code == 200
        data = resp.json()
        assert "beliefs" in data
        assert "total" in data
        assert "offset" in data
        assert "limit" in data
        assert data["total"] >= 1

    async def test_list_beliefs_truth_state_filter(self, client: AsyncClient):
        """Filter by truth_state=true returns only true beliefs."""
        await self._seed(client)
        resp = await client.get("/beliefs", params={"truth_state": "true"})
        assert resp.status_code == 200
        data = resp.json()
        for b in data["beliefs"]:
            assert b["truth_state"] == "true"

    async def test_list_beliefs_belief_type_filter(self, client: AsyncClient):
        """Filter by belief_type=preference returns only preference beliefs."""
        await self._seed(client)
        resp = await client.get("/beliefs", params={"belief_type": "preference"})
        assert resp.status_code == 200
        data = resp.json()
        for b in data["beliefs"]:
            assert b["belief_type"] == "preference"

    async def test_list_beliefs_tag_filter(self, client: AsyncClient):
        """Filter by tag=habit returns only beliefs tagged with 'habit'."""
        await self._seed(client)
        resp = await client.get("/beliefs", params={"tag": "habit"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    async def test_list_beliefs_min_max_confidence(self, client: AsyncClient):
        """Filter by min/max confidence."""
        await self._seed(client)
        resp = await client.get(
            "/beliefs", params={"min_confidence": 0.0, "max_confidence": 1.0}
        )
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    async def test_list_beliefs_limit_and_offset(self, client: AsyncClient):
        """Limit and offset control pagination."""
        await self._seed(client)
        resp = await client.get("/beliefs", params={"limit": 1, "offset": 0})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["beliefs"]) <= 1
        assert data["limit"] == 1
        assert data["offset"] == 0

    async def test_list_beliefs_invalid_truth_state(self, client: AsyncClient):
        """Invalid truth_state query param returns 422."""
        resp = await client.get("/beliefs", params={"truth_state": "maybe"})
        assert resp.status_code == 422
        assert "truth_state" in resp.json()["detail"]

    async def test_list_beliefs_invalid_belief_type(self, client: AsyncClient):
        """Invalid belief_type query param returns 422."""
        resp = await client.get("/beliefs", params={"belief_type": "hunch"})
        assert resp.status_code == 422
        assert "belief_type" in resp.json()["detail"]


@pytest.mark.e2e
class TestFrameEndpoints:
    """Tests for the WorkingMemoryFrame REST endpoints."""

    async def _open_frame(self, client: AsyncClient) -> str:
        """Open a frame and return its frame_id."""
        import uuid

        resp = await client.post(
            "/frame/open",
            json={
                "query_id": str(uuid.uuid4()),
                "top_k": 5,
                "ttl_seconds": 300,
            },
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["frame_id"]

    async def _seed_belief(self, client: AsyncClient) -> str:
        """Create a belief and return its claim."""
        await client.post("/believe", json=_BELIEVE_PAYLOAD)
        return _BELIEVE_PAYLOAD["claim"]

    # --- open ---

    async def test_open_frame_returns_frame_id(self, client: AsyncClient):
        """POST /frame/open returns a valid frame_id."""
        import uuid

        resp = await client.post(
            "/frame/open",
            json={
                "query_id": str(uuid.uuid4()),
                "top_k": 10,
                "ttl_seconds": 60,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "frame_id" in data
        assert "beliefs_loaded" in data
        assert "conflicts" in data
        assert "snapshots" in data
        # Newly opened frame has no preloaded beliefs
        assert data["beliefs_loaded"] == 0

    async def test_open_frame_with_goal_id(self, client: AsyncClient):
        """POST /frame/open with an optional goal_id is accepted."""
        import uuid

        resp = await client.post(
            "/frame/open",
            json={
                "query_id": str(uuid.uuid4()),
                "goal_id": str(uuid.uuid4()),
                "top_k": 5,
                "ttl_seconds": 120,
            },
        )
        assert resp.status_code == 200

    # --- add ---

    async def test_add_belief_to_frame(self, client: AsyncClient):
        """POST /frame/{id}/add loads an existing belief into the frame."""
        claim = await self._seed_belief(client)
        frame_id = await self._open_frame(client)

        resp = await client.post(
            f"/frame/{frame_id}/add",
            json={"claim": claim},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "belief_id" in data
        assert data["claim"] == claim

    async def test_add_to_frame_not_found(self, client: AsyncClient):
        """POST /frame/{id}/add with unknown frame_id returns 404."""
        import uuid

        resp = await client.post(
            f"/frame/{uuid.uuid4()}/add",
            json={"claim": "anything"},
        )
        assert resp.status_code == 404

    async def test_add_to_frame_belief_not_found(self, client: AsyncClient):
        """POST /frame/{id}/add with a claim that doesn't exist returns 404."""
        frame_id = await self._open_frame(client)
        resp = await client.post(
            f"/frame/{frame_id}/add",
            json={"claim": "this belief absolutely does not exist in the store xyz987"},
        )
        assert resp.status_code == 404

    # --- scratchpad ---

    async def test_write_scratchpad(self, client: AsyncClient):
        """POST /frame/{id}/scratchpad stores a key-value pair."""
        frame_id = await self._open_frame(client)
        resp = await client.post(
            f"/frame/{frame_id}/scratchpad",
            json={"key": "interim_answer", "value": "42"},
        )
        assert resp.status_code == 204

    async def test_write_scratchpad_frame_not_found(self, client: AsyncClient):
        """POST /frame/{id}/scratchpad with unknown frame_id returns 404."""
        import uuid

        resp = await client.post(
            f"/frame/{uuid.uuid4()}/scratchpad",
            json={"key": "k", "value": "v"},
        )
        assert resp.status_code == 404

    # --- context ---

    async def test_get_frame_context(self, client: AsyncClient):
        """GET /frame/{id}/context returns the active frame context."""
        frame_id = await self._open_frame(client)
        # Write something to scratchpad so we get a non-trivial context
        await client.post(
            f"/frame/{frame_id}/scratchpad",
            json={"key": "note", "value": "hello"},
        )
        resp = await client.get(f"/frame/{frame_id}/context")
        assert resp.status_code == 200
        data = resp.json()
        assert "active_query" in data
        assert "beliefs" in data
        assert "scratchpad" in data
        assert "conflicts" in data
        assert "step_count" in data
        assert data["scratchpad"]["note"] == "hello"

    async def test_get_frame_context_not_found(self, client: AsyncClient):
        """GET /frame/{id}/context with unknown frame_id returns 404."""
        import uuid

        resp = await client.get(f"/frame/{uuid.uuid4()}/context")
        assert resp.status_code == 404

    # --- commit ---

    async def test_commit_frame_with_new_beliefs(self, client: AsyncClient):
        """POST /frame/{id}/commit creates new beliefs from the payload."""
        frame_id = await self._open_frame(client)
        resp = await client.post(
            f"/frame/{frame_id}/commit",
            json={
                "new_beliefs": [
                    {
                        "claim": "user prefers tea over coffee",
                        "evidence": [
                            {
                                "source_ref": "obs_99",
                                "content": "switched to tea",
                                "polarity": "supports",
                                "weight": 0.7,
                                "reliability": 0.8,
                            }
                        ],
                        "belief_type": "preference",
                        "tags": ["drink"],
                    }
                ],
                "revisions": [],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["beliefs_created"] == 1
        assert data["beliefs_revised"] == 0
        assert "frame_id" in data

    async def test_commit_frame_with_revisions(self, client: AsyncClient):
        """POST /frame/{id}/commit revises existing beliefs."""
        # Create a belief first
        create_resp = await client.post("/believe", json=_BELIEVE_PAYLOAD)
        belief_id = create_resp.json()["id"]

        frame_id = await self._open_frame(client)
        resp = await client.post(
            f"/frame/{frame_id}/commit",
            json={
                "new_beliefs": [],
                "revisions": [
                    {
                        "belief_id": belief_id,
                        "evidence": {
                            "source_ref": "obs_rev_1",
                            "content": "now prefers tea",
                            "polarity": "attacks",
                            "weight": 0.6,
                            "reliability": 0.7,
                        },
                    }
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["beliefs_revised"] == 1

    async def test_commit_frame_empty_payload(self, client: AsyncClient):
        """POST /frame/{id}/commit with empty lists commits successfully."""
        frame_id = await self._open_frame(client)
        resp = await client.post(
            f"/frame/{frame_id}/commit",
            json={"new_beliefs": [], "revisions": []},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["beliefs_created"] == 0
        assert data["beliefs_revised"] == 0

    async def test_commit_frame_not_found(self, client: AsyncClient):
        """POST /frame/{id}/commit with unknown frame_id returns 404."""
        import uuid

        resp = await client.post(
            f"/frame/{uuid.uuid4()}/commit",
            json={"new_beliefs": [], "revisions": []},
        )
        assert resp.status_code == 404

    # --- close ---

    async def test_close_frame(self, client: AsyncClient):
        """DELETE /frame/{id} closes an active frame with 204."""
        frame_id = await self._open_frame(client)
        resp = await client.delete(f"/frame/{frame_id}")
        assert resp.status_code == 204

    async def test_close_frame_not_found(self, client: AsyncClient):
        """DELETE /frame/{id} with unknown frame_id returns 404."""
        import uuid

        resp = await client.delete(f"/frame/{uuid.uuid4()}")
        assert resp.status_code == 404
