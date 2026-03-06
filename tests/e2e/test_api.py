"""End-to-end tests for the REST API."""

import os
import shutil
import tempfile

import pytest
from httpx import ASGITransport, AsyncClient

from mnemebrain_core.api.app import create_app

try:
    import sentence_transformers  # noqa: F401

    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not HAS_EMBEDDINGS, reason="sentence-transformers not installed"),
]


@pytest.fixture
def app():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_db")
    application = create_app(db_path=db_path)
    yield application
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
