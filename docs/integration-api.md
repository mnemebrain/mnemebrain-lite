# MnemeBrain Lite — Integration API Reference

> **Version:** 0.1.0a3
> **Base URL:** `http://localhost:8000`
> **Content-Type:** `application/json`

MnemeBrain Lite is a lightweight belief memory system for LLM agents. It provides four core operations — **believe**, **retract**, **explain**, **revise** — backed by Belnap's four-valued logic and an append-only evidence ledger.

> **Lite mode:** On platforms where `sentence-transformers` is unavailable (e.g. Intel Mac — torch 2.3+ has no x86_64 wheels), `believe` and `explain` require a custom `EmbeddingProvider`. Without one, these endpoints return **501 Not Implemented**. `health`, `retract`, and `revise` always work.

---

## Quick Start

```bash
# Full install (Linux / Apple Silicon)
uv sync --extra dev --extra embeddings

# OpenAI embeddings (any platform, requires OPENAI_API_KEY)
uv sync --extra dev --extra openai

# Without embeddings (Intel Mac — torch 2.3+ has no x86_64 wheels)
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Start API server
uv run python -m mnemebrain_core
# Default DB_PATH: ./mnemebrain_data
# Listens on 0.0.0.0:8000
```

## Python API

```python
from mnemebrain_core.memory import BeliefMemory
from mnemebrain_core.providers.base import EvidenceInput

mem = BeliefMemory(db_path="./my_data")

# Revise with new evidence
mem.revise(
    belief_id=result.id,
    new_evidence=EvidenceInput(
        source_ref="docs.python.org",
        content="Confirmed in official docs",
        polarity="supports",
        weight=0.8,
        reliability=1.0,
    ),
)

# Retract evidence
mem.retract(evidence_id=some_evidence_uuid)
```

---

## REST API Endpoints

### `GET /health`

Health check. Always available.

**Response:**
```json
{ "status": "ok" }
```

---

### `POST /retract`

Invalidate a piece of evidence by its ID. Recomputes truth state and confidence for all affected beliefs. Always available.

**Request Body:**

| Field         | Type     | Required | Description       |
|---------------|----------|----------|-------------------|
| `evidence_id` | `string` | Yes      | UUID of evidence  |

**Response:** `BeliefResponse[]` — all affected beliefs with recomputed states.

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "truth_state": "neither",
    "confidence": 0.5,
    "conflict": false
  }
]
```

**Example:**

```bash
curl -X POST http://localhost:8000/retract \
  -H "Content-Type: application/json" \
  -d '{ "evidence_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7" }'
```

---

### `POST /revise`

Add new evidence to an existing belief and recompute its truth state and confidence. Always available.

**Request Body:**

| Field       | Type            | Required | Description              |
|-------------|-----------------|----------|--------------------------|
| `belief_id` | `string`        | Yes      | UUID of belief to revise |
| `evidence`  | `EvidenceInput` | Yes      | New evidence to add      |

**Response:** `BeliefResponse`

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "truth_state": "both",
  "confidence": 0.55,
  "conflict": true
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/revise \
  -H "Content-Type: application/json" \
  -d '{
    "belief_id": "550e8400-e29b-41d4-a716-446655440000",
    "evidence": {
      "source_ref": "new-study",
      "content": "Contradicting evidence found",
      "polarity": "attacks",
      "weight": 0.8,
      "reliability": 0.9
    }
  }'
```

---

### `POST /believe` (requires embeddings)

Store a belief with one or more evidence items. If a semantically similar belief exists (cosine similarity >= 0.92), evidence is merged into the existing belief.

Returns **501 Not Implemented** when no embedding provider is available.

**Request Body:**

| Field          | Type               | Required | Default       | Description                                        |
|----------------|--------------------|----------|---------------|----------------------------------------------------|
| `claim`        | `string`           | Yes      |               | Natural language claim                             |
| `evidence`     | `EvidenceInput[]`  | Yes      |               | One or more evidence items                         |
| `belief_type`  | `string`           | No       | `"inference"` | One of: `fact`, `preference`, `inference`, `prediction` |
| `tags`         | `string[]`         | No       | `[]`          | Arbitrary tags for filtering                       |
| `source_agent` | `string`           | No       | `""`          | Agent that produced this belief                    |

**EvidenceInput:**

| Field         | Type             | Required | Default | Description                        |
|---------------|------------------|----------|---------|------------------------------------|
| `source_ref`  | `string`         | Yes      |         | Source identifier (URL, doc name)  |
| `content`     | `string`         | Yes      |         | Evidence text                      |
| `polarity`    | `string`         | Yes      |         | `"supports"` or `"attacks"`        |
| `weight`      | `float` (0-1)    | No       | `0.7`   | Evidence strength                  |
| `reliability` | `float` (0-1)    | No       | `0.8`   | Source reliability                 |
| `scope`       | `string \| null` | No       | `null`  | Scope qualifier                    |

**Response:** `BeliefResponse`

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "truth_state": "true",
  "confidence": 0.82,
  "conflict": false
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/believe \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "Earth orbits the Sun",
    "evidence": [{
      "source_ref": "astronomy-textbook",
      "content": "Confirmed by centuries of observation",
      "polarity": "supports",
      "weight": 0.95,
      "reliability": 1.0
    }],
    "belief_type": "fact",
    "tags": ["astronomy"]
  }'
```

---

### `GET /explain` (requires embeddings)

Return the full justification chain for a belief. Finds beliefs by semantic similarity (threshold >= 0.8) or exact claim match.

Returns **501 Not Implemented** when no embedding provider is available.

**Query Parameters:**

| Param   | Type     | Required | Description           |
|---------|----------|----------|-----------------------|
| `claim` | `string` | Yes      | Claim to look up      |

**Response:** `ExplanationResponse`

```json
{
  "claim": "Earth orbits the Sun",
  "truth_state": "true",
  "confidence": 0.82,
  "supporting": [
    {
      "id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
      "source_ref": "astronomy-textbook",
      "content": "Confirmed by centuries of observation",
      "polarity": "supports",
      "weight": 0.95,
      "reliability": 1.0,
      "scope": null
    }
  ],
  "attacking": [],
  "expired": []
}
```

**Errors:**
- `404` — No matching belief found.
- `501` — No embedding provider available.

**Example:**

```bash
curl "http://localhost:8000/explain?claim=Earth%20orbits%20the%20Sun"
```

---

### `GET /search` (requires embeddings)

Semantic search over beliefs with ranked scoring. Returns beliefs ordered by a composite rank score combining similarity, confidence, and stability.

**Query Parameters:**

| Param             | Type    | Required | Default     | Description                                      |
|-------------------|---------|----------|-------------|--------------------------------------------------|
| `query`           | `string`| Yes      |             | Natural language search query                    |
| `limit`           | `int`   | No       | `10`        | Maximum results to return                        |
| `alpha`           | `float` | No       | `0.7`       | Similarity weight in rank score (0–1)            |
| `conflict_policy` | `string`| No       | `"surface"` | One of: `surface`, `conservative`, `optimistic`  |

**Conflict policies:**
- `surface` — include all beliefs, including contradictions
- `conservative` — exclude beliefs with `BOTH` truth state
- `optimistic` — for `BOTH` beliefs, treat as `TRUE`

**Response:** `SearchResponse`

```json
{
  "results": [
    {
      "belief_id": "550e8400-e29b-41d4-a716-446655440000",
      "claim": "user is vegetarian",
      "truth_state": "true",
      "confidence": 0.82,
      "similarity": 0.95,
      "rank_score": 0.87
    }
  ]
}
```

**Example:**

```bash
curl "http://localhost:8000/search?query=vegetarian&limit=5&alpha=0.7"
```

---

### `GET /beliefs`

List beliefs with filtering and pagination. Always available.

**Query Parameters:**

| Param            | Type    | Required | Default | Description                                         |
|------------------|---------|----------|---------|-----------------------------------------------------|
| `truth_state`    | `string`| No       |         | Comma-separated: `true`, `false`, `both`, `neither` |
| `belief_type`    | `string`| No       |         | Comma-separated: `fact`, `preference`, `inference`, `prediction` |
| `tag`            | `string`| No       |         | Filter by tag                                       |
| `min_confidence` | `float` | No       | `0.0`   | Minimum confidence threshold                        |
| `max_confidence` | `float` | No       | `1.0`   | Maximum confidence threshold                        |
| `limit`          | `int`   | No       | `50`    | Page size                                           |
| `offset`         | `int`   | No       | `0`     | Pagination offset                                   |

**Response:** `BeliefListResponse`

```json
{
  "beliefs": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "claim": "user is vegetarian",
      "belief_type": "preference",
      "truth_state": "both",
      "confidence": 0.55,
      "tag_count": 1,
      "evidence_count": 2,
      "created_at": "2026-03-08T12:00:00",
      "last_revised": "2026-03-08T14:30:00"
    }
  ],
  "total": 42,
  "offset": 0,
  "limit": 50
}
```

**Example:**

```bash
curl "http://localhost:8000/beliefs?truth_state=BOTH&min_confidence=0.5"
```

---

### `POST /frame/open`

Open a working memory frame — an active context buffer for multi-step reasoning episodes. Frames have a TTL and hold belief snapshots for the duration of a reasoning task.

**Request Body:**

| Field          | Type          | Required | Default | Description                          |
|----------------|---------------|----------|---------|--------------------------------------|
| `query_id`     | `string (UUID)` | Yes    |         | ID of the query/task this frame serves |
| `goal_id`      | `string (UUID)` | No     | `null`  | Optional goal this frame is working toward |
| `top_k`        | `int`         | No       | `20`    | Max beliefs to load (1–1000)         |
| `ttl_seconds`  | `int`         | No       | `300`   | Frame expiry in seconds (10–3600)    |

**Response:** `OpenFrameResponse`

```json
{
  "frame_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "beliefs_loaded": 5,
  "conflicts": 1,
  "snapshots": [
    {
      "belief_id": "550e8400-...",
      "claim": "user is vegetarian",
      "truth_state": "both",
      "confidence": 0.55,
      "belief_type": "preference",
      "evidence_count": 2,
      "conflict": true
    }
  ]
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/frame/open \
  -H "Content-Type: application/json" \
  -d '{"query_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890", "top_k": 20, "ttl_seconds": 300}'
```

---

### `POST /frame/{frame_id}/add`

Add a belief to an open frame by claim text. Returns the belief snapshot if found.

**Request Body:**

| Field   | Type     | Required | Description        |
|---------|----------|----------|--------------------|
| `claim` | `string` | Yes      | Claim to look up   |

**Response:** `BeliefSnapshotResponse`

```json
{
  "belief_id": "550e8400-...",
  "claim": "user is vegetarian",
  "truth_state": "both",
  "confidence": 0.55,
  "belief_type": "preference",
  "evidence_count": 2,
  "conflict": true
}
```

**Errors:**
- `404` — Frame not found or belief not found for claim.

**Example:**

```bash
curl -X POST http://localhost:8000/frame/abc123/add \
  -H "Content-Type: application/json" \
  -d '{"claim": "user is vegetarian"}'
```

---

### `POST /frame/{frame_id}/scratchpad`

Write a key-value pair to the frame's scratchpad. The scratchpad is a free-form dictionary for intermediate reasoning state. Returns `204 No Content`.

**Request Body:**

| Field   | Type  | Required | Description            |
|---------|-------|----------|------------------------|
| `key`   | `string` | Yes   | Scratchpad key         |
| `value` | `any`    | Yes   | Arbitrary JSON value   |

**Errors:**
- `404` — Frame not found.

**Example:**

```bash
curl -X POST http://localhost:8000/frame/abc123/scratchpad \
  -H "Content-Type: application/json" \
  -d '{"key": "reasoning_step", "value": "checking dietary preferences"}'
```

---

### `GET /frame/{frame_id}/context`

Get the full frame context, suitable for injecting into an LLM prompt. Returns all active beliefs, conflicts, scratchpad state, and step count.

**Response:** `FrameContextResponse`

```json
{
  "active_query": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "active_goal": null,
  "beliefs": [
    {
      "belief_id": "550e8400-...",
      "claim": "user is vegetarian",
      "truth_state": "both",
      "confidence": 0.55,
      "belief_type": "preference",
      "evidence_count": 2,
      "conflict": true
    }
  ],
  "scratchpad": {"reasoning_step": "checking dietary preferences"},
  "conflicts": [
    {
      "belief_id": "550e8400-...",
      "claim": "user is vegetarian",
      "truth_state": "both",
      "confidence": 0.55,
      "belief_type": "preference",
      "evidence_count": 2,
      "conflict": true
    }
  ],
  "step_count": 3
}
```

**Errors:**
- `404` — Frame not found.

**Example:**

```bash
curl http://localhost:8000/frame/abc123/context
```

---

### `POST /frame/{frame_id}/commit`

Commit frame results back to the belief graph. Creates new beliefs and/or revises existing ones, then closes the frame.

**Request Body:**

| Field         | Type                 | Required | Default | Description                    |
|---------------|----------------------|----------|---------|--------------------------------|
| `new_beliefs` | `NewBeliefPayload[]` | No       | `[]`    | New beliefs to create          |
| `revisions`   | `RevisionPayload[]`  | No       | `[]`    | Revisions to apply             |

**NewBeliefPayload:**

| Field         | Type              | Required | Default       | Description          |
|---------------|-------------------|----------|---------------|----------------------|
| `claim`       | `string`          | Yes      |               | Claim text           |
| `evidence`    | `EvidenceInput[]` | No       | `[]`          | Supporting evidence  |
| `belief_type` | `string`          | No       | `"inference"` | Belief type          |
| `tags`        | `string[]`        | No       | `[]`          | Tags                 |

**RevisionPayload:**

| Field       | Type            | Required | Description              |
|-------------|-----------------|----------|--------------------------|
| `belief_id` | `string (UUID)` | Yes      | Belief to revise         |
| `evidence`  | `EvidenceInput` | Yes      | New evidence to add      |

**Response:** `CommitFrameResponse`

```json
{
  "frame_id": "abc123",
  "beliefs_created": 2,
  "beliefs_revised": 1
}
```

**Errors:**
- `404` — Frame not found.

**Example:**

```bash
curl -X POST http://localhost:8000/frame/abc123/commit \
  -H "Content-Type: application/json" \
  -d '{
    "new_beliefs": [{"claim": "user prefers plant-based meals", "belief_type": "inference"}],
    "revisions": []
  }'
```

---

### `DELETE /frame/{frame_id}`

Close and discard a frame without committing. Returns `204 No Content`.

**Errors:**
- `404` — Frame not found.

**Example:**

```bash
curl -X DELETE http://localhost:8000/frame/abc123
```

---

## Data Model

### Truth States (Belnap's Four-Valued Logic)

| Value     | Meaning                                  |
|-----------|------------------------------------------|
| `true`    | Sufficient support, insufficient attack  |
| `false`   | Sufficient attack, insufficient support  |
| `both`    | Sufficient support AND attack (conflict) |
| `neither` | Insufficient evidence in either direction|

### Belief Types & Decay Half-Lives

| Type         | Half-Life | Description                     |
|--------------|-----------|---------------------------------|
| `fact`       | 365 days  | Stable knowledge                |
| `preference` | 90 days   | User preferences                |
| `inference`  | 30 days   | Derived conclusions             |
| `prediction` | 3 days    | Time-sensitive predictions      |

### Confidence

Confidence is a **ranking signal** computed via log-odds with sigmoid, not a probability. Returns `0.5` when no evidence exists (maximum uncertainty). Used for ordering beliefs, not as ground truth.

### Deduplication

Claims are deduplicated via embedding similarity. When a new claim has cosine similarity >= 0.92 with an existing belief, evidence is merged into the existing belief rather than creating a new one. Requires embeddings.

### Thresholds

| Threshold           | Value | Purpose                                     |
|---------------------|-------|---------------------------------------------|
| Support threshold   | 0.3   | Minimum weighted support for `true`/`both`  |
| Attack threshold    | 0.3   | Minimum weighted attack for `false`/`both`  |
| Dedup similarity    | 0.92  | Merge threshold for believe()               |
| Explain similarity  | 0.80  | Match threshold for explain()               |

## Embedding Provider

MnemeBrain Lite auto-detects the best available embedding provider:

1. **sentence-transformers** (`all-MiniLM-L6-v2`) — local, no API keys. Install with `pip install mnemebrain-lite[embeddings]`.
2. **OpenAI** (`text-embedding-3-small`) — requires `OPENAI_API_KEY` in environment or `.env` file. Install with `pip install mnemebrain-lite[openai]`.
3. **Custom** — provide your own `EmbeddingProvider` implementation.

```python
from mnemebrain_core.providers.base import EmbeddingProvider

class MyProvider(EmbeddingProvider):
    def embed(self, text: str) -> list[float]:
        # Your embedding logic here
        ...

mem = BeliefMemory(db_path="./data", embedding_provider=MyProvider())
```

Without any embedding provider, `believe` and `explain` return **501 Not Implemented**. All other endpoints work.

## Architecture

See [architecture.md](architecture.md) for the full architecture overview, source layout, and phase roadmap.
