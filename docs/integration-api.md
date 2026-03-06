# MnemeBrain Lite — Integration API Reference

> **Version:** 0.1.0
> **Base URL:** `http://localhost:8000`
> **Content-Type:** `application/json`

MnemeBrain Lite is a lightweight belief memory system for LLM agents. It provides four core operations — **believe**, **retract**, **explain**, **revise** — backed by Belnap's four-valued logic and an append-only evidence ledger.

---

## Quick Start

```bash
# Start the server
python -m mnemebrain [DB_PATH]
# Default DB_PATH: ./mnemebrain_data
# Listens on 0.0.0.0:8000
```

## Python API

```python
from mnemebrain.memory import BeliefMemory
from mnemebrain.providers.base import EvidenceInput

mem = BeliefMemory(db_path="./my_data")

# Store a belief
result = mem.believe(
    claim="Python 3.12 supports type parameter syntax",
    evidence_items=[
        EvidenceInput(
            source_ref="pep-695",
            content="PEP 695 introduces type parameter syntax",
            polarity="supports",
            weight=0.9,
            reliability=0.95,
        )
    ],
    belief_type="fact",
    tags=["python", "typing"],
    source_agent="research-agent",
)

# Explain a belief
explanation = mem.explain("Python 3.12 supports type parameter syntax")

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

Health check.

**Response:**
```json
{ "status": "ok" }
```

---

### `POST /believe`

Store a belief with one or more evidence items. If a semantically similar belief exists (cosine similarity >= 0.92), evidence is merged into the existing belief.

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
| `weight`      | `float` (0–1)    | No       | `0.7`   | Evidence strength                  |
| `reliability` | `float` (0–1)    | No       | `0.8`   | Source reliability                 |
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

### `POST /retract`

Invalidate a piece of evidence by its ID. Recomputes truth state and confidence for all affected beliefs.

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

### `GET /explain`

Return the full justification chain for a belief. Finds beliefs by semantic similarity (threshold >= 0.8) or exact claim match.

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

**Example:**

```bash
curl "http://localhost:8000/explain?claim=Earth%20orbits%20the%20Sun"
```

---

### `POST /revise`

Add new evidence to an existing belief and recompute its truth state and confidence.

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

Claims are deduplicated via embedding similarity. When a new claim has cosine similarity >= 0.92 with an existing belief, evidence is merged into the existing belief rather than creating a new one.

### Thresholds

| Threshold           | Value | Purpose                                     |
|---------------------|-------|---------------------------------------------|
| Support threshold   | 0.3   | Minimum weighted support for `true`/`both`  |
| Attack threshold    | 0.3   | Minimum weighted attack for `false`/`both`  |
| Dedup similarity    | 0.92  | Merge threshold for believe()               |
| Explain similarity  | 0.80  | Match threshold for explain()               |

## Embedding Provider

MnemeBrain Lite uses **sentence-transformers** with the `all-MiniLM-L6-v2` model by default. No API keys required — runs locally.

## Architecture

```
Client → FastAPI → BeliefMemory → KuzuGraphStore (embedded graph DB)
                        ↓
                  Truth Engine (pure functions)
                        ↓
                  EmbeddingProvider (sentence-transformers)
```

- **Append-only evidence ledger** — evidence is never deleted, only invalidated
- **Kuzu embedded graph DB** — stores beliefs, evidence, and embeddings
- **Stateless computation** — truth states and confidence recomputed from evidence on every mutation
