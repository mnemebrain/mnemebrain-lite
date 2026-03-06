# MnemeBrain Lite

Give your AI a real brain.

Agents today store text. Mnemebrain stores beliefs — with evidence, confidence, provenance, and revision logic.

Beliefs carry evidence, confidence, provenance, and causal justification. The system can explain, retract, and revise — unlike flat key-value agent memory.

## Core Concepts

**TruthState** — 4-valued logic (Belnap): `TRUE`, `FALSE`, `BOTH` (contradiction), `NEITHER` (insufficient evidence).

**Evidence Ledger** — Append-only. Evidence supports or attacks a belief. Never deleted, only invalidated.

**Time Decay** — Evidence weight decays based on belief type:

| BeliefType | Half-life |
|------------|-----------|
| FACT | 365 days |
| PREFERENCE | 90 days |
| INFERENCE | 30 days |
| PREDICTION | 3 days |

## 4 Core Operations

| Operation | Description |
|-----------|-------------|
| `believe()` | Store a belief with evidence. Merges duplicates via embedding similarity. |
| `retract()` | Invalidate evidence and recompute affected beliefs. |
| `explain()` | Return full justification chain — supporting, attacking, and expired evidence. |
| `revise()` | Add new evidence to an existing belief and recompute. |

## Quick Start

```bash
# Install
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Start API server
uv run python -m mnemebrain
```

## REST API

```bash
# Create a belief
curl -X POST http://localhost:8000/believe \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "user is vegetarian",
    "evidence": [{
      "source_ref": "msg_12",
      "content": "They said no meat please",
      "polarity": "supports",
      "weight": 0.8,
      "reliability": 0.9
    }]
  }'

# Explain a belief
curl "http://localhost:8000/explain?claim=user+is+vegetarian"

# Retract evidence
curl -X POST http://localhost:8000/retract \
  -H "Content-Type: application/json" \
  -d '{"evidence_id": "<uuid>"}'

# Revise with new evidence
curl -X POST http://localhost:8000/revise \
  -H "Content-Type: application/json" \
  -d '{
    "belief_id": "<uuid>",
    "evidence": {
      "source_ref": "msg_50",
      "content": "confirmed vegetarian",
      "polarity": "supports",
      "weight": 0.9,
      "reliability": 0.95
    }
  }'
```

## Python API

```python
from mnemebrain.memory import BeliefMemory
from mnemebrain.providers.base import EvidenceInput

memory = BeliefMemory(db_path="./my_data")

# Store a belief
result = memory.believe(
    claim="user is vegetarian",
    evidence_items=[
        EvidenceInput(
            source_ref="msg_12",
            content="They said no meat please",
            polarity="supports",
            weight=0.8,
            reliability=0.9,
        )
    ],
)
print(result.truth_state)  # TruthState.TRUE
print(result.confidence)   # > 0.5

# Explain
explanation = memory.explain("user is vegetarian")
print(explanation.supporting)
print(explanation.attacking)
```

## Architecture

```
src/mnemebrain/
├── models.py          # Belief, Evidence, TruthState, BeliefType
├── engine.py          # Pure functions: compute_truth_state, confidence, decay
├── store.py           # KuzuGraphStore — embedded graph DB
├── memory.py          # BeliefMemory — the 4 core operations
├── providers/
│   ├── base.py        # Abstract EmbeddingProvider
│   └── embeddings/    # sentence-transformers (local, free)
└── api/
    ├── app.py         # FastAPI application factory
    ├── routes.py      # REST endpoints
    └── schemas.py     # Request/response models
```

## Tech Stack

Python 3.12+, uv, FastAPI, Kuzu, sentence-transformers, Pydantic v2, pytest

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT
