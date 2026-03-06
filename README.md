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

| Operation | Description | Requires embeddings? |
|-----------|-------------|:--------------------:|
| `believe()` | Store a belief with evidence. Merges duplicates via embedding similarity. | Yes |
| `retract()` | Invalidate evidence and recompute affected beliefs. | No |
| `explain()` | Return full justification chain — supporting, attacking, and expired evidence. | Yes |
| `revise()` | Add new evidence to an existing belief and recompute. | No |

## Quick Start

```bash
# Full install (Linux / Apple Silicon)
uv sync --extra dev --extra embeddings

# Without embeddings (Intel Mac — torch 2.3+ has no x86_64 wheels)
uv sync --extra dev

# Run tests
uv run pytest tests/ -v
# Without embeddings: unit tests pass, integration/e2e skip automatically

# Start API server
uv run python -m mnemebrain_core
```

> **Intel Mac note:** `sentence-transformers` requires PyTorch, which no longer
> ships x86_64 macOS wheels (torch 2.3+). Without embeddings, `/believe` and
> `/explain` return **501 Not Implemented**. All other endpoints work normally.
> To use all operations, provide a custom `EmbeddingProvider` or install on a
> supported platform.

## REST API

```bash
# Health check
curl http://localhost:8000/health

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

With embeddings installed, two additional endpoints are available:

```bash
# Create a belief (requires embeddings)
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

# Explain a belief (requires embeddings)
curl "http://localhost:8000/explain?claim=user+is+vegetarian"
```

For full endpoint documentation, see [docs/integration-api.md](docs/integration-api.md).

## Python API

```python
from mnemebrain_core.memory import BeliefMemory
from mnemebrain_core.providers.base import EvidenceInput

memory = BeliefMemory(db_path="./my_data")

# Retract evidence (always works)
memory.retract(evidence_id=some_uuid)

# Revise a belief (always works)
memory.revise(
    belief_id=belief_uuid,
    new_evidence=EvidenceInput(
        source_ref="msg_50",
        content="confirmed vegetarian",
        polarity="supports",
        weight=0.9,
        reliability=0.95,
    ),
)

# believe() and explain() require embeddings — see docs/integration-api.md
```

## Architecture

```
src/mnemebrain_core/
├── models.py          # Belief, Evidence, TruthState, BeliefType
├── engine.py          # Pure functions: compute_truth_state, confidence, decay
├── store.py           # KuzuGraphStore — embedded graph DB
├── memory.py          # BeliefMemory — the 4 core operations
├── providers/
│   ├── base.py        # Abstract EmbeddingProvider
│   └── embeddings/    # sentence-transformers (optional)
└── api/
    ├── app.py         # FastAPI application factory
    ├── routes.py      # REST endpoints
    └── schemas.py     # Request/response models
```

## Tech Stack

Python 3.12+, uv, FastAPI, Kuzu, Pydantic v2, pytest, sentence-transformers (optional)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT
