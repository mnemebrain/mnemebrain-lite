# MnemeBrain Lite

[![GitHub Release](https://img.shields.io/github/v/release/mnemebrain/mnemebrain-lite)](https://github.com/mnemebrain/mnemebrain-lite/releases)
[![codecov](https://codecov.io/gh/mnemebrain/mnemebrain-lite/graph/badge.svg)](https://codecov.io/gh/mnemebrain/mnemebrain-lite)

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
pip install mnemebrain-lite

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

### Search & List

```bash
# Semantic search with ranked scoring
curl "http://localhost:8000/search?query=vegetarian&limit=5&alpha=0.7"

# List beliefs with filters
curl "http://localhost:8000/beliefs?truth_state=TRUE&min_confidence=0.5&limit=20"
```

### Working Memory Frames

Working memory frames provide an active context buffer for multi-step reasoning episodes.

```bash
# Open a frame
curl -X POST http://localhost:8000/frame/open \
  -H "Content-Type: application/json" \
  -d '{"query_id": "<uuid>", "top_k": 20, "ttl_seconds": 300}'

# Add a belief to the frame
curl -X POST http://localhost:8000/frame/<frame_id>/add \
  -H "Content-Type: application/json" \
  -d '{"claim": "user is vegetarian"}'

# Get frame context (for LLM prompt injection)
curl http://localhost:8000/frame/<frame_id>/context

# Commit frame results back to the belief graph
curl -X POST http://localhost:8000/frame/<frame_id>/commit \
  -H "Content-Type: application/json" \
  -d '{"new_beliefs": [], "revisions": []}'

# Close/abandon a frame
curl -X DELETE http://localhost:8000/frame/<frame_id>
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
├── memory.py          # BeliefMemory — the 4 core operations + search/list
├── working_memory.py  # WorkingMemoryFrame — active context for multi-step reasoning
├── providers/
│   ├── base.py        # Abstract EmbeddingProvider
│   └── embeddings/    # sentence-transformers (optional)
└── api/
    ├── app.py         # FastAPI application factory
    ├── routes.py      # REST endpoints
    └── schemas.py     # Request/response models
```

## BMB Leaderboard

The **Belief Maintenance Benchmark** is an open benchmark for agent memory systems. 48 tasks, 8 categories — contradiction detection, belief revision, evidence tracking, temporal decay, counterfactual reasoning, consolidation, multi-hop retrieval, and pattern separation. Every RAG-based system scored 0% on contradiction detection — they overwrite instead of tracking conflicting evidence.

| System | Score |
|--------|------:|
| MnemeBrain | **100%** |
| Structured Memory | 36% |
| Mem0 (API) | 29% |
| Naive baseline | 0% |
| RAG baseline | 0% |
| OpenAI RAG (API) | 0% |
| LangChain buffer | 0% |

**Add your system.** Implement the [`MemorySystemAdapter`](src/mnemebrain_core/benchmark/interface.py) interface, drop it in `benchmark/adapters/`, and run:

```bash
pip install mnemebrain-lite[embeddings]
python run_bmb_benchmark.py --adapters your_adapter
```

We welcome adapters from competing systems. All tests are deterministic, all scoring is open-source, and we publish every result — including systems that outscore ours. See [benchmark/README.md](src/mnemebrain_core/benchmark/README.md) for adapter docs and [BMB_REPORT.md](src/mnemebrain_core/benchmark/BMB_REPORT.md) for full results.

## Tech Stack

Python 3.12+, uv, FastAPI, Kuzu, Pydantic v2, pytest, sentence-transformers (optional)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT
