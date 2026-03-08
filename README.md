# MnemeBrain Lite

[![GitHub Release](https://img.shields.io/github/v/release/mnemebrain/mnemebrain-lite)](https://github.com/mnemebrain/mnemebrain-lite/releases)
[![PyPI](https://img.shields.io/pypi/v/mnemebrain-lite)](https://pypi.org/project/mnemebrain-lite/)
[![codecov](https://codecov.io/gh/mnemebrain/mnemebrain-lite/graph/badge.svg)](https://codecov.io/gh/mnemebrain/mnemebrain-lite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**The belief layer for AI agents.**

> ⭐ **Building AI agents?** Run the [BMB benchmark](https://github.com/mnemebrain/mnemebrain-benchmark) on your memory stack.
> ```bash
> pip install mnemebrain-benchmark
> bmb run
> ```
> No API keys. No LLM calls. See how your system handles contradictions, belief revision, and temporal decay — in 60 seconds.

---

## Project Structure

MnemeBrain has two layers:

- **mnemebrain-lite (this repo)** — Open-source core belief memory engine providing belief storage, evidence graphs, Belnap-style truth states, belief revision primitives, temporal decay, and semantic search.
- **MnemeBrain Core ([private research repo](https://github.com/mnemebrain/mnemebrain))** — Extended cognitive architecture used in the MnemeBrain research system, including consolidation daemons, advanced retrieval (HippoRAG-style), pattern separation, and long-term memory consolidation mechanisms.

```
         ┌─────────────────────────┐
         │    Agent / LLM Layer    │
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │        MnemeBrain       │
         │   (private research)    │
         │   consolidation engine  │
         │   advanced retrieval    │
         │   pattern separation    │
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │    mnemebrain-lite      │
         │   open-source belief    │
         │   memory engine         │
         │                         │
         │ evidence · truth ·      │
         │ revision · search       │
         └─────────────────────────┘
```

The lite engine is fully usable on its own and powers the public [MnemeBrain benchmark suite](https://github.com/mnemebrain/mnemebrain-benchmark).

---

## Why This Exists

Most AI agent memory systems store facts. But agents don't reason about facts — they reason about beliefs.

LLM memory systems retrieve information, but they rarely maintain beliefs.

```
User: "I'm vegetarian"
[later]
User: "I ate steak yesterday"
```

A RAG system either silently overwrites the first statement — or returns both without acknowledging the conflict. It cannot represent contradictions.

MnemeBrain is a belief memory engine that tracks contradiction, uncertainty, and revision instead of overwriting information. It borrows ideas from truth maintenance systems and belief revision research, but packages them into a developer-friendly memory engine for LLM agents.

---

## Quick Start

```bash
pip install mnemebrain-lite                # core only (no embeddings)
pip install mnemebrain-lite[embeddings]    # + local sentence-transformers
pip install mnemebrain-lite[openai]        # + OpenAI embeddings (set OPENAI_API_KEY)
pip install mnemebrain-lite[all]           # everything
```

```python
from mnemebrain_core.memory import BeliefMemory
from mnemebrain_core.providers.base import EvidenceInput

memory = BeliefMemory(db_path="./my_data")

# Store a belief
belief = memory.believe(
    claim="user is vegetarian",
    evidence=[EvidenceInput(
        source_ref="msg_12",
        content="They said no meat please",
        polarity="supports",
        weight=0.8,
        reliability=0.9,
    )]
)

# Introduce conflicting evidence — belief becomes BOTH (contradiction)
memory.revise(
    belief_id=belief.id,
    new_evidence=EvidenceInput(
        source_ref="msg_47",
        content="User ordered steak",
        polarity="attacks",
        weight=0.9,
        reliability=0.95,
    )
)

# Explain the contradiction
result = memory.explain(claim="user is vegetarian")
# → truth_state=BOTH, supporting_count=1, attacking_count=1
```

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

> **Intel Mac note:** `sentence-transformers` requires PyTorch, which no longer ships x86_64 macOS wheels (torch 2.3+). Use `mnemebrain-lite[openai]` instead — it works on any platform. Without any embedding provider, `/believe` and `/explain` return **501 Not Implemented**. All other endpoints work.

---

## Core Operations

| Operation | Description | Embeddings? |
|-----------|-------------|:-----------:|
| `believe()` | Store a belief with evidence. Merges duplicates via embedding similarity. | Yes |
| `retract()` | Invalidate evidence and recompute affected beliefs. | No |
| `explain()` | Return full justification chain — supporting, attacking, and expired evidence. | Yes |
| `revise()` | Add new evidence to an existing belief and recompute. | No |

---

## Documentation

Detailed documentation is available in the `/docs` folder:

- [Architecture](docs/architecture.md)
- [REST API](docs/integration-api.md)
- [Benchmark](docs/benchmark.md)
- [Design notes](docs/design.md)

---

## Tech Stack

Python 3.12+, uv, FastAPI, Kuzu, Pydantic v2, pytest, sentence-transformers or OpenAI embeddings (optional)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT
