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
         │    MnemeBrain Core      │
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

## Benchmark Results

**Belief Maintenance Benchmark (BMB)** — 48 tasks · 8 categories · ~100 checks

```
  mnemebrain           ████████████████████ 100%
  structured_memory    ███████ 36%
  mem0 (real API)      █████ 29%
  openai_rag (real)     0%
  langchain_buffer      0%
  naive_baseline        0%
  rag_baseline          0%
```

| System | Score | Notes |
|--------|------:|-------|
| **MnemeBrain** | **100%** | 62/62 checks, all 30 scenarios |
| Structured Memory | 36% | No Belnap logic, no polarity tracking |
| Mem0 (real API) | 29% | Always `truth_state=true`, aggressive dedup |
| OpenAI RAG (real API) | 0% | `truth_state=None`, overwrites on conflict |
| LangChain Buffer | 0% | Store + query only |
| RAG Baseline | 0% | Store + query only |

Every RAG-based system scored **0% on contradiction detection**. They overwrite instead of tracking conflicting evidence.

Full results: [BMB_REPORT.md](https://github.com/mnemebrain/mnemebrain-benchmark/blob/main/BMB_REPORT.md)

---

## How It Works

```
Evidence (supports / attacks)
        ↓
   Belief Node
        ↓
 TruthState (TRUE / FALSE / BOTH / NEITHER)
        ↓
 Confidence + Temporal Decay
        ↓
    Agent API
```

**TruthState** uses Belnap's four-valued logic. Instead of overwriting on conflict, the system represents the contradiction explicitly with `BOTH` — then lets you resolve it with new evidence.

**Evidence Ledger** is append-only. Evidence is never deleted, only invalidated. Every belief carries a full justification chain: what supports it, what attacks it, and what has expired.

**Temporal Decay** degrades evidence weight by belief type:

| BeliefType | Half-life |
|------------|-----------|
| FACT | 365 days |
| PREFERENCE | 90 days |
| INFERENCE | 30 days |
| PREDICTION | 3 days |

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

## Formal Model

MnemeBrain is grounded in two well-established theories from knowledge representation and belief revision:

- **Belnap four-valued logic (1977)** — used to represent contradictory evidence without collapsing the belief system. Instead of overwriting, the system holds `BOTH` as a valid, stable state.
- **AGM belief revision (Alchourrón, Gärdenfors, Makinson, 1985)** — defines how a rational agent updates beliefs when new evidence arrives, with minimal disturbance to existing knowledge.

**TruthState** is computed over the evidence ledger using Belnap's lattice:

```
TruthState ∈ { TRUE, FALSE, BOTH, NEITHER }

TRUE     — net supporting evidence dominates
FALSE    — net attacking evidence dominates
BOTH     — significant supporting AND attacking evidence (contradiction)
NEITHER  — insufficient evidence to determine
```

**Confidence** is derived from weighted, time-decayed evidence:

```
confidence = Σ(support_weight × decay(t)) / (Σ(support_weight × decay(t)) + Σ(attack_weight × decay(t)))
```

where `decay(t) = 0.5 ^ (t / half_life)` and `half_life` varies by belief type (3 days for PREDICTION → 365 days for FACT).

**Belief ranking** uses a composite score across three signals:

```
rank_score = 0.60 × similarity        # semantic relevance to query
           + 0.25 × confidence        # evidence strength
           + 0.15 × stability         # inverse of revision volatility
```

Stability is `1 / (1 + revision_count)` — beliefs that have been revised frequently rank lower than beliefs that have been stable, even at equal confidence. This prevents contradicted high-confidence beliefs from polluting retrieval.

**Revision policy** follows AGM minimal change: when new evidence contradicts an existing belief, the system retracts the minimum set of evidence necessary to restore consistency. Pluggable policies (recency, confidence-weighted, entrenchment-based) determine selection order.

**Counterfactual reasoning** uses copy-on-write sandbox isolation: hypothetical evidence is applied to a forked belief graph, leaving the canonical state unchanged.

---

## REST API

```bash
# Start the server
uv run python -m mnemebrain_core

# Store a belief
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

# Explain a belief (returns truth_state, confidence, full evidence chain)
curl "http://localhost:8000/explain?claim=user+is+vegetarian"

# Revise with new evidence
curl -X POST http://localhost:8000/revise \
  -H "Content-Type: application/json" \
  -d '{
    "belief_id": "<uuid>",
    "evidence": {
      "source_ref": "msg_50",
      "content": "User ordered steak",
      "polarity": "attacks",
      "weight": 0.9,
      "reliability": 0.95
    }
  }'

# Retract evidence
curl -X POST http://localhost:8000/retract \
  -H "Content-Type: application/json" \
  -d '{"evidence_id": "<uuid>"}'

# Semantic search with ranked scoring
curl "http://localhost:8000/search?query=vegetarian&limit=5&alpha=0.7"

# List beliefs by state
curl "http://localhost:8000/beliefs?truth_state=BOTH&min_confidence=0.5"
```

### Working Memory Frames

Frames are active context buffers for multi-step reasoning episodes.

```bash
# Open a frame
curl -X POST http://localhost:8000/frame/open \
  -H "Content-Type: application/json" \
  -d '{"query_id": "<uuid>", "top_k": 20, "ttl_seconds": 300}'

# Add a belief to the frame
curl -X POST http://localhost:8000/frame/<frame_id>/add \
  -H "Content-Type: application/json" \
  -d '{"claim": "user is vegetarian"}'

# Get context for LLM prompt injection
curl http://localhost:8000/frame/<frame_id>/context

# Commit results back to the belief graph
curl -X POST http://localhost:8000/frame/<frame_id>/commit \
  -H "Content-Type: application/json" \
  -d '{"new_beliefs": [], "revisions": []}'
```

Full endpoint docs: [docs/integration-api.md](docs/integration-api.md)

---

## Architecture

```
src/mnemebrain_core/
├── models.py            # Belief, Evidence, TruthState, BeliefType
├── engine.py            # Pure functions: compute_truth_state, confidence, decay
├── store.py             # KuzuGraphStore — embedded graph DB
├── memory.py            # BeliefMemory — 4 core operations + search/list
├── working_memory.py    # WorkingMemoryFrame — active context for multi-step reasoning
├── triple_relations.py  # TripleRelation — typed inter-triple edges (attacks, supports, depends_on)
├── providers/
│   ├── base.py        # Abstract EmbeddingProvider
│   └── embeddings/    # sentence-transformers or OpenAI (optional)
└── api/
    ├── app.py         # FastAPI application factory
    ├── routes.py      # REST endpoints
    └── schemas.py     # Request/response models
```

**Architecture phases (mnemebrain-lite):**

| Phase | Adds | Status |
|-------|------|--------|
| 1 | EvidenceLedger + TruthState + 4 core operations | ✅ Shipped |
| 1.5 | Confidence ranking + stability score + TruthState multiplier | ✅ Shipped |
| 2 | WorkingMemoryFrame (context cache) | ✅ Shipped |
| 2.5 | BeliefSandbox (copy-on-write hypothetical reasoning) | ✅ Shipped |
| 3 | AGM revision policies + ATTACKS edges | ✅ Shipped |
| 4 | Reconsolidation windows + GoalNode | ✅ Shipped |
| 4.5 | PolicyNode + EWMA learning + blame attribution | ✅ Shipped |

**Full architecture (MnemeBrain Core — private):**

| Phase | Adds | Status |
|-------|------|--------|
| 5 | ConsolidationDaemon + HippoRAG retrieval + pattern separation | In progress (see [mnemebrain](https://github.com/mnemebrain/mnemebrain)) |

---

## BMB Leaderboard

The **Belief Maintenance Benchmark** is an open evaluation for agent memory systems. 48 tasks, 8 categories — contradiction detection, belief revision, evidence tracking, temporal decay, retraction, dedup, extraction, and lifecycle.

| System | Score |
|--------|------:|
| **MnemeBrain** | **100%** |
| Structured Memory | 36% |
| Mem0 (real API) | 29% |
| OpenAI RAG (real API) | 0% |
| LangChain Buffer | 0% |
| RAG Baseline | 0% |

**Add your system.** Implement the [`MemorySystemAdapter`](https://github.com/mnemebrain/mnemebrain-benchmark/blob/main/src/mnemebrain_benchmark/interface.py) interface, drop it in `adapters/`, and run:

```bash
pip install mnemebrain-benchmark
bmb run --adapters your_adapter
```

All tests are deterministic. All scoring is open-source. We publish every result — including systems that outscore ours.

Adapter docs: [mnemebrain-benchmark README](https://github.com/mnemebrain/mnemebrain-benchmark) · Full results: [BMB_REPORT.md](https://github.com/mnemebrain/mnemebrain-benchmark/blob/main/BMB_REPORT.md)

---

## References

- Belnap, N. D. (1977). A useful four-valued logic. In *Modern Uses of Multiple-Valued Logic*. Reidel.
- Alchourrón, C. E., Gärdenfors, P., & Makinson, D. (1985). On the logic of theory change: Partial meet contraction and revision functions. *Journal of Symbolic Logic*, 50(2), 510–530.
- Lewis, D. (1973). *Counterfactuals*. Harvard University Press.
- Gutierrez, B. J., et al. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. *NeurIPS 2024*.

---

## Tech Stack

Python 3.12+, uv, FastAPI, Kuzu, Pydantic v2, pytest, sentence-transformers or OpenAI embeddings (optional)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT