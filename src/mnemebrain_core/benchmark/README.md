# Belief Maintenance Benchmark (BMB)

An open benchmark that measures whether a memory system can **maintain, revise, and explain beliefs over time**. 30 tasks across 5 categories.

See [BMB_REPORT.md](BMB_REPORT.md) for detailed results with real API comparisons.

## Why BMB exists

Standard memory benchmarks test retrieval. BMB tests **belief dynamics**: what happens when evidence conflicts, when time passes, when you need to reason hypothetically. These are the capabilities that separate a belief graph from a vector store.

## Structure: 30 tasks, 5 categories

| Category                | What it tests                              | Tasks |
|-------------------------|--------------------------------------------|-------|
| Contradiction Detection | Detecting BOTH state instead of overwriting | 6     |
| Belief Revision         | AGM-style revision with evidence chains    | 6     |
| Evidence Tracking       | explain() returns full justification chain | 6     |
| Temporal Updates        | Decay, staleness, time-validity expiry     | 6     |
| Counterfactual Reasoning| Sandbox simulation without canonical mutation | 6   |

## Quick Start

```bash
pip install mnemebrain-lite[embeddings]
python run_bmb_benchmark.py
```

Or with uv:

```bash
uv sync --extra embeddings
uv run python run_bmb_benchmark.py
```

## Adapters included

| Adapter              | Type              | Capabilities    | API Key |
|----------------------|-------------------|-----------------|---------|
| `naive_baseline`     | Flat vector store | store, query    | None    |
| `langchain_buffer`   | Text buffer       | store, query    | None    |
| `rag_baseline`       | RAG vector store  | store, query    | None    |
| `structured_memory`  | Mem0-style k/v    | store, query, retract, explain, revise | None |

Additional adapters available in the full [mnemebrain](https://pypi.org/project/mnemebrain/) package:
- `mnemebrain` -- full belief graph (9 capabilities)
- `mem0` -- real Mem0 cloud API
- `openai_rag` -- real OpenAI embeddings

## CLI options

```bash
# All baseline adapters
python run_bmb_benchmark.py

# Single adapter
python run_bmb_benchmark.py --adapter naive_baseline

# Single category
python run_bmb_benchmark.py --category contradiction

# Single scenario
python run_bmb_benchmark.py --scenario bmb_vegetarian_contradiction

# Custom output
python run_bmb_benchmark.py --output results/bmb_report.json
```

## Scoring

Each task produces 1-3 checks (binary pass/fail). Per-category scores are the average of scenario scores (0.0-1.0). Overall score is the average across categories.

| Check type          | What it verifies                          |
|---------------------|-------------------------------------------|
| Correct belief state | truth_state matches expected (true/false/both/neither) |
| Correct confidence   | confidence above/below threshold          |
| Explanation quality   | Evidence chain present, counts correct    |

## Adding your own adapter

Implement the `MemorySystem` ABC:

```python
from mnemebrain_core.benchmark.interface import (
    Capability, MemorySystem, StoreResult, QueryResult
)

class MyAdapter(MemorySystem):
    def name(self) -> str:
        return "my_system"

    def capabilities(self) -> set[Capability]:
        return {Capability.STORE, Capability.QUERY}

    def store(self, claim: str, evidence: list[dict]) -> StoreResult:
        ...

    def query(self, claim: str) -> list[QueryResult]:
        ...

    def reset(self) -> None:
        ...
```

**9 capabilities** (implement more to unlock more scenarios):

| Capability | Method | Unlocks |
|------------|--------|---------|
| `STORE` | `store()` | All scenarios |
| `QUERY` | `query()` | Most scenarios |
| `RETRACT` | `retract()` | Retraction, evidence tracking |
| `EXPLAIN` | `explain()` | Evidence tracking |
| `DECAY` | `set_time_offset_days()` | Temporal updates |
| `REVISE` | `revise()` | Belief revision |
| `SANDBOX` | `sandbox_fork/assume/resolve/discard()` | Counterfactual reasoning |
| `ATTACK` | `add_attack()` | Attack edge scenarios |

## Adding scenarios

Add entries to `scenarios/data/bmb_scenarios.json`:

```json
{
  "name": "bmb_your_scenario",
  "description": "What this tests",
  "category": "contradiction",
  "requires": ["store"],
  "actions": [
    {"label": "step1", "type": "store", "claim": "...", "evidence": [...]}
  ],
  "expectations": [
    {"action_label": "step1", "truth_state": "both"}
  ]
}
```

**Action types:** `store`, `query`, `retract`, `explain`, `wait_days`, `revise`, `sandbox_fork`, `sandbox_assume`, `sandbox_resolve`, `sandbox_discard`, `add_attack`

## Architecture

```
benchmark/
  bmb_cli.py               # CLI entry point
  interface.py             # MemorySystem ABC + 9 capabilities
  scoring.py               # Expectation evaluation (16 check types)
  system_runner.py         # Scenario executor (11 action types)
  system_report.py         # Scorecard + JSON export
  adapters/
    naive_baseline.py      # Flat vector store
    langchain_buffer.py    # Append-only text buffer
    rag_baseline.py        # RAG with overwrite-on-conflict
    structured_memory.py   # Mem0-style key-value memory
  scenarios/
    schema.py              # Action, Expectation, Scenario dataclasses
    loader.py              # JSON loader + validation
    data/
      bmb_scenarios.json   # 30 BMB scenarios
```

## Contributing

We welcome adapters from competing memory systems — the benchmark is only credible if others can challenge the leaderboard. All tests are deterministic and all scoring is open-source. We publish every result, including systems that outscore ours.

Submit a PR with your adapter or open an issue to discuss. See the project [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines.
