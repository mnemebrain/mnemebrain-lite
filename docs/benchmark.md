# Benchmark

## Benchmark Results

**Belief Maintenance Benchmark (BMB)** — 48 tasks · 8 categories · ~100 checks

```
  mnemebrain (full)    ████████████████████ 100%
  mnemebrain_lite      ██████████████████   93%
  structured_memory    ███████              36%
  mem0 (real API)      █████                29%
  openai_rag (real)                          0%
  langchain_buffer                           0%
  naive_baseline                             0%
  rag_baseline                               0%
```

### MnemeBrain Lite: 93% BMB

| Category | Score | Notes |
|----------|------:|-------|
| Belief Revision | **100%** | Full parity with backend |
| Evidence Tracking | **100%** | Full parity with backend |
| Contradiction Detection | **91.7%** | Belnap BOTH state working |
| Temporal Updates | **83.3%** | Type-specific decay working |
| Counterfactual Reasoning | N/A | Requires sandbox (full backend) |
| Consolidation | N/A | Requires consolidation daemon (full backend) |
| Multi-hop Retrieval | N/A | Requires HippoRAG (full backend) |
| Pattern Separation | N/A | Requires ANN index (full backend) |

On the 4 categories Lite supports, it scores **93.8%** — the core belief engine is the architectural differentiator (93% vs 0-36% for all baselines).

### System Benchmark (22 scenarios)

| Category | mnemebrain_lite | naive_baseline |
|----------|:---------------:|:--------------:|
| Contradiction | **87.5%** | N/A |
| Decay | **100%** | N/A |
| Dedup | 50.0% | 75.0% |
| Extraction | **100%** | 75.0% |
| Lifecycle | **62.5%** | 50.0% |
| Retraction | **83.3%** | N/A |
| **Overall** | **80.6%** | **66.7%** |

### Task-Level Evaluations (18 scenarios, ~59 questions)

Measures downstream task correctness — does better memory produce correct answers?

| Suite | mnemebrain_lite | naive_baseline |
|-------|:---------------:|:--------------:|
| Preference Tracking (31 questions) | **74.2%** | 32.3% |
| Long-Horizon QA (25 questions) | **76.0%** | 24.0% |

Lite's retraction, revision, and truth_state filtering give it 2-3x the accuracy of baselines on real preference tracking and QA tasks.

### Comparison: Lite vs Full Backend

| Feature | mnemebrain_lite | mnemebrain (full) |
|---------|:-:|:-:|
| BMB Score | 93% (4 categories) | 100% (8 categories) |
| Task Eval Accuracy | ~75% | ~95% |
| Deployment | `pip install` | Server + SDK |
| Belnap logic | Yes | Yes |
| Evidence ledger | Yes | Yes |
| Temporal decay | Yes | Yes |
| Sandbox | No | Yes |
| Consolidation | No | Yes |
| HippoRAG | No | Yes |
| Pattern separation | No | Yes |

Full comparison: [LITE_VS_FULL_REPORT.md](https://github.com/mnemebrain/mnemebrain-benchmark/blob/main/LITE_VS_FULL_REPORT.md)

## BMB Leaderboard

The **Belief Maintenance Benchmark** is an open evaluation for agent memory systems. 48 tasks, 8 categories — contradiction detection, belief revision, evidence tracking, temporal decay, counterfactual reasoning, consolidation, multi-hop retrieval, and pattern separation.

| System | Score |
|--------|------:|
| **MnemeBrain (full)** | **100%** |
| **MnemeBrain Lite** | **93%** |
| Structured Memory | 36% |
| Mem0 (real API) | 29% |
| OpenAI RAG (real API) | 0% |
| LangChain Buffer | 0% |
| RAG Baseline | 0% |

**Add your system.** Implement the [`MemorySystem`](https://github.com/mnemebrain/mnemebrain-benchmark/blob/main/src/mnemebrain_benchmark/interface.py) interface, drop it in `adapters/`, and run:

```bash
pip install mnemebrain-benchmark[embeddings]
mnemebrain-bmb --adapter your_adapter
```

All tests are deterministic. All scoring is open-source.

Adapter docs: [mnemebrain-benchmark README](https://github.com/mnemebrain/mnemebrain-benchmark) · Full results: [BMB_REPORT.md](https://github.com/mnemebrain/mnemebrain-benchmark/blob/main/BMB_REPORT.md)

## Run Benchmarks Locally

```bash
# Install mnemebrain-lite with embeddings
pip install mnemebrain-lite[embeddings]

# Clone and install the benchmark suite
git clone https://github.com/mnemebrain/mnemebrain-benchmark
cd mnemebrain-benchmark
pip install -e ".[embeddings]"

# Run BMB
mnemebrain-bmb --adapter mnemebrain_lite

# Run system benchmark
mnemebrain-benchmark --adapter mnemebrain_lite

# Run task evaluations
python -m mnemebrain_benchmark.task_evals --adapter mnemebrain_lite
```
