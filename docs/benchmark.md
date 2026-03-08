# Benchmark

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
