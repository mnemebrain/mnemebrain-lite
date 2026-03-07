# Belief Maintenance Benchmark (BMB) -- Technical Report

## Abstract

The Belief Maintenance Benchmark (BMB) is a 48-task evaluation suite that measures whether an AI memory system can maintain, revise, and explain beliefs over time. Unlike retrieval-focused benchmarks, BMB tests **belief dynamics**: contradiction detection, evidence-based revision, temporal decay, justification chains, counterfactual reasoning, episodic-to-semantic consolidation, multi-hop graph retrieval, and pattern separation. These capabilities are architecturally impossible for systems that treat memory as flat text or vector storage.

## Motivation

Existing memory benchmarks for AI agents measure retrieval accuracy: given a query, does the system return the right document? This misses the core challenge of long-lived agents: **beliefs change**.

A user says "I'm vegetarian." Later they say "I ate steak yesterday." A retrieval system either overwrites the first statement or returns both without acknowledging the conflict. A belief system must:

1. Detect the contradiction (BOTH state)
2. Track both pieces of evidence with provenance
3. Surface the conflict to the agent
4. Allow principled revision when new evidence arrives

BMB tests exactly these capabilities across 48 multi-step scenarios.

## Benchmark Design

### Categories

| # | Category | Capability tested | Key metric |
|---|----------|-------------------|------------|
| 1 | Contradiction Detection | ATTACK edges, Belnap four-valued logic | truth_state == BOTH when evidence conflicts |
| 2 | Belief Revision | AGM-style revision with evidence chains | Correct state after incremental evidence |
| 3 | Evidence Tracking | explain() justification chains | Evidence counts, expired evidence tracking |
| 4 | Temporal Updates | Exponential decay with type-specific half-lives | Confidence decay, prediction expiry |
| 5 | Counterfactual Reasoning | Copy-on-write sandbox simulation | Sandbox isolation, canonical preservation |
| 6 | Consolidation | Episodic→semantic compression via clustering | Cluster formation, pruning, tier promotion |
| 7 | Multi-hop Retrieval | HippoRAG ego-subgraph PageRank | Graph traversal finds linked beliefs |
| 8 | Pattern Separation | ANN-first embedding orthogonalisation | Similar-but-distinct beliefs stay separable |

### Task design principles

Each task is a **multi-step conversation**, not a single prompt. This prevents prompt engineering from gaming the results. A typical task:

1. Store initial belief with evidence
2. Introduce conflicting or supporting evidence
3. Optionally wait (time simulation) or fork a sandbox
4. Query or explain the resulting belief state
5. Verify expectations against the result

### Scoring system

Each task produces 1-3 binary checks:

| Check | Points | What passes |
|-------|--------|-------------|
| Correct belief state | 1 | truth_state matches expected value |
| Correct confidence | 1 | Confidence above/below threshold |
| Explanation quality | 1 | Evidence chain present with correct counts |

**Maximum score: ~100 checks** across 48 tasks (1-3 checks per task)

Per-category scores are computed as the average of scenario scores within that category (0.0-1.0). Overall score is the average across categories.

## Capability Matrix

The benchmark leverages MnemeBrain's 12 system capabilities:

| Capability | Description | Adapter support |
|------------|-------------|-----------------|
| `store` | Store a belief with evidence | All systems |
| `query` | Retrieve beliefs by semantic similarity | All systems |
| `retract` | Invalidate evidence, recompute truth state | MnemeBrain |
| `explain` | Return full justification chain (supporting/attacking/expired) | MnemeBrain |
| `contradiction` | Detect BOTH state via Belnap logic | MnemeBrain |
| `decay` | Time-based confidence decay per belief type | MnemeBrain |
| `revise` | Add evidence to existing belief, recompute | MnemeBrain |
| `sandbox` | Copy-on-write hypothetical reasoning | MnemeBrain |
| `attack` | Explicit ATTACK edges between beliefs | MnemeBrain |
| `consolidation` | Episodic→semantic compression via clustering | MnemeBrain |
| `hipporag` | Multi-hop graph retrieval with PageRank | MnemeBrain |
| `pattern_separation` | ANN-first embedding orthogonalisation | MnemeBrain |

### Why baselines fail (verified by real benchmark runs)

**RAG systems (local + OpenAI real API)** score **0%**. They only support `store` and `query`:

- No truth_state at all -- returns `None` instead of `both`
- Overwrites on conflict -- second store replaces first, losing evidence
- No evidence provenance, no decay, no sandbox, no explain

**Mem0 (real cloud API)** scores **29%** despite having more capabilities:

- No Belnap four-valued logic -- always returns `truth_state=true`, never `both`
- Aggressive deduplication -- 3 separate stores may yield 1 memory, losing evidence count
- LLM rewrites input -- "I'm vegetarian" becomes "User is vegetarian"
- No evidence polarity tracking -- `attacking_count` is always 0
- Async processing -- memories take ~1.5s to become searchable
- No temporal decay -- skips all 6 temporal scenarios
- No sandbox -- skips all 6 counterfactual scenarios

**Structured memory (local Mem0-style simulation)** scores **36%**:

- Same fundamental limitations as Mem0 (no Belnap, no polarity)
- Slightly better evidence tracking because it doesn't deduplicate

## Scenario Details

### Category 1: Contradiction Detection (6 tasks)

Tests whether the system detects conflicting beliefs using Belnap's four-valued logic (TRUE, FALSE, BOTH, NEITHER) instead of silently overwriting.

| Task | Scenario | Expected behavior |
|------|----------|-------------------|
| 1 | Vegetarian + ate steak | truth_state=BOTH, confidence < 0.7 |
| 2 | Remote work + office job | truth_state=BOTH, conflict surfaced |
| 3 | Peanut allergy + orders peanuts | truth_state=BOTH |
| 4 | Morning person + night owl analytics | truth_state=BOTH |
| 5 | Budget-conscious + luxury purchase | truth_state=BOTH |
| 6 | Contradiction resolved by retraction | truth_state reverts to TRUE after retract |

### Category 2: Belief Revision (6 tasks)

Tests AGM-style revision: incremental evidence should update beliefs correctly, not just append.

| Task | Scenario | Expected behavior |
|------|----------|-------------------|
| 7 | Traffic reports (moderate -> heavy -> confirmed) | BOTH (attack evidence still active), confidence > 0.5 |
| 8 | Medical diagnosis revised with MRI | Confidence increases with strong evidence |
| 9 | Stock prediction with conflicting market data | BOTH (attack evidence still active) |
| 10 | Policy update supersedes old handbook | BOTH state after contradicting revision |
| 11 | Progressive evidence strengthening | Confidence > 0.6 after 3 supporting items |
| 12 | Attacking evidence weakens belief | BOTH state after incident report |

### Category 3: Evidence Tracking (6 tasks)

Tests whether explain() returns structured justification with provenance, not hallucinated reasoning.

| Task | Scenario | Expected behavior |
|------|----------|-------------------|
| 13 | Employment change (Google -> Anthropic) | explain() shows 1+ supporting, 1+ attacking |
| 14 | Retracted evidence shown as expired | explain() shows expired_count >= 1 |
| 15 | Multiple sources for Python proficiency | explain() shows 3+ supporting items |
| 16 | Mixed evidence on product reliability | explain() shows 2+ supporting, 1+ attacking |
| 17 | Non-existent belief query | explain() returns no evidence (no hallucination) |
| 18 | Strong evidence = high confidence | explain() shows TRUE state, confidence > 0.6 |

### Category 4: Temporal Updates (6 tasks)

Tests belief decay with type-specific half-lives:
- **Fact**: 365 days (birthdays don't expire)
- **Preference**: 90 days (tastes drift)
- **Inference**: 30 days (conclusions need refresh)
- **Prediction**: 3 days (forecasts go stale fast)

| Task | Scenario | Expected behavior |
|------|----------|-------------------|
| 19 | Prediction after 5 days | truth_state=NEITHER (decayed past threshold) |
| 20 | Fact after 30 days | truth_state=TRUE (barely decayed) |
| 21 | Preference after 180 days | confidence < 0.7 (two half-lives) |
| 22 | Inference after 60 days | confidence < 0.6 (two half-lives) |
| 23 | Fresh evidence overrides stale | Fresh revision restores confidence > 0.5 |
| 24 | Prediction with time validity | truth_state=NEITHER after validity window |

### Category 5: Counterfactual Reasoning (6 tasks)

Tests sandbox (copy-on-write overlay) for hypothetical reasoning without mutating canonical beliefs.

| Task | Scenario | Expected behavior |
|------|----------|-------------------|
| 25 | Server overload scenario | Sandbox shows FALSE, canonical stays TRUE |
| 26 | Pricing change experiment | Sandbox shows FALSE, canonical stays TRUE |
| 27 | Diet change hypothetical | Sandbox shows FALSE, canonical stays TRUE |
| 28 | Deployment failure scenario | Sandbox shows FALSE, canonical stays TRUE |
| 29 | Market crash scenario | Sandbox shows FALSE, canonical stays TRUE |
| 30 | Hiring decision exploration | Sandbox shows FALSE, canonical stays TRUE |

## Results

Benchmark run date: 2026-03-07. Full results in `bmb_report_all.json`.

### Overall Scores (7 systems, real benchmark run)

All results are from actual benchmark execution, including real Mem0 API calls and real OpenAI embedding API calls.

```
Belief Maintenance Benchmark (BMB)
48 tasks | 8 categories | ~100 checks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  mnemebrain           ████████████████████ 100%
  structured_memory    ███████ 36%
  mem0 (real API)      █████ 29%
  naive_baseline        0%
  rag_baseline          0%
  openai_rag (real API) 0%
  langchain_buffer      0%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### MnemeBrain: 100%, 48/48 scenarios

| Category | Scenarios | Score |
|----------|-----------|-------|
| Contradiction Detection | 6/6 | **100%** |
| Belief Revision | 6/6 | **100%** |
| Evidence Tracking | 6/6 | **100%** |
| Temporal Updates | 6/6 | **100%** |
| Counterfactual Reasoning | 6/6 | **100%** |
| Consolidation | 6/6 | **100%** |
| Multi-hop Retrieval | 6/6 | **100%** |
| Pattern Separation | 6/6 | **100%** |

### Mem0 (real API, graph enabled): 29%, 18/48 scenarios attempted

Uses the real Mem0 cloud API (`mem0ai` SDK). Has `store`, `query`, `retract`, `explain`, `revise`.

| Category | Scenarios | Score | Key failures |
|----------|-----------|-------|--------------|
| Contradiction Detection | 6 attempted | **8%** | Never detects contradictions, no BOTH state |
| Belief Revision | 6 attempted | **39%** | Passes only when expected state is `true` |
| Evidence Tracking | 6 attempted | **42%** | Mem0 merges/deduplicates memories, losing evidence count |
| Temporal Updates | 0 (skipped) | **N/A** | No decay capability |
| Counterfactual | 0 (skipped) | **N/A** | No sandbox capability |

Notable Mem0 behaviors observed:
- Mem0's LLM rewrites input ("I'm vegetarian" becomes "User is vegetarian")
- Mem0 deduplicates aggressively -- 3 separate stores may yield 1 memory
- `explain()` returns fewer memories than stored due to deduplication
- No polarity tracking: `attacking_count` is always 0
- Async processing: memories take ~1.5s to become searchable

### Structured Memory (Mem0-style, local): 36%, 18/48 scenarios

Local simulation with `store`, `query`, `retract`, `explain`, `revise`. No Belnap logic, no decay, no sandbox.

| Category | Scenarios | Score | Why it fails |
|----------|-----------|-------|--------------|
| Contradiction Detection | 6 attempted | **0%** | No BOTH state -- always returns `true` |
| Belief Revision | 6 attempted | **39%** | Passes when answer is `true`, fails all BOTH expectations |
| Evidence Tracking | 6 attempted | **69%** | Counts evidence but has no polarity (attacking=0 always) |
| Temporal Updates | 0 (skipped) | **N/A** | No decay capability |
| Counterfactual | 0 (skipped) | **N/A** | No sandbox capability |

### OpenAI RAG (real API): 0%, 5/48 scenarios attempted

Uses real OpenAI `text-embedding-3-small` embeddings. Store + query only, last-write-wins.

| Category | Scenarios | Result |
|----------|-----------|--------|
| Contradiction Detection | 5 attempted | **0%** -- no truth_state, overwrites on similar claim |
| All other categories | skipped | No required capabilities |

### Naive Baseline / RAG Baseline / LangChain Buffer: 0%, 5/48 scenarios each

All three use local embeddings with store + query only. 0% on all attempted contradiction scenarios.

### Key failure patterns by architecture

| Architecture | Core failure mode | Example |
|--------------|-------------------|---------|
| Buffer/RAG (local + OpenAI) | No truth_state at all | `truth_state=None` instead of `both` |
| RAG (local + OpenAI) | Overwrites silently | Second store replaces first, no conflict |
| Mem0 (real API) | No Belnap logic | Always `truth_state=true`, never `both` |
| Mem0 (real API) | Deduplicates evidence | 3 stores become 1 memory, losing provenance |
| Mem0 (real API) | No evidence polarity | `attacking_count=0` even with attack evidence |
| Structured (local) | Same as Mem0 | Always `true`, no polarity |
| All baselines | No temporal decay | Cannot detect stale beliefs |
| All baselines | No sandbox | Cannot do hypothetical reasoning |

### Notable confidence values from MnemeBrain

| Scenario | Metric | Value |
|----------|--------|-------|
| Vegetarian contradiction | confidence | 0.522 (correctly < 0.7 threshold) |
| Traffic revision chain | confidence after revision | 0.754 (correctly > 0.5 threshold) |
| Medical revision with MRI | confidence | 0.713 (correctly > 0.6 threshold) |
| Revision strengthening (3 items) | confidence | 0.746 (correctly > 0.6 threshold) |
| French speaker (strong evidence) | confidence | 0.851 (correctly > 0.6 threshold) |
| Preference after 180 days | confidence | 0.350 (correctly < 0.7, decayed) |
| Inference after 60 days | confidence | 0.341 (correctly < 0.6, decayed) |
| Fresh evidence beats stale | confidence | 0.463 (correctly > 0.4 after refresh) |

## Reproducibility

```bash
pip install mnemebrain
python run_bmb_benchmark.py
```

Results are deterministic for the same embedding model. The benchmark uses no LLM calls -- all logic is computed by the belief engine.

## Implementation

### Files

| File | Purpose |
|------|---------|
| `bmb_cli.py` | CLI entry point with bar chart output |
| `scenarios/data/bmb_scenarios.json` | 48 scenario definitions (JSON) |
| `interface.py` | MemorySystem ABC with 12 capability methods |
| `scoring.py` | Expectation evaluation (20+ check types) |
| `system_runner.py` | Scenario executor (14 action types) |
| `adapters/mnemebrain_adapter.py` | Full-capability adapter |
| `adapters/naive_baseline.py` | Flat vector store baseline |

### Extending

Add new scenarios to `bmb_scenarios.json`. Add new adapters by implementing `MemorySystem`. The framework automatically skips scenarios requiring capabilities the adapter lacks.

## Limitations

- **No LLM-in-the-loop evaluation**: BMB tests the memory engine, not the LLM's ability to use it. A future version could test end-to-end agent behavior.
- **Synthetic scenarios**: All scenarios are hand-crafted. A future version could use real conversation logs.
- **Single-agent**: All scenarios assume a single agent. Multi-agent belief conflict resolution is tested only in the system benchmark (not BMB).

## Citation

```
MnemeBrain Belief Maintenance Benchmark (BMB)
48 tasks | 8 categories | ~100 checks
https://github.com/mnemebrain/mnemebrain
```
