# Architecture

## Source Layout

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

## Architecture Phases (mnemebrain-lite)

| Phase | Adds | Status |
|-------|------|--------|
| 1 | EvidenceLedger + TruthState + 4 core operations | Shipped |
| 1.5 | Confidence ranking + stability score + TruthState multiplier | Shipped |
| 2 | WorkingMemoryFrame (context cache) | Shipped |
| 2.5 | BeliefSandbox (copy-on-write hypothetical reasoning) | Shipped |
| 3 | AGM revision policies + ATTACKS edges | Shipped |
| 4 | Reconsolidation windows + GoalNode | Shipped |
| 4.5 | PolicyNode + EWMA learning + blame attribution | Shipped |

## Full Architecture (MnemeBrain Core — private)

| Phase | Adds | Status |
|-------|------|--------|
| 5 | ConsolidationDaemon + HippoRAG retrieval + pattern separation | In progress (see [mnemebrain](https://github.com/mnemebrain/mnemebrain)) |
