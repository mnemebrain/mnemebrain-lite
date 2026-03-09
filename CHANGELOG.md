# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0a5] - 2026-03-09

### Fixed

- Add `asgi-lifespan>=2.0` to `dev` optional-dependencies (was missing, caused test skip)
- Fix `test_lifespan_via_asgi_lifespan` to use `app.state` directly instead of `manager.app.state`

## [0.1.0a4] - 2026-03-08

### Added

- Inter-triple relations module (`triple_relations.py`) — typed, weighted, directional edges between triples
- `TripleRelationType` enum with Stage 2 (`attacks`, `supports`, `depends_on`) and Stage 3 (`narrows`, `overrides`, `exception_to`, `derived_from`) variants
- `TripleRelation` Pydantic model with weight bounds validation and self-relation guard
- `RelationIndex` in-memory index with secondary indices by source/target triple ID
- 45 unit tests covering model validation, index CRUD, filtering, edge cases

## [0.1.0a3] - 2026-03-08

### Added

- OpenAI `text-embedding-3-small` embedding provider as fallback when sentence-transformers is unavailable
- Auto-detection of embedding provider: tries sentence-transformers first, then OpenAI if `OPENAI_API_KEY` is set
- `[openai]` optional dependency extra for OpenAI embeddings (`pip install mnemebrain-lite[openai]`)
- `load_dotenv(usecwd=True)` so `.env` files are found when running via `python -m`

### Fixed

- Error message now mentions both `[embeddings]` and `[openai]` install options
- Error message now references `uv pip` instead of bare `pip`

### Docs

- Document OpenAI embedding provider as install option in Quick Start and Architecture sections
- Reformat BMB section as competitive leaderboard with adapter invite
- Add competing-systems invite and clarify 0% claim applies to contradiction detection

## [0.1.0a2] - 2026-03-07

### Added

- Working memory frames for multi-step reasoning (`WorkingMemoryFrame`, `WorkingMemoryManager`)
- Frame lifecycle: open, add beliefs, scratchpad, commit, close with TTL-based expiry
- REST API endpoints: `/frame/open`, `/frame/{id}/add`, `/frame/{id}/scratchpad`, `/frame/{id}/context`, `/frame/{id}/commit`, `DELETE /frame/{id}`
- Semantic search endpoint (`GET /search`) with ranked scoring (similarity × confidence blend)
- Belief listing endpoint (`GET /beliefs`) with filtering by truth state, belief type, tag, and confidence range
- `ConflictPolicy` enum (surface, conservative, optimistic) for search result filtering
- `rank_score()` and `apply_conflict_policy()` engine functions
- Belief Maintenance Benchmark (BMB) module and runner for evaluating belief systems
- CI coverage reporting and README badges

### Changed

- `KuzuGraphStore.find_similar()` now returns `(Belief, similarity)` tuples instead of plain `Belief` objects
- `BeliefMemory.believe()` and `explain()` updated for new `find_similar()` return type
- `KuzuGraphStore.find_similar()` results are now sorted by similarity descending

## [0.1.0a1] - 2026-03-06

### Added

- Core belief memory system with 4 operations: `believe`, `retract`, `explain`, `revise`
- Belnap four-valued truth logic (TRUE, FALSE, BOTH, NEITHER)
- Evidence-based confidence scoring with time decay
- Kuzu embedded graph store for persistent belief storage
- Sentence-transformers embedding provider for semantic similarity
- FastAPI REST API with health, believe, retract, explain, and revise endpoints
- CLI entry point (`python -m mnemebrain_core`)
- Optional dependency extras: `[api]`, `[embeddings]`, `[openai]`, `[all]`, `[dev]`
- CI workflows: tests, lint, CodeQL, dependency review, pylint
- Release workflow: tag-triggered build + publish to PyPI and GitHub Releases

### Architecture

- Base package (`pydantic`, `kuzu`, `numpy`) is lightweight
- Heavy deps (`fastapi`, `sentence-transformers`, `torch`) are optional extras
- Stateless truth engine — pure functions for TruthState and confidence computation
- Append-only evidence ledger — evidence is never modified, only invalidated
