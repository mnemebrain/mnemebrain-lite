# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0a2] - 2026-03-07

### Added

- Working memory frames for multi-step reasoning (`WorkingMemoryFrame`, `WorkingMemoryManager`)
- Frame lifecycle: open, add beliefs, scratchpad, commit, close with TTL-based expiry
- REST API endpoints: `/frame/open`, `/frame/{id}/add`, `/frame/{id}/scratchpad`, `/frame/{id}/context`, `/frame/{id}/commit`, `DELETE /frame/{id}`
- Semantic search endpoint (`GET /search`) with ranked scoring (similarity × confidence blend)
- Belief listing endpoint (`GET /beliefs`) with filtering by truth state, belief type, tag, and confidence range
- `ConflictPolicy` enum (surface, conservative, optimistic) for search result filtering
- `rank_score()` and `apply_conflict_policy()` engine functions

### Changed

- `KuzuGraphStore.find_similar()` now returns `(Belief, similarity)` tuples instead of plain `Belief` objects
- `BeliefMemory.believe()` and `explain()` updated for new `find_similar()` return type
- `KuzuGraphStore.find_similar()` results are now sorted by similarity descending

## [1.0.0a1] - 2026-03-06

### Added

- Core belief memory system with 4 operations: `believe`, `retract`, `explain`, `revise`
- Belnap four-valued truth logic (TRUE, FALSE, BOTH, NEITHER)
- Evidence-based confidence scoring with time decay
- Kuzu embedded graph store for persistent belief storage
- Sentence-transformers embedding provider for semantic similarity
- FastAPI REST API with health, believe, retract, explain, and revise endpoints
- CLI entry point (`python -m mnemebrain_core`)
- Optional dependency extras: `[api]`, `[embeddings]`, `[all]`, `[dev]`
- CI workflows: tests, lint, CodeQL, dependency review, pylint
- Release workflow: tag-triggered build + publish to PyPI and GitHub Releases

### Architecture

- Base package (`pydantic`, `kuzu`, `numpy`) is lightweight
- Heavy deps (`fastapi`, `sentence-transformers`, `torch`) are optional extras
- Stateless truth engine — pure functions for TruthState and confidence computation
- Append-only evidence ledger — evidence is never modified, only invalidated
