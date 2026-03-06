# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
