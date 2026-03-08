# Contributing to MnemeBrain Lite

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Getting Started

```bash
# Clone the repository
git clone git@github.com:mnemebrain/mnemebrain-lite.git
cd mnemebrain-lite

# Install dependencies (including dev extras)
uv sync --extra dev

# Run all tests
uv run pytest tests/ -v

# Run only unit tests (fast, no external deps)
uv run pytest tests/unit/ -v

# Run integration tests (requires sentence-transformers model download)
uv run pytest tests/integration/ -v -m integration

# Run e2e tests (full API tests)
uv run pytest tests/e2e/ -v -m e2e

# Lint (ruff check + format)
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Auto-fix lint issues
uv run ruff check --fix src/ tests/
uv run ruff format src/ tests/

# Type check (requires pyright install)
uv run pip install pyright
uv run pyright src/
```

## Project Structure

```
src/mnemebrain_core/
├── models.py          # Core data models (Belief, Evidence, TruthState)
├── engine.py          # Pure computation functions (truth state, confidence, decay)
├── store.py           # KuzuGraphStore — embedded graph database
├── memory.py          # BeliefMemory — the 4 core operations
├── working_memory.py  # WorkingMemoryFrame — active context for multi-step reasoning
├── providers/
│   ├── base.py        # Abstract EmbeddingProvider interface
│   └── embeddings/
│       ├── sentence_transformers.py  # Local embeddings (all-MiniLM-L6-v2)
│       └── openai.py                # OpenAI embeddings (text-embedding-3-small)
└── api/
    ├── app.py         # FastAPI application factory
    ├── routes.py      # REST endpoint handlers
    └── schemas.py     # Request/response Pydantic models

tests/
├── unit/              # Fast tests, no external dependencies
├── integration/       # Tests requiring Kuzu DB + embeddings
└── e2e/               # Full API endpoint tests
```

## How to Contribute

### Reporting Bugs

Open an issue using the **Bug Report** template. Include:
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS

### Suggesting Features

Open an issue using the **Feature Request** template. Describe:
- The problem you're solving
- Your proposed approach
- Alternatives you considered

### Submitting Code

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Write tests for your changes
4. Ensure all tests pass: `uv run pytest tests/ -v`
5. Run lint and format: `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`
6. Follow the code style (we use ruff for linting and formatting)
7. Commit with conventional commits: `feat(scope): description`
8. Push and open a Pull Request

### Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Usage |
|--------|-------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `test` | Adding or updating tests |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `chore` | Maintenance tasks |

Example: `feat(memory): add batch believe operation`

### Code Guidelines

- All public functions must have docstrings
- New features must include tests
- No `# type: ignore` or `# noqa` without justification
- Evidence is append-only — never delete, only invalidate
- Engine functions must be pure (no side effects, no mutations)

## Architecture Principles

MnemeBrain Lite follows these core design principles:

1. **Belnap 4-valued logic** — beliefs can be TRUE, FALSE, BOTH (contradiction), or NEITHER (insufficient evidence)
2. **Append-only evidence** — evidence is never deleted, only invalidated via `retract()`
3. **Time decay** — evidence weight decays based on belief type half-life
4. **Embedding dedup** — similar claims are merged automatically via embedding similarity
5. **Pure computation** — `engine.py` contains only pure functions; state lives in `store.py`

## Questions?

Open an issue with the **Question** template, or start a discussion.
