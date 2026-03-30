FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra api --no-install-project

COPY src/ src/
RUN uv sync --frozen --no-dev --extra api --no-editable

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "mnemebrain_core"]
