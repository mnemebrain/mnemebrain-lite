"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.requests import Request

from mnemebrain_core.api.routes import router, set_memory
from mnemebrain_core.memory import BeliefMemory


def create_app(db_path: str = "./mnemebrain_data") -> FastAPI:
    """Create FastAPI app with BeliefMemory."""
    app = FastAPI(
        title="MnemeBrain Lite",
        description="Biological belief memory for LLM agents",
        version="0.1.0",
    )
    memory = BeliefMemory(db_path=db_path)
    set_memory(memory)
    app.include_router(router)

    @app.exception_handler(ImportError)
    async def handle_missing_embeddings(_request: Request, exc: ImportError) -> JSONResponse:
        return JSONResponse(
            status_code=501,
            content={"detail": str(exc)},
        )

    return app
