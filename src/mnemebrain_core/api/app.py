"""FastAPI application factory."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.requests import Request

from mnemebrain_core.api.routes import router
from mnemebrain_core.memory import BeliefMemory
from mnemebrain_core.working_memory import WorkingMemoryManager


def create_app(db_path: str = "./mnemebrain_data") -> FastAPI:
    """Create FastAPI app with BeliefMemory."""
    try:
        from dotenv import find_dotenv, load_dotenv  # noqa: PLC0415

        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        memory = BeliefMemory(db_path=db_path)
        wm_manager = WorkingMemoryManager(memory)
        app.state.memory = memory
        app.state.wm_manager = wm_manager

        async def gc_loop():
            while True:
                await asyncio.sleep(60)
                wm_manager.gc_frames()

        gc_task = asyncio.create_task(gc_loop())
        app.state.gc_task = gc_task
        yield
        gc_task.cancel()
        memory.close()

    app = FastAPI(
        title="MnemeBrain Lite",
        description="Biological belief memory for LLM agents",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.exception_handler(ImportError)
    async def handle_missing_embeddings(
        _request: Request, exc: ImportError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=501,
            content={
                "detail": (
                    "Embedding provider not available. "
                    "Install mnemebrain-lite[embeddings] or mnemebrain-lite[openai]."
                )
            },
        )

    return app
