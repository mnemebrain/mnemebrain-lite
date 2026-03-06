"""FastAPI application factory."""
from __future__ import annotations

from fastapi import FastAPI

from mnemebrain.api.routes import router, set_memory
from mnemebrain.memory import BeliefMemory


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
    return app
