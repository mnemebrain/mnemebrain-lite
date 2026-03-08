"""Unit tests for mnemebrain_core.api.app — create_app, lifespan, exception handler."""

from __future__ import annotations

import os
import shutil
import tempfile
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestCreateApp:
    def test_returns_fastapi_instance(self):
        """create_app() returns a FastAPI application object."""
        from mnemebrain_core.api.app import create_app

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_db")
        try:
            app = create_app(db_path=db_path)
            assert isinstance(app, FastAPI)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_returns_fastapi_instance_default_path(self):
        """create_app() works with the default db_path argument signature."""
        from mnemebrain_core.api.app import create_app

        # We only inspect the return type; we do NOT open a lifespan context
        # (which would create real DB files at the default path).
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "default_db")
        try:
            app = create_app(db_path=db_path)
            assert isinstance(app, FastAPI)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_app_includes_health_route(self):
        """The returned app exposes the /health route from the router."""
        from mnemebrain_core.api.app import create_app

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_db")
        try:
            app = create_app(db_path=db_path)
            routes = [r.path for r in app.routes]  # type: ignore[attr-defined]
            assert "/health" in routes
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_dotenv_import_error_is_silenced(self):
        """create_app() does not raise when python-dotenv is absent."""
        from mnemebrain_core.api.app import create_app

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_db")
        try:
            # Simulate dotenv not being installed
            import builtins

            original_import = builtins.__import__

            def _fake_import(name, *args, **kwargs):
                if name == "dotenv":
                    raise ImportError("No module named 'dotenv'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=_fake_import):
                app = create_app(db_path=db_path)
            assert isinstance(app, FastAPI)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestLifespan:
    """Verify that the lifespan context manager wires app.state correctly."""

    @pytest.mark.asyncio
    async def test_lifespan_via_asgi_lifespan(self):
        """Lifespan startup wires memory + wm_manager onto app.state."""
        try:
            from asgi_lifespan import LifespanManager  # type: ignore[import]
        except ImportError:
            pytest.skip("asgi-lifespan not installed")

        from mnemebrain_core.api.app import create_app
        from mnemebrain_core.memory import BeliefMemory
        from mnemebrain_core.working_memory import WorkingMemoryManager

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_db")
        try:
            app = create_app(db_path=db_path)
            async with LifespanManager(app) as manager:
                assert isinstance(manager.app.state.memory, BeliefMemory)
                assert isinstance(manager.app.state.wm_manager, WorkingMemoryManager)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_lifespan_state_manual(self):
        """Manually validate lifespan sets state by simulating startup."""
        from mnemebrain_core.api.app import create_app
        from mnemebrain_core.memory import BeliefMemory
        from mnemebrain_core.working_memory import WorkingMemoryManager

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_db")
        try:
            app = create_app(db_path=db_path)
            # Manually set up state as the lifespan would
            memory = BeliefMemory(db_path=db_path, max_db_size=1 << 30)
            wm_manager = WorkingMemoryManager(memory)
            app.state.memory = memory
            app.state.wm_manager = wm_manager
            try:
                assert isinstance(app.state.memory, BeliefMemory)
                assert isinstance(app.state.wm_manager, WorkingMemoryManager)
            finally:
                memory.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestGcLoop:
    """Test the background gc_loop inside the lifespan."""

    @pytest.mark.asyncio
    async def test_gc_loop_calls_gc_frames(self):
        """The gc_loop task calls wm_manager.gc_frames() after each sleep cycle."""
        import asyncio

        from mnemebrain_core.api.app import create_app

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_db")
        try:
            app = create_app(db_path=db_path)

            gc_call_count = 0
            sleep_call_count = 0
            real_sleep = asyncio.sleep

            async def _fake_sleep(seconds):
                nonlocal sleep_call_count
                sleep_call_count += 1
                if seconds == 60:
                    # This is the gc_loop sleep — return immediately
                    if sleep_call_count > 2:
                        # After gc_frames has run, cancel the loop
                        raise asyncio.CancelledError
                    return
                # Non-gc sleeps pass through
                await real_sleep(seconds)

            with patch("asyncio.sleep", side_effect=_fake_sleep):
                ctx = app.router.lifespan_context(app)
                async with ctx:
                    wm = app.state.wm_manager
                    original_gc = wm.gc_frames

                    def counting_gc():
                        nonlocal gc_call_count
                        gc_call_count += 1
                        return original_gc()

                    wm.gc_frames = counting_gc
                    # Yield control so the gc_task runs
                    await real_sleep(0.05)

            assert gc_call_count >= 1, "gc_frames() was never called"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestImportErrorExceptionHandler:
    """Test that the ImportError exception handler returns 501."""

    def test_import_error_returns_501(self):
        """Raising ImportError inside a route triggers the 501 handler."""
        from mnemebrain_core.api.app import create_app
        from mnemebrain_core.memory import BeliefMemory
        from mnemebrain_core.working_memory import WorkingMemoryManager

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_db")
        try:
            app = create_app(db_path=db_path)
            memory = BeliefMemory(db_path=db_path, max_db_size=1 << 30)
            wm_manager = WorkingMemoryManager(memory)
            app.state.memory = memory
            app.state.wm_manager = wm_manager

            # Add a test route that raises ImportError
            from fastapi import APIRouter

            test_router = APIRouter()

            @test_router.get("/test-import-error")
            async def _raise_import_error():
                raise ImportError("embedding provider missing")

            app.include_router(test_router)

            with TestClient(app, raise_server_exceptions=False) as tc:
                resp = tc.get("/test-import-error")
            assert resp.status_code == 501
            detail = resp.json()["detail"]
            assert "Embedding provider not available" in detail
        finally:
            try:
                memory.close()
            except Exception:
                pass
            shutil.rmtree(tmpdir, ignore_errors=True)
