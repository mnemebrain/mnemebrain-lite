"""Tests for uncovered route branches."""

import pytest
from unittest.mock import MagicMock

from starlette.datastructures import State

from mnemebrain_core.api.routes import get_memory, get_wm_manager


class TestGetMemoryUninitialized:
    def test_raises_when_not_initialized(self):
        """get_memory raises RuntimeError when memory is not on app.state."""
        request = MagicMock()
        request.app.state = State()
        with pytest.raises(RuntimeError, match="not initialized"):
            get_memory(request)


class TestGetWmManagerUninitialized:
    def test_raises_when_not_initialized(self):
        """get_wm_manager raises RuntimeError when wm_manager is not on app.state."""
        request = MagicMock()
        request.app.state = State()
        with pytest.raises(RuntimeError, match="not initialized"):
            get_wm_manager(request)
