"""Tests for uncovered route branches."""
import pytest

from mnemebrain_core.api.routes import get_memory, set_memory


class TestGetMemoryUninitialized:
    def test_raises_when_not_initialized(self):
        """get_memory raises RuntimeError when _memory is None (line 33)."""
        set_memory(None)
        with pytest.raises(RuntimeError, match="not initialized"):
            get_memory()
