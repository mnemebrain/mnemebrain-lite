"""Tests for mnemebrain_core.__init__ — covers version detection branches."""

import importlib
import sys
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch


class TestVersion:
    def test_version_is_string(self):
        """__version__ must be a non-empty string."""
        import mnemebrain_core

        assert isinstance(mnemebrain_core.__version__, str)
        assert len(mnemebrain_core.__version__) > 0

    def test_version_fallback_on_package_not_found(self):
        """When importlib.metadata.version raises, __version__ falls back to the hardcoded string."""
        # Remove the cached module so the module body re-executes.
        sys.modules.pop("mnemebrain_core", None)

        with patch(
            "importlib.metadata.version",
            side_effect=PackageNotFoundError("mnemebrain-lite"),
        ):
            import mnemebrain_core as mc

            assert mc.__version__ == "0.1.0a3"

        # Restore the real module for subsequent tests.
        sys.modules.pop("mnemebrain_core", None)
        importlib.import_module("mnemebrain_core")
