"""Tests for __main__.py CLI entry point."""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestMain:
    @patch("uvicorn.run")
    @patch("mnemebrain_core.api.app.create_app")
    def test_main_default_db_path(self, mock_create_app, mock_uvicorn_run):
        """main() uses default db path and localhost when no args/env."""
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app
        with (
            patch.object(sys, "argv", ["mnemebrain"]),
            patch.dict("os.environ", {}, clear=True),
        ):
            from mnemebrain_core.__main__ import main

            main()
        mock_create_app.assert_called_once_with(db_path="./mnemebrain_data")
        mock_uvicorn_run.assert_called_once_with(mock_app, host="127.0.0.1", port=8000)

    @patch("uvicorn.run")
    @patch("mnemebrain_core.api.app.create_app")
    def test_main_custom_db_path(self, mock_create_app, mock_uvicorn_run):
        """main() uses argv[1] as db path when provided."""
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app
        with (
            patch.object(sys, "argv", ["mnemebrain", "/custom/path"]),
            patch.dict("os.environ", {}, clear=True),
        ):
            from mnemebrain_core.__main__ import main

            main()
        mock_create_app.assert_called_once_with(db_path="/custom/path")

    def test_main_uvicorn_import_error(self):
        """When uvicorn is not installed, main() prints a message and calls sys.exit(1)."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "uvicorn":
                raise ImportError("No module named 'uvicorn'")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            patch.object(sys, "argv", ["mnemebrain"]),
            patch.dict("os.environ", {}, clear=True),
        ):
            from mnemebrain_core.__main__ import main

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch("uvicorn.run")
    @patch("mnemebrain_core.api.app.create_app")
    def test_main_env_vars_host_and_port(self, mock_create_app, mock_uvicorn_run):
        """MNEMEBRAIN_HOST and MNEMEBRAIN_PORT env vars override defaults."""
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app
        with (
            patch.object(sys, "argv", ["mnemebrain"]),
            patch.dict(
                "os.environ",
                {"MNEMEBRAIN_HOST": "0.0.0.0", "MNEMEBRAIN_PORT": "9090"},
                clear=True,
            ),
        ):
            from mnemebrain_core.__main__ import main

            main()
        mock_uvicorn_run.assert_called_once_with(mock_app, host="0.0.0.0", port=9090)
