"""Tests for __main__.py CLI entry point."""

from unittest.mock import patch, MagicMock
import sys


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
