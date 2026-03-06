"""Tests for __main__.py CLI entry point."""
from unittest.mock import patch, MagicMock

from mnemebrain_core.__main__ import main


class TestMain:
    @patch("mnemebrain.__main__.uvicorn")
    @patch("mnemebrain.__main__.create_app")
    def test_main_default_db_path(self, mock_create_app, mock_uvicorn):
        """main() uses default db path when no args."""
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app
        with patch("mnemebrain.__main__.sys") as mock_sys:
            mock_sys.argv = ["mnemebrain"]
            main()
        mock_create_app.assert_called_once_with(db_path="./mnemebrain_data")
        mock_uvicorn.run.assert_called_once_with(mock_app, host="0.0.0.0", port=8000)

    @patch("mnemebrain.__main__.uvicorn")
    @patch("mnemebrain.__main__.create_app")
    def test_main_custom_db_path(self, mock_create_app, mock_uvicorn):
        """main() uses argv[1] as db path when provided."""
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app
        with patch("mnemebrain.__main__.sys") as mock_sys:
            mock_sys.argv = ["mnemebrain", "/custom/path"]
            main()
        mock_create_app.assert_called_once_with(db_path="/custom/path")
