"""CLI entry point — run the MnemeBrain API server."""

import os
import sys


def main():
    try:
        import uvicorn
    except ImportError:
        print(
            "API dependencies not installed. Run: uv pip install -e '../mnemebrain-lite[api]'"
        )
        sys.exit(1)

    from mnemebrain_core.api.app import create_app

    db_path = sys.argv[1] if len(sys.argv) > 1 else "./mnemebrain_data"
    host = os.environ.get("MNEMEBRAIN_HOST", "127.0.0.1")
    port = int(os.environ.get("MNEMEBRAIN_PORT", "8000"))
    app = create_app(db_path=db_path)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    main()
