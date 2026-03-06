"""CLI entry point — run the MnemeBrain API server."""

import sys


def main():
    try:
        import uvicorn
    except ImportError:
        print("API dependencies not installed. Run: pip install mnemebrain-lite[api]")
        sys.exit(1)

    from mnemebrain_core.api.app import create_app

    db_path = sys.argv[1] if len(sys.argv) > 1 else "./mnemebrain_data"
    app = create_app(db_path=db_path)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()
