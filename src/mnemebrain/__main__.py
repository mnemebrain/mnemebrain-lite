"""CLI entry point — run the MnemeBrain API server."""
import sys

import uvicorn

from mnemebrain.api.app import create_app


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else "./mnemebrain_data"
    app = create_app(db_path=db_path)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
