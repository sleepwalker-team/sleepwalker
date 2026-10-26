import argparse
from pathlib import Path

import uvicorn

from sleepwalker.server.app import create_app, create_local_app

from .app import create_ui_app


def main():
    parser = argparse.ArgumentParser(description="Run Sleepwalker API and browser UI in one process.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--models", type=Path, help="Directory of existing packaged models.")
    parser.add_argument("--inference-python", type=Path, help="Python executable of the compatible inference environment.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--storage", type=Path, help="Dedicated temporary directory for this instance.")
    args = parser.parse_args()
    if args.models and not args.inference_python:
        parser.error("--models requires --inference-python")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    api = (
        create_local_app(args.models, inference_python=args.inference_python, device=args.device, batch_size=args.batch_size)
        if args.models else create_app()
    )
    if args.storage:
        # Configure the existing shared storage object; every API service retains it.
        storage = api.state.temporary_storage
        storage.root = args.storage.expanduser().absolute()
        storage.root.mkdir(parents=True, exist_ok=True)
    uvicorn.run(create_ui_app(api), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
