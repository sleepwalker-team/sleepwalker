import argparse
from pathlib import Path

import uvicorn

from .app import create_app, create_local_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local Sleepwalker API server."
    )
    parser.add_argument(
        "--models",
        type=Path,
        help="Directory containing .swmodel packages.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used for inference, for example cpu or cuda.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--host", default="127.0.0.1", help="Server bind address.")
    parser.add_argument("--port", type=int, default=6006, help="Server port.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = (
        create_local_app(args.models, device=args.device, batch_size=args.batch_size)
        if args.models is not None
        else create_app()
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
