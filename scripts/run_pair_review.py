from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the local FastAPI pair review tool."
    )
    parser.add_argument(
        "--pairs",
        required=True,
        help="Path to pair_candidates.csv.",
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to annotations.csv. The file will be created if missing.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host for the FastAPI server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port for the FastAPI server.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn reload for local UI iteration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "uvicorn is not installed. Install review extras first: "
            "pip install -e .[review]"
        ) from exc

    from pair_review_app import create_app

    pair_candidates_path = Path(args.pairs).resolve()
    annotations_path = Path(args.annotations).resolve()
    app = create_app(pair_candidates_path, annotations_path)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
