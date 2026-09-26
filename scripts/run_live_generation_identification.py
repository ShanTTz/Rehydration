from __future__ import annotations

import argparse
import json
from pathlib import Path

from bdmtf.revision.live_generation_identification import run_identification_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the live-generation versus frozen-intent identification diagnostic."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/live_generation_identification.json"),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Call the configured OpenAI-compatible API for uncached tasks.",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    result = run_identification_experiment(repo_root, config_path, execute=args.execute)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
