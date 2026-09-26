from __future__ import annotations

import argparse
import json
from pathlib import Path

from bdmtf.revision.multithread_identification import (
    run_multithread_identification_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the held-out multi-thread live-generation versus frozen-intent "
            "identification diagnostic."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/multithread_identification.json"),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Call the configured OpenAI-compatible API for uncached tasks.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Lock the held-out thread sample without generating outcomes.",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    result = run_multithread_identification_experiment(
        repo_root,
        config_path,
        execute=args.execute,
        plan_only=args.plan_only,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
