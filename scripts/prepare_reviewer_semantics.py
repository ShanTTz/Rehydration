from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.reviewer_semantics import (
    execute_reviewer_semantic_tasks,
    prepare_reviewer_semantic_tasks,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze or execute reviewer-facing semantic calibration tasks."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "reviewer_semantic_calibration.json"),
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT / "artifacts" / "reviewer_validation" / "semantic_calibration"
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute frozen tasks through the configured OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--pilot-batches",
        type=int,
        default=None,
        help="Execute only the first N frozen batches; completed calls remain cached.",
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    tasks_path = output / "semantic_annotation_tasks.jsonl"
    if args.execute:
        result = execute_reviewer_semantic_tasks(
            tasks_path,
            config,
            output,
            pilot_batches=args.pilot_batches,
        )
    else:
        splits = pd.read_csv(ROOT / "artifacts" / "splits" / "post_splits.csv")
        result = prepare_reviewer_semantic_tasks(
            ROOT / "data" / "social_paper",
            splits,
            config,
            output,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
