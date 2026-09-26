from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.experiments import config_from_dict, interventions_from_config, run_batch
from bdmtf.paper_outputs import write_paper_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete local paper reproduction pipeline.")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "social_paper"))
    parser.add_argument("--output", default=str(ROOT / "run_outputs" / "paper_reproduction"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper_reproduction.json"))
    parser.add_argument("--quick", action="store_true", help="Small validation run: 2 posts, seed 0 only.")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if args.quick:
        raw["posts_per_community"] = 2
        raw["seeds"] = [0]

    config = config_from_dict(raw)
    interventions = interventions_from_config(raw)
    records = run_batch(
        social_root=args.data_root,
        output_dir=args.output,
        communities=raw["communities"],
        posts_per_community=int(raw["posts_per_community"]),
        seeds=raw["seeds"],
        config=config,
        interventions=interventions,
    )
    write_paper_outputs(records, Path(args.output))
    print(f"Paper reproduction complete: {len(records)} runs")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
