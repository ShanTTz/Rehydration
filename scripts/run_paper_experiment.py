from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.experiments import config_from_dict, interventions_from_config, run_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BDMTF paper-style experiments")
    parser.add_argument("--social-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default_experiment.json"))
    parser.add_argument("--communities", nargs="+", default=None)
    parser.add_argument("--posts-per-community", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    config = config_from_dict(raw)
    interventions = interventions_from_config(raw)
    communities = args.communities or raw.get("communities", ["funny"])
    posts_per_community = args.posts_per_community or int(raw.get("posts_per_community", 20))
    seeds = args.seeds or raw.get("seeds", [0, 1, 2])

    records = run_batch(
        social_root=args.social_root,
        output_dir=args.output,
        communities=communities,
        posts_per_community=posts_per_community,
        seeds=seeds,
        config=config,
        interventions=interventions,
    )
    print(f"Completed {len(records)} BDMTF runs.")
    print(f"Outputs are in {args.output}")


if __name__ == "__main__":
    main()
