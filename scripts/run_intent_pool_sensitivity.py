from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.revision.intent_pool_sensitivity import run_intent_pool_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen-intent pool exhaustion")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "intent_pool_sensitivity.json",
    )
    parser.add_argument("--posts-per-community", type=int)
    parser.add_argument("--capacity-multipliers", nargs="+", type=int)
    parser.add_argument("--communities", nargs="+")
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--output-dir", type=str)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.posts_per_community is not None:
        config["posts_per_community"] = args.posts_per_community
    if args.capacity_multipliers is not None:
        config["capacity_multipliers"] = args.capacity_multipliers
    if args.communities is not None:
        config["communities"] = args.communities
    if args.bootstrap_samples is not None:
        config["bootstrap_samples"] = args.bootstrap_samples
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    result = run_intent_pool_sensitivity(ROOT, config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
