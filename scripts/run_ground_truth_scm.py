from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.revision.ground_truth_scm import run_ground_truth_scm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare counterfactual estimators against analytic truth"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "ground_truth_scm.json",
    )
    parser.add_argument("--replications", type=int)
    parser.add_argument("--output-dir", type=str)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.replications is not None:
        config["replications"] = args.replications
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    manifest = run_ground_truth_scm(ROOT, config, args.config)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

