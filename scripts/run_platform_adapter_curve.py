from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.platform_adapter_curve import run_platform_adapter_curve_workflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Hacker News platform-adapter data-budget curve."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "platform_adapter_curve.json"),
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT
            / "artifacts"
            / "external_validation"
            / "platform_adapter_curve"
        ),
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest = run_platform_adapter_curve_workflow(
        ROOT, config, args.output
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
