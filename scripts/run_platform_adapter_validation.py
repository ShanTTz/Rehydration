from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.platform_adapter_validation import run_platform_adapter_workflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a target-platform adapter on validation data and test it prospectively."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "platform_adapter_validation.json"),
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT
            / "artifacts"
            / "external_validation"
            / "platform_adapter"
        ),
    )
    parser.add_argument("--max-validation-cascades", type=int, default=0)
    parser.add_argument("--max-test-cascades", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.max_validation_cascades > 0:
        config["selection"]["max_validation_cascades"] = (
            args.max_validation_cascades
        )
    if args.max_test_cascades > 0:
        config["test"]["max_test_cascades"] = args.max_test_cascades
    if args.bootstrap_samples > 0:
        config["test"]["bootstrap_samples"] = args.bootstrap_samples
    result = run_platform_adapter_workflow(ROOT, config, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
