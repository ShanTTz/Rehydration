from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.platform_adapter_gain import write_adapter_gain


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the frozen platform-adaptation success gate."
    )
    parser.add_argument(
        "--fidelity",
        default=str(
            ROOT
            / "artifacts"
            / "external_validation"
            / "platform_adapter"
            / "evaluation"
            / "fidelity_distances.csv"
        ),
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
    args = parser.parse_args()
    result = write_adapter_gain(args.fidelity, args.config, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
