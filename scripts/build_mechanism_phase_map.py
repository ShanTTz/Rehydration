from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.mechanism_phase_map import build_mechanism_phase_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Separate confirmatory local uncertainty from the global stress "
            "boundary map and render mechanism phase figures."
        )
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "mechanism_phase_map.json"),
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT / "artifacts" / "reviewer_validation" / "mechanism_phase_map"
        ),
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = build_mechanism_phase_map(ROOT, config, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
