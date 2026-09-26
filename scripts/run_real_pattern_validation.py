from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.real_pattern_validation import run_real_pattern_validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate BDMTF directional mechanisms in held-out real Reddit threads."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "real_pattern_validation.json"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifacts" / "reviewer_validation" / "real_patterns"),
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_real_pattern_validation(ROOT, config, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
