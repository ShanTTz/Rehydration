from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.revision.cross_platform_budget_audit import (
    run_cross_platform_budget_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit equal target-data budgets")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "cross_platform_budget_audit.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run_cross_platform_budget_audit(ROOT, config), indent=2))


if __name__ == "__main__":
    main()

