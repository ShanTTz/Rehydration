from __future__ import annotations

import argparse
import json
from pathlib import Path

from bdmtf.revision.tbbt_qualified_controls import (
    run_tbbt_qualified_control_analysis,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TBBT analysis with independent qualified donor controls"
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "tbbt_qualified_controls.json"),
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_tbbt_qualified_control_analysis(ROOT, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
