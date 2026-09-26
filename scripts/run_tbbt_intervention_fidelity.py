from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bdmtf.revision.tbbt_intervention_fidelity import (  # noqa: E402
    freeze_tbbt_predictions,
    score_tbbt_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze and score TBBT BDMTF intervention predictions"
    )
    parser.add_argument(
        "stage",
        choices=("freeze", "score", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "tbbt_intervention_fidelity.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result: dict[str, object] = {}
    if args.stage in {"freeze", "all"}:
        result["freeze"] = freeze_tbbt_predictions(ROOT, config)
    if args.stage in {"score", "all"}:
        result["score"] = score_tbbt_predictions(ROOT, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
