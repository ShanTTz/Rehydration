from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.reviewer_validation import write_factorial_outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze paired Core x Ranking reviewer-confirmatory runs."
    )
    parser.add_argument(
        "--records",
        default=str(ROOT / "run_outputs" / "reviewer_confirmatory_factorial" / "runs.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT
            / "run_outputs"
            / "reviewer_confirmatory_factorial"
            / "reviewer_analysis"
        ),
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "reviewer_factorial.json"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=30371)
    args = parser.parse_args()

    raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cells = raw.get("analysis", {}).get("factorial_cells")
    manifest = write_factorial_outputs(
        args.records,
        args.output,
        cells=cells,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
