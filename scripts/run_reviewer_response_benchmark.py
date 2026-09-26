from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.reviewer_response_benchmark import write_response_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark held-out factorial intervention-response prediction."
    )
    parser.add_argument(
        "--contrasts",
        default=str(
            ROOT
            / "run_outputs"
            / "reviewer_confirmatory_factorial"
            / "reviewer_analysis"
            / "factorial_contrasts.csv"
        ),
    )
    parser.add_argument(
        "--splits",
        default=str(ROOT / "artifacts" / "splits" / "post_splits.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT
            / "artifacts"
            / "reviewer_validation"
            / "response_benchmark"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=30371)
    args = parser.parse_args()
    result = write_response_benchmark(
        args.contrasts,
        args.splits,
        args.output,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
