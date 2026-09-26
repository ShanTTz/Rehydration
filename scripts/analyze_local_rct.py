from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.revision.local_rct_analysis import analyze_local_rct


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen analysis for merged local RCT results."
    )
    parser.add_argument(
        "--config",
        default="configs/human_rct.json",
    )
    parser.add_argument(
        "--merged",
        default="artifacts/human_rct/raw/merged",
    )
    parser.add_argument(
        "--output",
        default="artifacts/human_rct/analysis",
    )
    args = parser.parse_args()
    payload = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    protocol = payload.get("human_rct", payload)
    merged = ROOT / args.merged
    result = analyze_local_rct(
        merged / "participant_analysis.csv",
        merged / "trial_analysis.csv",
        protocol,
        ROOT / args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
