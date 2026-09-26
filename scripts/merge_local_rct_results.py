from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.revision.local_rct_analysis import merge_result_bundles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and merge returned local RCT JSON files."
    )
    parser.add_argument("input", help="Directory containing participant JSON files")
    parser.add_argument(
        "--output",
        default="artifacts/human_rct/raw/merged",
    )
    args = parser.parse_args()
    paths = list(Path(args.input).glob("*.json"))
    result = merge_result_bundles(paths, ROOT / args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

