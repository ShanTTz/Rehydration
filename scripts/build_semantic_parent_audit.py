from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.revision.semantic_parent_audit import build_semantic_parent_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a blinded dynamic-parent semantic relevance audit"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "semantic_parent_audit.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = build_semantic_parent_audit(ROOT, config, args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

