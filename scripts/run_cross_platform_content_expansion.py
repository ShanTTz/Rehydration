from __future__ import annotations

import argparse
import json
from pathlib import Path

from bdmtf.revision.cross_platform_content_expansion import (
    run_cross_platform_content_expansion,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand content-matched Hacker News, Lemmy, and Voat evidence"
    )
    parser.add_argument(
        "--config",
        default=str(
            ROOT / "configs" / "cross_platform_content_expansion.json"
        ),
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_cross_platform_content_expansion(ROOT, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
