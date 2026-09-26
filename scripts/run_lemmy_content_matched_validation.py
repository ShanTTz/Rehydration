from __future__ import annotations

import argparse
import json
from pathlib import Path

from bdmtf.revision.lemmy_content_matched_validation import (
    run_lemmy_content_matched_validation,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Lemmy/Hacker News content-matched validation"
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "lemmy_content_matched_validation.json"),
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_lemmy_content_matched_validation(ROOT, config, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
