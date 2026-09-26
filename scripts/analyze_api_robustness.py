from __future__ import annotations

import json
from pathlib import Path

from bdmtf.revision.api_robustness_analysis import (
    analyze_api_robustness,
)


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    print(
        json.dumps(
            analyze_api_robustness(ROOT),
            ensure_ascii=False,
            indent=2,
        )
    )
