from __future__ import annotations

import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.revision.workflows import (  # noqa: E402
    lemmy_intervention_fidelity_workflow,
    load_config,
)


RUN_DIR = ROOT / "artifacts" / "background_runs" / "lemmy_intervention_fidelity"
STATUS_PATH = RUN_DIR / "pipeline_status.json"


def _status(stage: str, status: str, **details: object) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(
        json.dumps(
            {
                "stage": stage,
                "status": status,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                **details,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    config = load_config(ROOT / "configs" / "lemmy_intervention_fidelity.json")
    try:
        _status("prediction_fidelity", "running")
        result = lemmy_intervention_fidelity_workflow(ROOT, config)
        _status(
            "prediction_fidelity",
            "complete",
            n_pairs=result["n_pairs"],
            test_pairs=result["test_pairs"],
            framework_support=result["framework_support"],
        )
        _status("paper_ready_table", "running")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "update_lemmy_intervention_fidelity_table.py"),
            ],
            cwd=ROOT,
            check=True,
        )
        _status(
            "pipeline",
            "complete",
            n_pairs=result["n_pairs"],
            test_pairs=result["test_pairs"],
            bdmtf_primary_mae=result["bdmtf_primary_mae"],
            bdmtf_direction_accuracy=result["bdmtf_direction_accuracy"],
            framework_support=result["framework_support"],
            paper_ready_tables_updated=True,
        )
    except Exception as exc:
        _status(
            "pipeline",
            "failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
