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
    expand_reddit_workflow,
    load_config,
    reddit_expansion_validation_workflow,
)


RUN_DIR = ROOT / "artifacts" / "background_runs" / "reddit_expansion"
STATUS_PATH = RUN_DIR / "pipeline_status.json"


def _write_status(stage: str, status: str, **details: object) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    STATUS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    config = load_config(ROOT / "configs" / "external_sources.json")
    try:
        _write_status("import", "running")
        expansion = expand_reddit_workflow(ROOT, config)
        _write_status(
            "import",
            "complete",
            n_communities=expansion.get("n_communities", 0),
            n_cascades=expansion.get("n_cascades", 0),
            target_met=expansion.get("target_met", False),
        )
        if not expansion.get("target_met"):
            raise RuntimeError("Expanded Reddit data did not meet the frozen target")

        _write_status("five_model_validation", "running")
        validation = reddit_expansion_validation_workflow(ROOT, config)
        _write_status(
            "five_model_validation",
            "complete",
            n_communities=validation.get("n_communities", 0),
            n_cascades=validation.get("n_cascades", 0),
            protocols=list(validation.get("protocols", {})),
        )

        _write_status("paper_ready_tables", "running")
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_paper_ready_optimization_tables.py")],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "update_reddit_expansion_result_table.py")],
            cwd=ROOT,
            check=True,
        )
        _write_status(
            "pipeline",
            "complete",
            n_communities=validation.get("n_communities", 0),
            n_cascades=validation.get("n_cascades", 0),
            paper_ready_tables_updated=True,
        )
    except Exception as exc:
        _write_status(
            "pipeline",
            "failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
