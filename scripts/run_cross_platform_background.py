from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "artifacts" / "background_runs"
STATUS_PATH = RUN_DIR / "cross_platform.status.json"
DLL_HANDLES: list[object] = []


def write_status(payload: dict[str, object]) -> None:
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(STATUS_PATH)


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))
    os.environ["PYTHONPATH"] = str(ROOT / "src")
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        python_root = Path(sys.executable).resolve().parent
        for directory in (python_root / "DLLs", python_root / "Library" / "bin"):
            if directory.is_dir():
                DLL_HANDLES.append(os.add_dll_directory(str(directory)))
    started_at = datetime.now(timezone.utc).isoformat()
    command = [
        "run-cross-platform",
        "--root",
        str(ROOT),
        "--config",
        str(ROOT / "configs" / "external_sources.json"),
        "--seeds",
        "0",
        "--max-cascades",
        "5000",
        "--bootstrap-samples",
        "400",
    ]
    with (
        (RUN_DIR / "cross_platform.stdout.log").open("w", encoding="utf-8") as stdout,
        (RUN_DIR / "cross_platform.stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        sys.stdout = stdout
        sys.stderr = stderr
        status: dict[str, object] = {
            "status": "running",
            "pid": os.getpid(),
            "started_at": started_at,
            "command": command,
        }
        write_status(status)
        try:
            from bdmtf.cli import main as cli_main

            sys.argv = ["bdmtf", *command]
            cli_main()
            status.update(
                {
                    "status": "complete",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "exit_code": 0,
                }
            )
        except BaseException as exc:
            traceback.print_exc()
            status.update(
                {
                    "status": "failed",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "exit_code": 1,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            write_status(status)


if __name__ == "__main__":
    main()
