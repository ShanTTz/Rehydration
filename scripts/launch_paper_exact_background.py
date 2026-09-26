from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _clean_environment() -> dict[str, str]:
    clean: dict[str, str] = {}
    names: dict[str, str] = {}
    for name, value in os.environ.items():
        folded = name.casefold()
        previous = names.get(folded)
        if previous is not None:
            clean.pop(previous, None)
        names[folded] = name
        clean[name] = value
    return clean


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    log_dir = root / "artifacts" / "background_runs" / "paper_exact_full"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"
    command = [
        sys.executable,
        str(root / "scripts" / "reproduce_paper_resumable.py"),
        "--config",
        str(root / "configs" / "paper_exact_reproduction.json"),
        "--output",
        str(root / "run_outputs" / "paper_exact_full"),
    ]
    creation_flags = 0
    if os.name == "nt":
        creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=_clean_environment(),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            creationflags=creation_flags,
        )
    time.sleep(5)
    if process.poll() is not None:
        raise RuntimeError(
            f"Paper reproduction worker exited during startup with code {process.returncode}. "
            f"See {stderr_path}."
        )
    manifest = {
        "status": "running",
        "pid": process.pid,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "stdout": str(stdout_path.relative_to(root)),
        "stderr": str(stderr_path.relative_to(root)),
        "output": "run_outputs/paper_exact_full",
    }
    (log_dir / "launcher_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(process.pid)


if __name__ == "__main__":
    main()
