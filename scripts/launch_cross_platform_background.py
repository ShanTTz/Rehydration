from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        [sys.executable, str(root / "scripts" / "run_cross_platform_background.py")],
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    time.sleep(10)
    if process.poll() is not None:
        raise RuntimeError(f"background worker exited during startup with code {process.returncode}")
    print(process.pid)


if __name__ == "__main__":
    main()
