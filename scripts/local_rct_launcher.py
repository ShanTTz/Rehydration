from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from bdmtf.revision.local_rct import serve_human_experiment


def _message(title: str, body: str) -> None:
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, body, title, 0x10)
    except Exception:
        print(f"{title}: {body}", file=sys.stderr)


def main() -> None:
    frozen = bool(getattr(sys, "frozen", False))
    executable_dir = (
        Path(sys.executable).resolve().parent
        if frozen
        else Path(__file__).resolve().parents[1]
    )
    asset_root = Path(
        getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1])
    )
    config_path = executable_dir / "study_package.json"
    if not config_path.is_file():
        _message(
            "研究包不完整",
            "找不到 study_package.json，请解压整个文件夹后再运行。",
        )
        raise SystemExit(2)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        data_dir = executable_dir / "study_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        serve_human_experiment(
            executable_dir,
            config,
            host="127.0.0.1",
            port=0,
            database=data_dir / "human_rct.sqlite",
            app_root=asset_root / "experiment_app",
            open_browser=True,
        )
    except Exception as exc:
        try:
            error_dir = executable_dir / "study_data"
            error_dir.mkdir(parents=True, exist_ok=True)
            (error_dir / "startup_error.txt").write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )
        except Exception:
            pass
        _message("研究程序无法启动", str(exc))
        raise


if __name__ == "__main__":
    main()
