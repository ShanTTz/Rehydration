from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.revision.local_rct import (
    create_invite_codes,
    invite_records,
    validate_protocol,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the offline Windows human-RCT participant package."
    )
    parser.add_argument(
        "--config",
        default="configs/human_rct.json",
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--output",
        default="dist/local_human_rct",
    )
    parser.add_argument(
        "--skip-exe",
        action="store_true",
        help="Prepare source assets without invoking PyInstaller.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    config = dict(source.get("human_rct", source))
    mode = str(config.get("mode", "demo"))
    enrollment_mode = str(config.get("enrollment_mode", "invite"))
    validate_protocol(config, allow_demo=mode == "demo")
    if enrollment_mode == "open":
        codes: list[str] = []
        config.pop("invite_records", None)
    else:
        prefix = "DEMO" if mode == "demo" else "RCT"
        codes = (
            [f"DEMO-{index:04d}" for index in range(1, args.count + 1)]
            if mode == "demo"
            else create_invite_codes(args.count, prefix=prefix)
        )
        config["invite_records"] = invite_records(
            codes,
            study_id=str(config["study_id"]),
        )

    output = ROOT / args.output
    if output.exists():
        shutil.rmtree(output)
    participant_root = output / "participant"
    researcher_root = output / "researcher"
    participant_root.mkdir(parents=True)
    researcher_root.mkdir(parents=True)

    (participant_root / "study_package.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    invite_path: Path | None = None
    if codes:
        invite_path = researcher_root / "invite_codes.csv"
        with invite_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["invite_code", "assignment_index"],
            )
            writer.writeheader()
            writer.writerows(
                {
                    "invite_code": code,
                    "assignment_index": index,
                }
                for index, code in enumerate(codes)
            )

    if not args.skip_exe:
        command = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--windowed",
            "--onedir",
            "--name",
            "BDMTF_Thread_Study",
            "--paths",
            str(ROOT / "src"),
            "--add-data",
            f"{ROOT / 'experiment_app'};experiment_app",
            "--distpath",
            str(output / "_pyinstaller"),
            "--workpath",
            str(output / "_build"),
            "--specpath",
            str(output / "_spec"),
            str(ROOT / "scripts" / "local_rct_launcher.py"),
        ]
        if sys.platform == "win32":
            binary_root = Path(sys.prefix) / "Library" / "bin"
            required_dlls = (
                "sqlite3.dll",
                "libcrypto-3-x64.dll",
                "libssl-3-x64.dll",
                "libmpdec-4.dll",
                "liblzma.dll",
                "LIBBZ2.dll",
                "ffi.dll",
            )
            for name in required_dlls:
                candidate = binary_root / name
                if candidate.is_file():
                    command[-1:-1] = [
                        "--add-binary",
                        f"{candidate};.",
                    ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise SystemExit(
                "PyInstaller failed. Install it with "
                "`python -m pip install pyinstaller` and rerun."
            )
        built = output / "_pyinstaller" / "BDMTF_Thread_Study"
        for item in built.iterdir():
            destination = participant_root / item.name
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)
        shutil.rmtree(output / "_pyinstaller")
        shutil.rmtree(output / "_build")
        shutil.rmtree(output / "_spec")

    readme = f"""BDMTF 本地讨论实验

1. 解压整个 participant 文件夹。
2. 双击 BDMTF_Thread_Study.exe。
3. 阅读说明并完成知情同意后直接开始。
4. 完成任务后下载 JSON，并将该文件发回研究者。

模式：{mode}
方案版本：{config.get('protocol_version')}
参与方式：{'直接参加，无需招募码' if enrollment_mode == 'open' else '一次性招募码'}

不要只复制 exe；_internal 和 study_package.json 必须位于同一目录。
"""
    (participant_root / "使用说明.txt").write_text(
        readme,
        encoding="utf-8-sig",
    )
    archive_path = Path(
        shutil.make_archive(
            str(output / "BDMTF_Thread_Study_Windows"),
            "zip",
            participant_root,
        )
    )
    manifest = {
        "status": "complete",
        "mode": mode,
        "enrollment_mode": enrollment_mode,
        "participant_codes": len(codes),
        "participant_zip": str(archive_path),
        "participant_zip_sha256": sha256_file(archive_path),
        "researcher_invites": str(invite_path) if invite_path else None,
        "standalone_executable": not args.skip_exe,
    }
    (output / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
