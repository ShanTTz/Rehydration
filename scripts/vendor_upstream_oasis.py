from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "vendor" / "oasis" / "upstream"
DEFAULT_MANIFEST = ROOT / "vendor" / "oasis" / "upstream_manifest.json"
IGNORED_NAMES = {
    ".git",
    ".github",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "dist",
    "build",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Vendor an upstream camel-ai/oasis checkout into this repository.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-dir", help="Existing local OASIS checkout.")
    source.add_argument("--source-zip", help="Existing OASIS source archive.")
    source.add_argument("--github-ref", help="Download camel-ai/oasis archive for a branch, tag, or commit.")
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument("--replace", action="store_true", help="Replace an existing target directory inside this repo.")
    args = parser.parse_args()

    target = Path(args.target).resolve()
    _assert_inside_repo(target)
    if target.exists():
        if not args.replace:
            raise SystemExit(f"target already exists: {target}. Pass --replace to refresh it.")
        shutil.rmtree(target)

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        if args.source_dir:
            source_root = Path(args.source_dir).resolve()
        elif args.source_zip:
            source_root = _extract_zip(Path(args.source_zip).resolve(), tmp)
        else:
            archive = tmp / "oasis.zip"
            ref = str(args.github_ref)
            url = f"https://github.com/camel-ai/oasis/archive/{ref}.zip"
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, archive)
            source_root = _extract_zip(archive, tmp)

        if not source_root.exists():
            raise SystemExit(f"source does not exist: {source_root}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_root, target, ignore=_ignore)

    manifest = _build_manifest(target, args)
    DEFAULT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Vendored OASIS source to {target}")
    print(f"Manifest: {DEFAULT_MANIFEST}")


def _assert_inside_repo(path: Path) -> None:
    root = ROOT.resolve()
    if not path.is_relative_to(root):
        raise SystemExit(f"Refusing to write outside repository: {path}")


def _extract_zip(path: Path, tmp: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"zip archive not found: {path}")
    extract_dir = tmp / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(extract_dir)
    children = [child for child in extract_dir.iterdir() if child.is_dir()]
    if len(children) == 1:
        return children[0]
    return extract_dir


def _ignore(_directory: str, names: Iterable[str]) -> set[str]:
    ignored = set()
    for name in names:
        if name in IGNORED_NAMES or name.endswith(".egg-info"):
            ignored.add(name)
    return ignored


def _build_manifest(target: Path, args: argparse.Namespace) -> Dict[str, object]:
    file_count, total_bytes = _count_files(target)
    source_type = "source_dir" if args.source_dir else "source_zip" if args.source_zip else "github_ref"
    source_value = args.source_dir or args.source_zip or args.github_ref
    top_level = sorted(path.name for path in target.iterdir())[:50]
    return {
        "vendored_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_type": source_type,
        "source": str(source_value),
        "target": str(target),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "top_level_entries": top_level,
    }


def _count_files(root: Path) -> Tuple[int, int]:
    count = 0
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            count += 1
            total += path.stat().st_size
    return count, total


if __name__ == "__main__":
    main()
