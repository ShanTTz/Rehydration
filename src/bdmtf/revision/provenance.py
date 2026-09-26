from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path, excluded: Iterable[str] = ()) -> list[Path]:
    excluded_set = set(excluded)
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not any(part in excluded_set for part in path.parts)
    )


def tree_manifest(root: Path, excluded: Iterable[str] = ()) -> dict[str, Any]:
    records = []
    tree_digest = hashlib.sha256()
    for path in iter_files(root, excluded):
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        records.append({"path": relative, "size": size, "sha256": digest})
        tree_digest.update(f"{relative}|{digest}\n".encode("utf-8"))
    return {
        "root": str(root.resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(records),
        "total_bytes": sum(item["size"] for item in records),
        "tree_sha256": tree_digest.hexdigest(),
        "files": records,
    }


def run_manifest(command: str, config: dict[str, Any], data_hash: str = "") -> dict[str, Any]:
    safe_environment = {
        key: os.environ.get(key, "")
        for key in ("OPENAI_BASE_URL",)
        if os.environ.get(key)
    }
    return {
        "revision_version": "0.3.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "config": config,
        "data_tree_sha256": data_hash,
        "python": sys.version,
        "platform": platform.platform(),
        "environment": safe_environment,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
