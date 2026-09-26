from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from bdmtf.revision.api_intents import (
    _execute_tasks,
    build_annotation_tasks,
    build_intent_tasks,
    list_model_ids,
    select_models,
)
from bdmtf.revision.provenance import write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    key = os.environ.get("OPENAI_API_KEY", "")
    base_url = (
        os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        .strip()
        .rstrip("/")
    )
    if not key:
        raise SystemExit("OPENAI_API_KEY is required")

    config_path = ROOT / "configs" / "api_robustness.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selected = select_models(list_model_ids(base_url, key), config)
    intents = build_intent_tasks(
        ROOT / "data" / "social_paper",
        ROOT / "artifacts" / "splits" / "post_splits.csv",
        config,
    ).head(1)
    annotations = build_annotation_tasks(
        ROOT / "data" / "social_paper",
        config,
    ).head(1)

    output_dir = ROOT / "artifacts" / "api" / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    intent_result = _execute_tasks(
        intents,
        selected,
        config,
        base_url,
        key,
        output_dir / "intent_smoke.jsonl",
        "intent",
    )
    annotation_result = _execute_tasks(
        annotations,
        selected,
        config,
        base_url,
        key,
        output_dir / "annotation_smoke.jsonl",
        "annotation",
    )
    failures = intent_result["failed"] + annotation_result["failed"]
    manifest = {
        "status": "complete" if failures == 0 else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": selected,
        "base_host": base_url.split("//", 1)[-1].split("/", 1)[0],
        "key_stored": False,
        "config_sha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        "intent": intent_result,
        "annotation": annotation_result,
    }
    write_json(output_dir / "smoke_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
