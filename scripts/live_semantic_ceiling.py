"""Prepare and execute a dynamic-parent semantic ceiling experiment.

The API stage creates two strong comparators for each frozen-frame item:
an intent-preserving parent-aware rewrite and an unconstrained live reply.
Secrets are read only from the environment, and completed tasks are resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MASTER = (
    ROOT
    / "artifacts"
    / "reviewer_validation"
    / "semantic_parent_audit"
    / "human_annotation_package_20260825"
    / "researcher_only"
    / "RESEARCHER_MASTER_DO_NOT_SHARE.csv"
)
OUTPUT = ROOT / "artifacts" / "reviewer_validation" / "live_semantic_ceiling"
SYSTEM_PROMPT = """You produce comparison replies for a blinded semantic-validity study.
Return JSON only. Do not mention the experiment, ratings, policies, or candidate labels.
The intent-preserving reply must keep the supplied reply's stance and communicative intent
while making it a natural direct response to the supplied parent. The live reply should be
the best natural direct response to the parent with the requested polarity, without being
constrained to preserve the supplied intent. Do not add unverifiable factual claims."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sample(frame: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["_priority"] = frame["item_id"].map(
        lambda item: _sha256_text(f"{seed}::{item}")
    )
    strata = ["community", "condition", "depth_bin"]
    groups = list(frame.groupby(strata, sort=True))
    quota = max(1, int(np.ceil(sample_size / len(groups))))
    selected = []
    for _, group in groups:
        selected.append(group.sort_values("_priority").head(quota))
    result = pd.concat(selected, ignore_index=True).sort_values("_priority").head(sample_size)
    if len(result) < sample_size:
        remaining = frame[~frame["item_id"].isin(result["item_id"])]
        result = pd.concat(
            [result, remaining.sort_values("_priority").head(sample_size - len(result))],
            ignore_index=True,
        )
    return result.drop(columns="_priority").reset_index(drop=True)


def prepare(sample_size: int, seed: int, model: str) -> dict[str, object]:
    source = pd.read_csv(MASTER)
    sample = _stable_sample(source, sample_size, seed)
    tasks = []
    for row in sample.itertuples(index=False):
        frame_reply = row.candidate_a if row.candidate_a_source == "frame_rendered" else row.candidate_b
        user_prompt = (
            f"Parent comment:\n{row.parent_text}\n\n"
            f"Frozen reply intent anchor:\n{frame_reply}\n\n"
            f"Requested polarity: {row.polarity}\n\n"
            "Return exactly: "
            '{"intent_preserving_reply":"...","live_reply":"..."}'
        )
        task_id = _sha256_text(f"live-ceiling::{row.item_id}::{model}")[:24]
        tasks.append(
            {
                "task_id": task_id,
                "item_id": row.item_id,
                "community": row.community,
                "condition": row.condition,
                "depth_bin": row.depth_bin,
                "polarity": row.polarity,
                "parent_text": row.parent_text,
                "frame_reply": frame_reply,
                "model": model,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "prompt_sha256": _sha256_text(SYSTEM_PROMPT + "\n" + user_prompt),
            }
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    task_path = OUTPUT / "live_semantic_tasks.jsonl"
    with task_path.open("w", encoding="utf-8") as stream:
        for task in tasks:
            stream.write(json.dumps(task, ensure_ascii=False) + "\n")
    manifest = {
        "status": "tasks_ready",
        "tasks": len(tasks),
        "sample_seed": seed,
        "model": model,
        "strata": ["community", "condition", "depth_bin"],
        "source": str(MASTER),
        "source_sha256": _sha256_file(MASTER),
        "tasks_sha256": _sha256_file(task_path),
        "api_key_stored": False,
    }
    (OUTPUT / "task_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _post_json(url: str, api_key: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API HTTP {exc.code}: {body[:500]}") from exc


def execute(max_tasks: int | None, retries: int) -> dict[str, object]:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
    if not api_key or not base_url:
        raise SystemExit("OPENAI_API_KEY and OPENAI_BASE_URL are required in the environment")
    tasks = [json.loads(line) for line in (OUTPUT / "live_semantic_tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    response_path = OUTPUT / "live_semantic_responses.jsonl"
    completed: dict[str, dict[str, object]] = {}
    if response_path.exists():
        for line in response_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                completed[str(record["task_id"])] = record
    pending = [task for task in tasks if task["task_id"] not in completed]
    if max_tasks is not None:
        pending = pending[:max_tasks]
    with response_path.open("a", encoding="utf-8") as stream:
        for task in pending:
            payload = {
                "model": task["model"],
                "messages": [
                    {"role": "system", "content": task["system_prompt"]},
                    {"role": "user", "content": task["user_prompt"]},
                ],
                "temperature": 0.2,
                "max_tokens": 260,
                "response_format": {"type": "json_object"},
            }
            error = None
            for attempt in range(1, retries + 2):
                started = time.perf_counter()
                try:
                    raw = _post_json(f"{base_url}/chat/completions", api_key, payload)
                    content = str(raw["choices"][0]["message"]["content"])
                    parsed = json.loads(content)
                    record = {
                        "task_id": task["task_id"],
                        "item_id": task["item_id"],
                        "model_requested": task["model"],
                        "model_returned": raw.get("model"),
                        "temperature": 0.2,
                        "attempts": attempt,
                        "latency_seconds": time.perf_counter() - started,
                        "prompt_sha256": task["prompt_sha256"],
                        "intent_preserving_reply": str(parsed["intent_preserving_reply"]),
                        "live_reply": str(parsed["live_reply"]),
                        "response_sha256": _sha256_text(content),
                        "usage": raw.get("usage", {}),
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    completed[str(task["task_id"])] = record
                    error = None
                    break
                except Exception as exc:  # noqa: BLE001 - retry boundary
                    error = str(exc)
                    if attempt <= retries:
                        time.sleep(min(8.0, 2.0**attempt))
            if error is not None:
                raise RuntimeError(f"task {task['task_id']} failed: {error}")
    usage = [record.get("usage", {}) for record in completed.values()]
    manifest = {
        "status": "complete" if len(completed) == len(tasks) else "partial",
        "tasks": len(tasks),
        "completed": len(completed),
        "model": tasks[0]["model"] if tasks else None,
        "prompt_tokens": int(sum(int(item.get("prompt_tokens", 0)) for item in usage)),
        "completion_tokens": int(sum(int(item.get("completion_tokens", 0)) for item in usage)),
        "response_sha256": _sha256_file(response_path) if response_path.exists() else None,
        "api_key_stored": False,
    }
    (OUTPUT / "execution_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--sample-size", type=int, default=200)
    prepare_parser.add_argument("--seed", type=int, default=30371)
    prepare_parser.add_argument("--model", default="gpt-4o-mini")
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--max-tasks", type=int)
    execute_parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.sample_size, args.seed, args.model)
    else:
        result = execute(args.max_tasks, args.retries)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
