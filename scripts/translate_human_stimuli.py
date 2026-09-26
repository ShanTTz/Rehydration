from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "experiment_app" / "thread_fixture.json"
DEFAULT_CACHE = (
    ROOT
    / "artifacts"
    / "human_rct"
    / "stimulus_bank"
    / "zh_cn_translation_cache.jsonl"
)
DEFAULT_MANIFEST = (
    ROOT
    / "artifacts"
    / "human_rct"
    / "stimulus_bank"
    / "zh_cn_translation_manifest.json"
)
CHINESE = re.compile(r"[\u3400-\u9fff]")
PROMPT_VERSION = "human-stimulus-zh-cn-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate participant-visible human-study stimuli to Chinese."
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--batch-items", type=int, default=24)
    parser.add_argument("--batch-chars", type=int, default=5200)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--apply-only", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_id(text: str) -> str:
    return sha256_text(text)[:20]


def fixture_texts(fixture: dict[str, Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for thread in fixture["threads"]:
        candidates = [str(thread["title"])]
        neutral = thread["variants"]["neutral"]["comments"]
        conflict = thread["variants"]["conflict"]["comments"]
        if len(neutral) != len(conflict):
            raise ValueError("Neutral and conflict variants differ in length")
        for left, right in zip(neutral, conflict, strict=True):
            if str(left["comment_id"]) != str(right["comment_id"]):
                raise ValueError("Variant comment IDs do not align")
            if str(left["text"]) != str(right["text"]):
                raise ValueError("Variant comment text already differs")
            candidates.append(str(left["text"]))
        for text in candidates:
            if text not in seen:
                ordered.append(text)
                seen.add(text)
    return ordered


def structural_projection(fixture: dict[str, Any]) -> dict[str, Any]:
    projection = json.loads(json.dumps(fixture, ensure_ascii=False))
    projection.pop("translation", None)
    for thread in projection["threads"]:
        thread["title"] = "<translated-title>"
        for context in ("neutral", "conflict"):
            for comment in thread["variants"][context]["comments"]:
                comment["text"] = "<translated-comment>"
        source = thread.get("source", {})
        source.pop("language", None)
        source.pop("display_language", None)
        source.pop("translation_model", None)
    return projection


def structural_sha256(fixture: dict[str, Any]) -> str:
    return sha256_text(canonical_json(structural_projection(fixture)))


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        source = str(record["source"])
        if str(record["source_sha256"]) != sha256_text(source):
            raise ValueError(f"Translation cache source hash fails on line {number}")
        translation = str(record["translation"]).strip()
        if not translation:
            raise ValueError(f"Blank translation on line {number}")
        records[source] = record
    return records


def batches(
    texts: Iterable[str],
    *,
    max_items: int,
    max_chars: int,
) -> list[list[str]]:
    output: list[list[str]] = []
    current: list[str] = []
    characters = 0
    for text in texts:
        if current and (
            len(current) >= max_items
            or characters + len(text) > max_chars
        ):
            output.append(current)
            current = []
            characters = 0
        current.append(text)
        characters += len(text)
    if current:
        output.append(current)
    return output


def translation_prompt(texts: list[str]) -> str:
    items = [{"id": text_id(text), "source": text} for text in texts]
    return (
        "Translate every Reddit title or comment below into natural Simplified "
        "Chinese for a behavioral study. Preserve the original meaning, stance, "
        "tone, humor, uncertainty, and level of disagreement. Do not soften, "
        "summarize, censor, explain, or add facts. Keep proper nouns recognizable. "
        "Preserve placeholders such as [link removed] and u/[anonymous], numbers, "
        "and emojis. Return strict JSON only: {\"translations\":[{\"id\":"
        "\"...\",\"translation\":\"...\"}]}. Preserve every id exactly and "
        "return one non-empty translation per input item.\n\nItems:\n"
        + json.dumps(items, ensure_ascii=False)
    )


def parse_response(content: str, expected: list[str]) -> dict[str, str]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    payload = json.loads(cleaned)
    rows = payload.get("translations", [])
    parsed = {
        str(row["id"]): str(row["translation"]).strip()
        for row in rows
        if isinstance(row, dict)
        and row.get("id")
        and str(row.get("translation", "")).strip()
    }
    expected_ids = {text_id(text) for text in expected}
    if set(parsed) != expected_ids:
        missing = sorted(expected_ids - set(parsed))
        extra = sorted(set(parsed) - expected_ids)
        raise ValueError(
            f"Translation response IDs mismatch; missing={missing}, extra={extra}"
        )
    return parsed


def request_batch(
    base_url: str,
    key: str,
    model: str,
    texts: list[str],
    max_retries: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    prompt = translation_prompt(texts)
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    if "qwen" in model.casefold():
        payload["enable_thinking"] = False
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 2):
        try:
            response = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            raw = response.json()
            content = str(raw["choices"][0]["message"]["content"])
            parsed = parse_response(content, texts)
            return parsed, {
                "attempts": attempt,
                "returned_model": str(raw.get("model", model)),
                "prompt_sha256": sha256_text(prompt),
                "response_sha256": sha256_text(content),
                "usage": raw.get("usage", {}),
            }
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            requests.RequestException,
        ) as error:
            last_error = error
            if attempt <= max_retries:
                time.sleep(min(12.0, 1.8**attempt))
    raise RuntimeError(f"Translation batch failed: {last_error}")


def append_records(
    path: Path,
    texts: list[str],
    translations: dict[str, str],
    model: str,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for text in texts:
            translation = translations[text_id(text)]
            record = {
                "text_id": text_id(text),
                "source": text,
                "source_sha256": sha256_text(text),
                "translation": translation,
                "translation_sha256": sha256_text(translation),
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": metadata["prompt_sha256"],
                "response_sha256": metadata["response_sha256"],
                "returned_model": metadata["returned_model"],
                "attempts": metadata["attempts"],
                "usage": metadata.get("usage", {}),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def apply_translations(
    fixture: dict[str, Any],
    cache: dict[str, dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    required = fixture_texts(fixture)
    missing = [text for text in required if text not in cache]
    if missing:
        raise ValueError(f"Translation cache is incomplete: {len(missing)} missing")
    before = structural_sha256(fixture)
    for thread in fixture["threads"]:
        original_title = str(thread["title"])
        thread["title"] = str(cache[original_title]["translation"])
        thread["source"]["language"] = "en"
        thread["source"]["display_language"] = "zh-CN"
        thread["source"]["translation_model"] = model
        neutral = thread["variants"]["neutral"]["comments"]
        conflict = thread["variants"]["conflict"]["comments"]
        for left, right in zip(neutral, conflict, strict=True):
            original = str(left["text"])
            translated = str(cache[original]["translation"])
            left["text"] = translated
            right["text"] = translated
    fixture["translation"] = {
        "display_language": "zh-CN",
        "source_language": "en",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "translated_text_count": len(required),
    }
    after = structural_sha256(fixture)
    if before != after:
        raise ValueError("Non-text stimulus structure changed during translation")
    return fixture


def write_manifest(
    path: Path,
    fixture_path: Path,
    cache_path: Path,
    fixture: dict[str, Any],
    source_count: int,
    model: str,
) -> dict[str, Any]:
    translations = load_cache(cache_path)
    required = fixture_texts(fixture)
    chinese_present = sum(
        bool(CHINESE.search(str(translations[text]["translation"])))
        for text in required
    )
    manifest = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "source_text_count": source_count,
        "translated_text_count": len(required),
        "translations_with_chinese_characters": chinese_present,
        "fixture_path": str(fixture_path),
        "fixture_sha256": sha256_file(fixture_path),
        "structural_sha256": structural_sha256(fixture),
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "key_stored": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    texts = fixture_texts(fixture)
    cache = load_cache(args.cache)
    missing = [text for text in texts if text not in cache]
    if missing and not args.apply_only:
        key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get(
            "OPENAI_BASE_URL",
            "https://api.openai.com/v1",
        ).strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required; it is never stored")
        work = batches(
            missing,
            max_items=args.batch_items,
            max_chars=args.batch_chars,
        )
        if args.max_batches:
            work = work[: args.max_batches]
        for index, batch in enumerate(work, start=1):
            translated, metadata = request_batch(
                base_url,
                key,
                args.model,
                batch,
                args.max_retries,
            )
            append_records(
                args.cache,
                batch,
                translated,
                args.model,
                metadata,
            )
            print(
                f"translated batch {index}/{len(work)}; "
                f"cached {len(cache) + sum(len(item) for item in work[:index])}"
            )
        cache = load_cache(args.cache)
    remaining = [text for text in texts if text not in cache]
    if remaining:
        partial = {
            "status": "partial",
            "source_text_count": len(texts),
            "translated_text_count": len(texts) - len(remaining),
            "remaining_text_count": len(remaining),
            "cache_path": str(args.cache),
            "key_stored": False,
        }
        print(json.dumps(partial, ensure_ascii=False, indent=2))
        return
    translated_fixture = apply_translations(fixture, cache, args.model)
    args.fixture.write_text(
        json.dumps(translated_fixture, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = write_manifest(
        args.manifest,
        args.fixture,
        args.cache,
        translated_fixture,
        len(texts),
        args.model,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
