from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the OpenAI-compatible API environment for intent regeneration.")
    parser.add_argument("--env-file", default="", help="Optional .env file to load before checking variables.")
    parser.add_argument("--base-url", default="", help="Override OPENAI_BASE_URL.")
    parser.add_argument("--model", default="", help="Override OPENAI_MODEL.")
    parser.add_argument("--live", action="store_true", help="Send one tiny chat-completions request.")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    if args.env_file:
        _load_env_file(Path(args.env_file))

    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = (args.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
    model = args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    print(f"OPENAI_API_KEY: {'set' if api_key else 'missing'}")
    print(f"OPENAI_BASE_URL: {base_url}")
    print(f"OPENAI_MODEL: {model}")

    if not api_key:
        print("Status: incomplete. Set OPENAI_API_KEY before regenerating API-backed intents.")
        return 1

    if not args.live:
        print("Status: configured. Live API call skipped; pass --live to test network and credentials.")
        return 0

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "Return {\"ok\": true}."},
        ],
        "temperature": 0,
        "max_tokens": 20,
        "response_format": {"type": "json_object"},
    }
    try:
        response = _post_json(f"{base_url}/chat/completions", api_key, payload, timeout=args.timeout)
    except Exception as exc:
        print(f"Status: live check failed: {type(exc).__name__}: {exc}")
        return 2
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"Status: live check ok. Response content: {content[:120]}")
    return 0


def _load_env_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"env file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _post_json(url: str, api_key: str, payload: Dict[str, object], timeout: int) -> Dict[str, object]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API HTTP {exc.code}: {body[:500]}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
