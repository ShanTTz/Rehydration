from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.schema import Intent


SYSTEM_PROMPT = (
    "You generate offline semantic intents for a controlled social-media simulation. "
    "Return JSON only. Do not include popularity, ranking, likes, or live feedback signals."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen intents with an OpenAI-compatible API.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--intents-per-agent", type=int, default=2)
    parser.add_argument("--topic", default="a contentious online discussion")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required in the environment.")

    intents = []
    for agent_id in range(args.agents):
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Topic: {args.topic}\n"
                        f"Agent id: {agent_id}\n"
                        f"Generate {args.intents_per_agent} supportive and {args.intents_per_agent} antagonistic "
                        "short comment intents. Return exactly this JSON shape: "
                        '{"supportive":["..."],"antagonistic":["..."]}.'
                    ),
                },
            ],
            "temperature": 0.4,
            "max_tokens": 700,
            "response_format": {"type": "json_object"},
        }
        raw = _post_json(f"{base_url}/chat/completions", api_key, payload)
        content = raw["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        for polarity in ["supportive", "antagonistic"]:
            for idx, text in enumerate(parsed.get(polarity, [])[: args.intents_per_agent]):
                intents.append(
                    Intent(
                        intent_id=f"api_a{agent_id}_{polarity}_{idx}",
                        agent_id=agent_id,
                        polarity=polarity,
                        content=str(text),
                        metadata={"source": "api_offline_generation", "model": args.model},
                    )
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for intent in intents:
            handle.write(json.dumps(intent.__dict__, ensure_ascii=False) + "\n")
    print(f"Generated {len(intents)} frozen intents at {output}")


def _post_json(url: str, api_key: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API HTTP {exc.code}: {body[:500]}") from exc


if __name__ == "__main__":
    main()
