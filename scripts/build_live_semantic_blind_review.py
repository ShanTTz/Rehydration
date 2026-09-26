"""Build a three-arm blinded review package for dynamic-parent semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts" / "reviewer_validation" / "live_semantic_ceiling"
DEFAULT_OUTPUT = DEFAULT_INPUT / "blind_review_package"
SOURCES = (
    "frozen_frame_renderer",
    "parent_aware_intent_renderer",
    "dynamic_parent_live_generation",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build(
    input_dir: Path,
    output_dir: Path,
    *,
    ratings_per_item: int = 3,
    items_per_assignment: int = 20,
    seed: int = 30371,
) -> dict[str, object]:
    tasks_path = input_dir / "live_semantic_tasks.jsonl"
    responses_path = input_dir / "live_semantic_responses.jsonl"
    tasks = {str(row["task_id"]): row for row in _read_jsonl(tasks_path)}
    responses = {str(row["task_id"]): row for row in _read_jsonl(responses_path)}
    missing = sorted(set(tasks).difference(responses))
    if missing:
        raise ValueError(f"missing {len(missing)} API responses")
    if len(tasks) % items_per_assignment:
        raise ValueError("items must divide evenly into each rating wave")

    rng = np.random.default_rng(seed)
    item_ids = sorted(tasks)
    assignment_slots: list[list[str]] = []
    for _ in range(ratings_per_item):
        wave = item_ids.copy()
        rng.shuffle(wave)
        assignment_slots.extend(
            wave[start : start + items_per_assignment]
            for start in range(0, len(wave), items_per_assignment)
        )
    assignment_count = len(assignment_slots)

    public_dir = output_dir / "blinded_assignments"
    private_dir = output_dir / "researcher_only"
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    master_rows: list[dict[str, object]] = []
    assignment_hashes: dict[str, str] = {}

    for assignment_index in range(assignment_count):
        assignment_id = f"SEM-{assignment_index + 1:03d}"
        rows = []
        assignment_items = assignment_slots[assignment_index]
        for display_index, task_id in enumerate(assignment_items, start=1):
            task = tasks[str(task_id)]
            response = responses[str(task_id)]
            candidates = {
                "frozen_frame_renderer": str(task["frame_reply"]),
                "parent_aware_intent_renderer": str(response["intent_preserving_reply"]),
                "dynamic_parent_live_generation": str(response["live_reply"]),
            }
            order = list(SOURCES)
            rng.shuffle(order)
            labels = dict(zip(("A", "B", "C"), order, strict=True))
            public = {
                "assignment_id": assignment_id,
                "item_number": display_index,
                "parent_comment": str(task["parent_text"]),
                "frozen_intent_anchor": str(task["frame_reply"]),
            }
            private = {
                "assignment_id": assignment_id,
                "item_number": display_index,
                "task_id": task_id,
                "item_id": task["item_id"],
                "community": task["community"],
                "condition": task["condition"],
                "depth_bin": task["depth_bin"],
                "polarity": task["polarity"],
            }
            for label in ("A", "B", "C"):
                source = labels[label]
                public[f"candidate_{label}"] = candidates[source]
                public[f"relevance_{label}_1_to_5"] = ""
                public[f"coherence_{label}_1_to_5"] = ""
                public[f"intent_preservation_{label}_1_to_5"] = ""
                public[f"obvious_mismatch_{label}_0_or_1"] = ""
                private[f"candidate_{label}_source"] = source
            public["best_direct_response_A_B_C_or_tie"] = ""
            public["reviewer_note_optional"] = ""
            rows.append(public)
            master_rows.append({**private, **public})

        assignment_path = public_dir / f"{assignment_id}.csv"
        pd.DataFrame(rows).to_csv(assignment_path, index=False, encoding="utf-8-sig")
        assignment_hashes[assignment_id] = _sha256(assignment_path)

    master = pd.DataFrame(master_rows)
    counts = master.groupby("task_id")["assignment_id"].nunique()
    if not counts.eq(ratings_per_item).all():
        raise AssertionError("each item must receive the prespecified independent assignments")
    master_path = private_dir / "RESEARCHER_MASTER_DO_NOT_SHARE.csv"
    master.to_csv(master_path, index=False, encoding="utf-8-sig")

    instructions = """# Semantic Compatibility Blind Review

For each item, read the parent comment and frozen intent anchor, then score all
three anonymous candidates. Relevance and coherence measure direct-response
quality. Intent preservation measures whether the candidate retains the
anchor's stance and communicative goal. Mark obvious mismatch as 1 only for a
clear contradiction, broken reference, or reply to a different topic. Do not
infer which system produced a candidate. Leave an item blank if the content is
unsafe or impossible to judge.
"""
    (public_dir / "INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    manifest = {
        "status": "ready_for_blind_human_rating",
        "items": len(tasks),
        "ratings_per_item": ratings_per_item,
        "items_per_assignment": items_per_assignment,
        "assignments": assignment_count,
        "candidate_sources": list(SOURCES),
        "randomization_seed": seed,
        "tasks_sha256": _sha256(tasks_path),
        "responses_sha256": _sha256(responses_path),
        "researcher_master_sha256": _sha256(master_path),
        "assignment_hashes": assignment_hashes,
        "direct_participant_identifiers_requested": False,
    }
    (output_dir / "blind_review_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ratings-per-item", type=int, default=3)
    parser.add_argument("--items-per-assignment", type=int, default=20)
    parser.add_argument("--seed", type=int, default=30371)
    args = parser.parse_args()
    result = build(
        args.input_dir,
        args.output_dir,
        ratings_per_item=args.ratings_per_item,
        items_per_assignment=args.items_per_assignment,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
