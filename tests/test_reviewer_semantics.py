from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bdmtf.reviewer_semantics import (
    _annotation_prompt,
    _build_tasks,
    _stratified_select,
    build_semantic_consensus,
    fit_semantic_scorer,
    summarize_semantic_model_agreement,
)
from bdmtf.revision.api_intents import _validate_reviewer_semantic_response


def _candidate_frame() -> pd.DataFrame:
    rows = []
    for index in range(30):
        rows.append(
            {
                "community": "science",
                "post_id": f"p{index // 3}",
                "comment_id": f"c{index}",
                "split": "train" if index % 2 else "validation",
                "minutes_since_post": float(index),
                "depth_bucket": ("root_reply", "depth_1_2", "depth_3_plus")[
                    index % 3
                ],
                "lexicon_positive": index < 4,
                "text": f"comment {index}",
            }
        )
    return pd.DataFrame(rows)


def test_stratified_selection_is_deterministic_and_unique() -> None:
    frame = _candidate_frame()
    first = _stratified_select(frame, 15, seed=17, positive_target_share=0.4)
    second = _stratified_select(frame, 15, seed=17, positive_target_share=0.4)
    assert first["comment_id"].tolist() == second["comment_id"].tolist()
    assert first["comment_id"].is_unique
    assert len(first) == 15
    assert int(first["lexicon_positive"].sum()) == 4


def test_tasks_keep_audit_and_calibration_separate() -> None:
    calibration = _stratified_select(
        _candidate_frame(), 10, seed=3, positive_target_share=0.3
    )
    calibration["role"] = "calibration"
    audit = _stratified_select(
        _candidate_frame().assign(
            comment_id=lambda frame: "a_" + frame["comment_id"],
            split="test",
        ),
        5,
        seed=4,
        positive_target_share=0.3,
    )
    audit["role"] = "audit"
    tasks = _build_tasks(
        pd.concat([calibration, audit], ignore_index=True),
        {
            "antagonism": "0..1",
            "conflict_amplifying": "0..1",
            "constructive_disagreement": "boolean",
            "counterspeech": "boolean",
            "instruction_like_text": "boolean",
            "confidence": "0..1",
        },
        batch_size=4,
    )
    assert {task["role"] for task in tasks} == {"calibration", "audit"}
    assert sum(task["comment_count"] for task in tasks) == 15
    assert all("later" not in task["prompt"].lower() for task in tasks)


def test_annotation_prompt_requires_independent_axes_and_json() -> None:
    prompt = _annotation_prompt(
        [{"comment_id": "c1", "text": "You are wrong because the evidence changed."}],
        {
            "antagonism": "0..1",
            "conflict_amplifying": "0..1",
            "constructive_disagreement": "boolean",
            "counterspeech": "boolean",
            "instruction_like_text": "boolean",
            "confidence": "0..1",
        },
    )
    assert "axes as independent" in prompt
    assert "constructive_disagreement" in prompt
    assert "untrusted quoted data" in prompt
    assert "instruction_like_text" in prompt
    encoded = prompt.split("Comments: ", 1)[1]
    assert json.loads(encoded)[0]["comment_id"] == "c1"


def test_semantic_response_validation_requires_complete_ordered_schema() -> None:
    task = {"item_ids": ["c1"]}
    response = {
        "items": [
            {
                "comment_id": "c1",
                "antagonism": 0.2,
                "conflict_amplifying": 0.3,
                "constructive_disagreement": True,
                "counterspeech": False,
                "instruction_like_text": False,
                "confidence": 0.9,
            }
        ]
    }
    _validate_reviewer_semantic_response(json.dumps(response), task)


def test_consensus_and_scorer_keep_audit_out_of_fit(tmp_path: Path) -> None:
    rows = []
    for index in range(36):
        role = "audit" if index >= 30 else "calibration"
        split = "test" if role == "audit" else ("validation" if index >= 20 else "train")
        conflict = index % 2 == 1
        rows.append(
            {
                "community": "science",
                "post_id": f"p{index}",
                "comment_id": f"c{index}",
                "split": split,
                "minutes_since_post": 10.0,
                "depth_bucket": "root_reply",
                "lexicon_positive": conflict,
                "text": (
                    f"attack argue hostile repeated token {index % 3}"
                    if conflict
                    else f"calm evidence constructive repeated token {index % 3}"
                ),
                "role": role,
            }
        )
    selected = pd.DataFrame(rows)
    selected_path = tmp_path / "selected.parquet"
    selected.to_parquet(selected_path, index=False)
    cache_path = tmp_path / "cache.jsonl"
    with cache_path.open("w", encoding="utf-8") as handle:
        for family in ("openai", "deepseek", "qwen"):
            for row in rows:
                conflict = "attack" in row["text"]
                response = {
                    "items": [
                        {
                            "comment_id": row["comment_id"],
                            "antagonism": 0.8 if conflict else 0.1,
                            "conflict_amplifying": 0.9 if conflict else 0.1,
                            "constructive_disagreement": not conflict,
                            "counterspeech": False,
                            "instruction_like_text": False,
                            "confidence": 0.9,
                        }
                    ]
                }
                handle.write(
                    json.dumps(
                        {
                            "task_id": f"{family}_{row['comment_id']}",
                            "family": family,
                            "model": f"{family}-model",
                            "response": json.dumps(response),
                        }
                    )
                    + "\n"
                )
    consensus, manifest = build_semantic_consensus(
        cache_path, selected_path, required_families=3
    )
    assert manifest["status"] == "complete"
    assert len(consensus) == 36
    model_path = tmp_path / "scorer.joblib"
    scorer = fit_semantic_scorer(consensus, model_path)
    assert scorer["status"] == "complete"
    assert scorer["n_audit_labels"] == 6
    import joblib

    bundle = joblib.load(model_path)
    assert bundle["test_comment_ids_used_for_fit"] == []
    assert not set(bundle["fit_comment_ids"]) & {
        f"c{index}" for index in range(30, 36)
    }
    agreement, agreement_manifest = summarize_semantic_model_agreement(cache_path)
    assert agreement_manifest["unique_comments"] == 36
    conflict = agreement[agreement["label"] == "conflict_amplifying"].iloc[0]
    assert conflict["krippendorff_alpha_interval"] == 1.0
