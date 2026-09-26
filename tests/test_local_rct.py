from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bdmtf.revision.local_rct import (
    CORRECTIONS,
    complete_trial,
    enroll,
    export_participant_bundle,
    initialize_database,
    load_fixture,
    next_trial,
    record_event,
    record_intent,
    record_response,
    trial_plan_for_index,
)
from bdmtf.revision.local_rct_analysis import (
    analyze_local_rct,
    bundle_rows,
    merge_result_bundles,
    verify_bundle,
)


def _config() -> dict:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs" / "human_rct.json").read_text(encoding="utf-8")
    )["human_rct"]
    study_id = "test-study"
    config.update(
        {
            "study_id": study_id,
            "study_title": "Test",
            "protocol_version": "test-v3",
            "mode": "demo",
            "enrollment_mode": "open",
            "require_manual_stimulus_approval": False,
            "consent": {"version": "test-consent"},
        }
    )
    return config


def _complete_bundle(tmp_path: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    fixture = load_fixture(root / "experiment_app" / "thread_fixture.json")
    config = _config()
    database = tmp_path / "study.sqlite"
    initialize_database(database, config)
    session = enroll(
        database,
        config,
        fixture,
        "",
        consent=True,
        eligible=True,
        age_band="25-34",
        prior_platform_use=5,
        baseline_conflict_tolerance=4,
    )
    for trial_number in range(config["trials_per_participant"]):
        trial = next_trial(
            database,
            config,
            fixture,
            session.participant_id,
            session.session_token,
        )
        assert trial["trial_index"] == trial_number
        assert trial["phase"] == "intent"
        record_intent(
            database,
            config,
            session.participant_id,
            session.session_token,
            {
                "intent_uuid": f"intent-{trial_number}",
                "trial_uuid": trial["trial_uuid"],
                "intent_choice": "add_information",
                "intent_text": "Initial thought",
            },
        )
        trial = next_trial(
            database,
            config,
            fixture,
            session.participant_id,
            session.session_token,
        )
        assert trial["phase"] == "interaction"
        first = trial["thread"]["comments"][0]
        record_event(
            database,
            session.participant_id,
            session.session_token,
            {
                "event_uuid": f"event-{trial_number}",
                "event_type": "target_selected",
                "trial_uuid": trial["trial_uuid"],
                "thread_id": trial["thread"]["thread_id"],
                "content_id": first["comment_id"],
                "content_depth": first["depth"],
                "scroll_depth": 0.75,
            },
        )
        record_response(
            database,
            config,
            session.participant_id,
            session.session_token,
            {
                "response_uuid": f"response-{trial_number}",
                "trial_uuid": trial["trial_uuid"],
                "thread_id": trial["thread"]["thread_id"],
                "target_id": first["comment_id"],
                "target_depth": first["depth"],
                "response_text": "这是一条测试回复。",
                "skipped": False,
            },
        )
        complete_trial(
            database,
            session.participant_id,
            session.session_token,
            {
                "trial_uuid": trial["trial_uuid"],
                "survey_item_id": trial["survey_item"]["id"],
                "survey_response": 5,
            },
        )
    return export_participant_bundle(
        database,
        config,
        session.participant_id,
        session.session_token,
    )


def test_trial_plan_balances_all_corrections() -> None:
    plan = trial_plan_for_index(3, ["a", "b", "c"])
    assert len(plan) == 3
    assert {item["correction"] for item in plan} == set(CORRECTIONS)
    assert {item["thread_id"] for item in plan} == {"a", "b", "c"}


def test_complete_local_flow_exports_verified_bundle(
    tmp_path: Path,
) -> None:
    bundle = _complete_bundle(tmp_path)
    verify_bundle(bundle)
    participant, trials = bundle_rows(bundle)
    assert bundle["participant"]["enrollment_mode"] == "open"
    assert isinstance(bundle["participant"]["assignment_index"], int)
    assert participant["completed"] is True
    assert participant["reply_selected"] == 1.0
    assert len(trials) == 20
    assert {item["correction"] for item in trials} == set(CORRECTIONS)
    assert len({item["survey_item_id"] for item in trials}) == 20
    assert {item["stimulus_conflict_band"] for item in trials} == {
        "low",
        "medium",
        "high",
    }


def test_bundle_hash_rejects_tampering(tmp_path: Path) -> None:
    bundle = _complete_bundle(tmp_path)
    bundle["trials"][0]["survey_response"] = 1
    with pytest.raises(ValueError, match="hash"):
        verify_bundle(bundle)


def test_merge_rejects_duplicate_participant(tmp_path: Path) -> None:
    bundle = _complete_bundle(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    second.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        merge_result_bundles(
            [first, second],
            tmp_path / "merged",
        )


def test_response_rejects_another_participants_trial(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = load_fixture(root / "experiment_app" / "thread_fixture.json")
    config = _config()
    database = tmp_path / "study.sqlite"
    initialize_database(database, config)
    first = enroll(
        database,
        config,
        fixture,
        "",
        consent=True,
        eligible=True,
        age_band="25-34",
        prior_platform_use=5,
        baseline_conflict_tolerance=4,
    )
    second = enroll(
        database,
        config,
        fixture,
        "",
        consent=True,
        eligible=True,
        age_band="25-34",
        prior_platform_use=5,
        baseline_conflict_tolerance=4,
    )
    first_trial = next_trial(
        database,
        config,
        fixture,
        first.participant_id,
        first.session_token,
    )
    with pytest.raises(PermissionError, match="participant"):
        record_response(
            database,
            config,
            second.participant_id,
            second.session_token,
            {
                "response_uuid": "cross-participant",
                "trial_uuid": first_trial["trial_uuid"],
                "thread_id": first_trial["thread"]["thread_id"],
                "target_id": "comment",
                "target_depth": 1,
                "response_text": "test",
                "skipped": False,
            },
        )


def test_complete_rejects_unknown_trial(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = load_fixture(root / "experiment_app" / "thread_fixture.json")
    config = _config()
    database = tmp_path / "study.sqlite"
    initialize_database(database, config)
    session = enroll(
        database,
        config,
        fixture,
        "",
        consent=True,
        eligible=True,
        age_band="25-34",
        prior_platform_use=5,
        baseline_conflict_tolerance=4,
    )
    with pytest.raises(PermissionError, match="participant"):
        complete_trial(
            database,
            session.participant_id,
            session.session_token,
            {
                "trial_uuid": "unknown",
                "perceived_conflict": 4,
                "discussion_quality": 5,
                "return_intent": 6,
            },
        )


def test_complete_requires_reply_or_explicit_skip(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = load_fixture(root / "experiment_app" / "thread_fixture.json")
    config = _config()
    database = tmp_path / "study.sqlite"
    initialize_database(database, config)
    session = enroll(
        database,
        config,
        fixture,
        "",
        consent=True,
        eligible=True,
        age_band="25-34",
        prior_platform_use=5,
        baseline_conflict_tolerance=4,
    )
    trial = next_trial(
        database,
        config,
        fixture,
        session.participant_id,
        session.session_token,
    )
    with pytest.raises(ValueError, match="skip"):
        complete_trial(
            database,
            session.participant_id,
            session.session_token,
            {
                "trial_uuid": trial["trial_uuid"],
                "perceived_conflict": 4,
                "discussion_quality": 5,
                "return_intent": 6,
            },
        )


def test_confirmatory_gate_requires_frozen_sample_size(
    tmp_path: Path,
) -> None:
    rankings = ["new", "top", "hot", "best", "controversial"]
    contexts = ["neutral", "conflict"]
    participants = []
    trials = []
    index = 0
    for ranking in rankings:
        for context in contexts:
            for repeat in range(2):
                participant_id = f"participant-{index}"
                participants.append(
                    {
                        "participant_id": participant_id,
                        "arm": f"{ranking}_{context}",
                        "ranking": ranking,
                        "context": context,
                        "consent": True,
                        "eligible": True,
                        "completed": True,
                        "prior_platform_use": 3 + repeat,
                        "baseline_conflict_tolerance": 4 + repeat,
                        "scroll_depth": 0.6 + repeat * 0.1,
                        "reply_selected": float(repeat),
                        "target_depth": 1.0 + repeat,
                        "exit": 0,
                        "reparticipation": 1,
                    }
                )
                for trial_index, correction in enumerate(CORRECTIONS):
                    trials.append(
                        {
                            "participant_id": participant_id,
                            "correction": correction,
                            "trial_index": trial_index,
                            "context": context,
                            "community": "science",
                            "stimulus_conflict_band": "medium",
                            "survey_item_id": f"conflict_{trial_index}",
                            "survey_construct": "conflict",
                            "survey_response": (
                                5 if context == "conflict" else 3
                            ),
                            "mechanism_score": (
                                5 if context == "conflict" else 3
                            ),
                            "scroll_depth": 0.5 + trial_index * 0.1,
                            "reply_selected": int(trial_index != 0),
                            "target_depth": float(trial_index),
                            "exit": 0,
                            "reparticipation": 1,
                        }
                    )
                index += 1
    participant_path = tmp_path / "participants.csv"
    trial_path = tmp_path / "trials.csv"
    pd.DataFrame(participants).to_csv(participant_path, index=False)
    pd.DataFrame(trials).to_csv(trial_path, index=False)
    protocol = {
        "mode": "production",
        "ethics_approval_id": "approval",
        "preregistration_url": "https://example.org/prereg",
        "confirmatory_n": 1500,
        "ranking_arms": rankings,
        "contexts": contexts,
    }
    blocked = analyze_local_rct(
        participant_path,
        trial_path,
        protocol,
        tmp_path / "blocked",
    )
    assert blocked["confirmatory_arm_coverage_reached"] is True
    assert blocked["confirmatory_sample_size_reached"] is False
    assert blocked["claim_allowed"] is False

    protocol["confirmatory_n"] = 20
    allowed = analyze_local_rct(
        participant_path,
        trial_path,
        protocol,
        tmp_path / "allowed",
    )
    assert allowed["confirmatory_sample_size_reached"] is True
    assert allowed["claim_allowed"] is True
