from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from bdmtf.revision.local_rct import (
    SCHEMA_VERSION,
    canonical_json,
    sha256_text,
)
from bdmtf.revision.provenance import sha256_file, write_json


PRIMARY_OUTCOMES = (
    "scroll_depth",
    "reply_selected",
    "target_depth",
    "exit",
    "reparticipation",
)


def verify_bundle(bundle: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "study_id",
        "protocol_version",
        "participant",
        "design",
        "trials",
        "intents",
        "events",
        "responses",
        "payload_sha256",
    }
    missing = sorted(required.difference(bundle))
    if missing:
        raise ValueError("Bundle fields are missing: " + ", ".join(missing))
    expected = str(bundle["payload_sha256"])
    unsigned = dict(bundle)
    unsigned.pop("payload_sha256", None)
    actual = sha256_text(canonical_json(unsigned))
    if actual != expected:
        raise ValueError("Bundle payload hash does not match")
    if int(bundle["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("Unsupported local RCT bundle schema")
    participant = bundle["participant"]
    if not participant.get("participant_id"):
        raise ValueError("Participant ID is missing")
    if not participant.get("completed"):
        raise ValueError("Incomplete participant bundles cannot be merged")
    if participant.get("withdrawn"):
        raise ValueError("Withdrawn participant bundles cannot be merged")


def _trial_row(
    bundle: dict[str, Any],
    trial: dict[str, Any],
) -> dict[str, Any]:
    participant = bundle["participant"]
    trial_uuid = str(trial["trial_uuid"])
    events = [
        row
        for row in bundle["events"]
        if str(row.get("trial_uuid")) == trial_uuid
    ]
    responses = [
        row
        for row in bundle["responses"]
        if str(row.get("trial_uuid")) == trial_uuid
    ]
    intents = [
        row
        for row in bundle["intents"]
        if str(row.get("trial_uuid")) == trial_uuid
    ]
    scroll = [
        float(row["scroll_depth"])
        for row in events
        if row.get("scroll_depth") is not None
    ]
    submitted = [row for row in responses if not int(row.get("skipped", 0))]
    targets = [
        int(row["target_depth"])
        for row in submitted
        if row.get("target_depth") is not None
    ]
    intent = intents[0] if intents else {}
    reverse_by_id = {
        str(item.get("id", "")): bool(item.get("reverse_scored", False))
        for item in bundle.get("design", {}).get("mechanism_items", [])
    }
    survey_response = trial.get("survey_response")
    mechanism_score = None
    if survey_response is not None:
        mechanism_score = (
            8 - int(survey_response)
            if reverse_by_id.get(str(trial.get("survey_item_id", "")))
            else int(survey_response)
        )
    return {
        "participant_id": str(participant["participant_id"]),
        "trial_uuid": trial_uuid,
        "trial_index": int(trial["trial_index"]),
        "thread_id": str(trial["thread_id"]),
        "arm": str(participant["arm"]),
        "ranking": str(participant["ranking"]),
        "context": str(participant["context"]),
        "correction": str(trial["correction"]),
        "community": str(trial.get("community", "")),
        "stimulus_conflict_band": str(
            trial.get("stimulus_conflict_band", "")
        ),
        "survey_item_id": str(trial.get("survey_item_id", "")),
        "survey_construct": str(trial.get("survey_construct", "")),
        "survey_response": survey_response,
        "mechanism_score": mechanism_score,
        "intent_choice": str(intent.get("intent_choice", "")),
        "intent_length": int(intent.get("intent_length", 0)),
        "intent_activation": int(
            bool(intent) and intent.get("intent_choice") != "no_reply"
        ),
        "consent": bool(participant["consent"]),
        "eligible": bool(participant["eligible"]),
        "completed": bool(trial["completed_at"]),
        "age_band": str(participant["age_band"]),
        "prior_platform_use": int(participant["prior_platform_use"]),
        "baseline_conflict_tolerance": int(
            participant["baseline_conflict_tolerance"]
        ),
        "scroll_depth": max(scroll, default=0.0),
        "reply_selected": int(bool(submitted)),
        "target_depth": (
            float(np.mean(targets)) if targets else 0.0
        ),
        "exit": int(any(row["event_type"] == "exit" for row in events)),
        "reparticipation": int(
            any(row["event_type"] == "return" for row in events)
        ),
        "response_length": int(
            sum(int(row["response_length"]) for row in responses)
        ),
        "perceived_conflict": trial.get("perceived_conflict"),
        "discussion_quality": trial.get("discussion_quality"),
        "return_intent": trial.get("return_intent"),
    }


def bundle_rows(
    bundle: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    verify_bundle(bundle)
    trials = [_trial_row(bundle, trial) for trial in bundle["trials"]]
    expected = int(bundle.get("design", {}).get("expected_trial_count", 3))
    if len(trials) != expected:
        raise ValueError(
            f"Each completed bundle must contain {expected} trials"
        )
    first = trials[0]
    participant = {
        key: first[key]
        for key in (
            "participant_id",
            "arm",
            "ranking",
            "context",
            "consent",
            "eligible",
            "age_band",
            "prior_platform_use",
            "baseline_conflict_tolerance",
        )
    }
    participant.update(
        {
            "correction": "within_participant",
            "completed": all(row["completed"] for row in trials),
            "scroll_depth": float(
                np.mean([row["scroll_depth"] for row in trials])
            ),
            "reply_selected": float(
                np.mean([row["reply_selected"] for row in trials])
            ),
            "target_depth": float(
                np.mean([row["target_depth"] for row in trials])
            ),
            "exit": int(any(row["exit"] for row in trials)),
            "reparticipation": int(
                any(row["reparticipation"] for row in trials)
            ),
        }
    )
    return participant, trials


def merge_result_bundles(
    paths: Iterable[Path],
    output_dir: Path,
) -> dict[str, Any]:
    files = sorted({Path(path).resolve() for path in paths})
    if not files:
        raise ValueError("No participant JSON bundles were found")
    participants: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    study_id = ""
    protocol_version = ""
    seen: set[str] = set()
    modes: set[str] = set()
    for path in files:
        bundle = json.loads(path.read_text(encoding="utf-8"))
        verify_bundle(bundle)
        if not study_id:
            study_id = str(bundle["study_id"])
            protocol_version = str(bundle["protocol_version"])
        if str(bundle["study_id"]) != study_id:
            raise ValueError("Bundles contain more than one study ID")
        if str(bundle["protocol_version"]) != protocol_version:
            raise ValueError("Bundles contain more than one protocol version")
        participant, trial_rows = bundle_rows(bundle)
        participant_id = str(participant["participant_id"])
        if participant_id in seen:
            raise ValueError(f"Duplicate participant: {participant_id}")
        seen.add(participant_id)
        participants.append(participant)
        trials.extend(trial_rows)
        modes.add(str(bundle.get("mode", "unknown")))
        sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    participant_frame = pd.DataFrame(participants).sort_values(
        "participant_id"
    )
    trial_frame = pd.DataFrame(trials).sort_values(
        ["participant_id", "trial_index"]
    )
    participant_path = output_dir / "participant_analysis.csv"
    trial_path = output_dir / "trial_analysis.csv"
    participant_frame.to_csv(participant_path, index=False)
    trial_frame.to_csv(trial_path, index=False)
    with (output_dir / "verified_bundles.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in files:
            handle.write(path.read_text(encoding="utf-8").strip() + "\n")
    manifest = {
        "status": "complete",
        "study_id": study_id,
        "protocol_version": protocol_version,
        "modes": sorted(modes),
        "participants": int(len(participant_frame)),
        "trials": int(len(trial_frame)),
        "duplicate_participants": 0,
        "participant_analysis_sha256": sha256_file(participant_path),
        "trial_analysis_sha256": sha256_file(trial_path),
        "sources": sources,
    }
    write_json(output_dir / "merge_manifest.json", manifest)
    return manifest


def _holm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(values) - rank) * values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def _hc3_fit(
    frame: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    rankings = ["best", "controversial", "hot", "top"]
    design = pd.DataFrame({"intercept": 1.0}, index=frame.index)
    conflict = frame["context"].eq("conflict").astype(float)
    design["context_conflict"] = conflict
    for ranking in rankings:
        indicator = frame["ranking"].eq(ranking).astype(float)
        design[f"ranking_{ranking}"] = indicator
        design[f"ranking_{ranking}:context_conflict"] = (
            indicator * conflict
        )
    design["prior_platform_use"] = pd.to_numeric(
        frame["prior_platform_use"],
        errors="coerce",
    )
    design["baseline_conflict_tolerance"] = pd.to_numeric(
        frame["baseline_conflict_tolerance"],
        errors="coerce",
    )
    y = pd.to_numeric(frame[outcome], errors="coerce")
    valid = design.notna().all(axis=1) & y.notna()
    x = design.loc[valid].to_numpy(float)
    yv = y.loc[valid].to_numpy(float)
    if len(yv) <= x.shape[1]:
        return pd.DataFrame()
    bread = np.linalg.pinv(x.T @ x)
    beta = bread @ x.T @ yv
    residual = yv - x @ beta
    leverage = np.sum((x @ bread) * x, axis=1)
    corrected = residual / np.maximum(1e-6, 1.0 - leverage)
    meat = x.T @ (x * corrected[:, None] ** 2)
    variance = bread @ meat @ bread
    se = np.sqrt(np.maximum(0.0, np.diag(variance)))
    statistic = np.divide(
        beta,
        se,
        out=np.zeros_like(beta),
        where=se > 0,
    )
    degrees = max(1, len(yv) - x.shape[1])
    p_value = 2 * stats.t.sf(np.abs(statistic), degrees)
    return pd.DataFrame(
        {
            "outcome": outcome,
            "term": design.columns,
            "estimate": beta,
            "std_error_hc3": se,
            "ci_low": beta - stats.t.ppf(0.975, degrees) * se,
            "ci_high": beta + stats.t.ppf(0.975, degrees) * se,
            "p_value": p_value,
            "n": len(yv),
            "reference_ranking": "new",
            "reference_context": "neutral",
        }
    )


def _within_correction_fit(
    trials: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    prepared = trials.copy()
    prepared["_y"] = pd.to_numeric(prepared[outcome], errors="coerce")
    prepared["counterspeech"] = prepared["correction"].eq(
        "counterspeech"
    ).astype(float)
    prepared["moderator_explanation"] = prepared["correction"].eq(
        "moderator_explanation"
    ).astype(float)
    prepared["order"] = pd.to_numeric(
        prepared["trial_index"],
        errors="coerce",
    )
    columns = ["counterspeech", "moderator_explanation", "order"]
    prepared = prepared.dropna(
        subset=["_y", "participant_id", *columns]
    )
    if prepared.empty:
        return pd.DataFrame()
    for column in ["_y", *columns]:
        prepared[column] = prepared[column] - prepared.groupby(
            "participant_id"
        )[column].transform("mean")
    x = prepared[columns].to_numpy(float)
    y = prepared["_y"].to_numpy(float)
    bread = np.linalg.pinv(x.T @ x)
    beta = bread @ x.T @ y
    residual = y - x @ beta
    meat = np.zeros((x.shape[1], x.shape[1]))
    for _, indices in prepared.groupby("participant_id").groups.items():
        positions = prepared.index.get_indexer(indices)
        xg = x[positions]
        ug = residual[positions]
        score = xg.T @ ug
        meat += np.outer(score, score)
    clusters = prepared["participant_id"].nunique()
    if clusters > 1:
        meat *= clusters / (clusters - 1)
    variance = bread @ meat @ bread
    se = np.sqrt(np.maximum(0.0, np.diag(variance)))
    statistic = np.divide(
        beta,
        se,
        out=np.zeros_like(beta),
        where=se > 0,
    )
    degrees = max(1, clusters - 1)
    p_value = 2 * stats.t.sf(np.abs(statistic), degrees)
    return pd.DataFrame(
        {
            "outcome": outcome,
            "term": columns,
            "estimate": beta,
            "cluster_std_error": se,
            "ci_low": beta - stats.t.ppf(0.975, degrees) * se,
            "ci_high": beta + stats.t.ppf(0.975, degrees) * se,
            "p_value": p_value,
            "n_trials": len(prepared),
            "n_participants": clusters,
            "reference_correction": "none",
        }
    )


def analyze_local_rct(
    participant_path: Path,
    trial_path: Path,
    protocol: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    participants = pd.read_csv(participant_path)
    trials = pd.read_csv(trial_path)
    required_participant = {
        "participant_id",
        "arm",
        "ranking",
        "context",
        "consent",
        "eligible",
        "completed",
        *PRIMARY_OUTCOMES,
    }
    required_trial = {
        "participant_id",
        "correction",
        "trial_index",
        "community",
        "stimulus_conflict_band",
        "survey_item_id",
        "survey_construct",
        "survey_response",
        "mechanism_score",
        *PRIMARY_OUTCOMES,
    }
    missing = sorted(
        required_participant.difference(participants)
        | required_trial.difference(trials)
    )
    ethics_ready = bool(protocol.get("ethics_approval_id"))
    preregistered = bool(protocol.get("preregistration_url"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if missing:
        result = {
            "status": "invalid",
            "claim_allowed": False,
            "missing_columns": missing,
        }
        write_json(output_dir / "local_rct_summary.json", result)
        return result

    completion = (
        participants.groupby("arm", observed=True)["completed"]
        .agg(["mean", "count"])
        .reset_index()
    )
    completion.to_csv(output_dir / "attrition_by_arm.csv", index=False)
    balance = (
        participants.groupby("arm", observed=True)[
            ["prior_platform_use", "baseline_conflict_tolerance"]
        ]
        .agg(["mean", "std", "count"])
    )
    balance.to_csv(output_dir / "randomization_balance.csv")

    factorial_parts = [
        _hc3_fit(participants, outcome)
        for outcome in PRIMARY_OUTCOMES
    ]
    valid_factorial = [
        frame for frame in factorial_parts if not frame.empty
    ]
    factorial = (
        pd.concat(valid_factorial, ignore_index=True)
        if valid_factorial
        else pd.DataFrame()
    )
    correction_parts = [
        _within_correction_fit(trials, outcome)
        for outcome in PRIMARY_OUTCOMES
    ]
    valid_correction = [
        frame for frame in correction_parts if not frame.empty
    ]
    correction = (
        pd.concat(valid_correction, ignore_index=True)
        if valid_correction
        else pd.DataFrame()
    )
    for frame in (factorial, correction):
        if not frame.empty:
            frame["holm_p_value"] = _holm(
                frame["p_value"].to_numpy(float)
            )
    factorial.to_csv(
        output_dir / "factorial_primary_effects.csv",
        index=False,
    )
    correction.to_csv(
        output_dir / "within_participant_correction_effects.csv",
        index=False,
    )

    mechanism_scores = (
        trials.groupby(
            ["participant_id", "context", "survey_construct"],
            observed=True,
        )["mechanism_score"]
        .mean()
        .reset_index()
    )
    mechanism_scores.to_csv(
        output_dir / "mechanism_construct_scores.csv",
        index=False,
    )
    conflict_scores = mechanism_scores.loc[
        mechanism_scores["survey_construct"].eq("conflict")
    ]
    neutral = conflict_scores.loc[
        conflict_scores["context"].eq("neutral"),
        "mechanism_score",
    ]
    conflict_values = conflict_scores.loc[
        conflict_scores["context"].eq("conflict"),
        "mechanism_score",
    ]
    manipulation = {
        "neutral_mean": (
            float(neutral.mean()) if len(neutral) else None
        ),
        "conflict_mean": (
            float(conflict_values.mean()) if len(conflict_values) else None
        ),
        "difference": (
            float(conflict_values.mean() - neutral.mean())
            if len(neutral) and len(conflict_values)
            else None
        ),
        "p_value": (
            float(
                stats.ttest_ind(
                    conflict_values,
                    neutral,
                    equal_var=False,
                ).pvalue
            )
            if len(neutral) > 1 and len(conflict_values) > 1
            else None
        ),
    }
    completed_participants = int(
        participants["completed"].astype(bool).sum()
    )
    minimum_confirmatory_n = int(protocol.get("confirmatory_n", 0))
    sample_size_ready = (
        minimum_confirmatory_n > 0
        and completed_participants >= minimum_confirmatory_n
    )
    expected_arms = (
        len(protocol.get("ranking_arms", []))
        * len(protocol.get("contexts", []))
    ) or 10
    observed_arms = int(participants["arm"].nunique())
    arm_coverage_ready = observed_arms == expected_arms
    claim_allowed = (
        ethics_ready
        and preregistered
        and sample_size_ready
        and arm_coverage_ready
        and str(protocol.get("mode", "demo")) == "production"
    )
    result = {
        "status": "complete",
        "claim_allowed": claim_allowed,
        "analysis_scope": (
            "confirmatory"
            if claim_allowed
            else "pipeline/demo only"
        ),
        "participants": int(len(participants)),
        "completed_participants": completed_participants,
        "minimum_confirmatory_n": minimum_confirmatory_n,
        "confirmatory_sample_size_reached": sample_size_ready,
        "trials": int(len(trials)),
        "arms": observed_arms,
        "expected_arms": expected_arms,
        "confirmatory_arm_coverage_reached": arm_coverage_ready,
        "factorial_primary_contrasts": int(len(factorial)),
        "within_participant_correction_contrasts": int(len(correction)),
        "mechanism_construct_rows": int(len(mechanism_scores)),
        "manipulation_check": manipulation,
        "ethics_approval_recorded": ethics_ready,
        "preregistration_recorded": preregistered,
        "participant_source_sha256": sha256_file(participant_path),
        "trial_source_sha256": sha256_file(trial_path),
        "methods": {
            "ranking_context": (
                "intention-to-treat linear model with HC3 standard errors; "
                "new and neutral are reference levels"
            ),
            "correction": (
                "participant-demeaned linear model with participant-clustered "
                "standard errors; none is the reference"
            ),
            "multiplicity": "Holm adjustment across frozen primary contrasts",
            "binary_sensitivity": (
                "The primary table uses linear probability effects; "
                "preregistered logistic sensitivity remains required for "
                "confirmatory reporting"
            ),
        },
    }
    write_json(output_dir / "local_rct_summary.json", result)
    return result
