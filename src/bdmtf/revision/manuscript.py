from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from bdmtf.revision.provenance import sha256_file, write_json


SECTION_ORDER = (
    "Introduction",
    "Related Work",
    "Methodology",
    "Agent Modeling",
    "Experimental Setup and Results",
    "Conclusion",
)


def _source_extract(root: Path) -> Path:
    candidates = (
        root / "evidence" / "What_Makes_Content_Go_Vi_extracted.txt",
        root.parent.parent / "work" / "pdfs" / "What_Makes_Content_Go_Vi_extracted.txt",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("The original PDF extract was not found in evidence/ or the restored work directory")


def _decode_extract(path: Path) -> str:
    return path.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="replace").replace("\r", "")


def sanitize_original_extract(text: str) -> tuple[str, dict[str, Any]]:
    lines = text.splitlines()
    cleaned: list[str] = []
    removed_blocks: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        lower = stripped.lower()
        if lower.startswith("for icml") and "review" in lower:
            start = index
            while index < len(lines) and not lines[index].strip().startswith("===== PAGE"):
                index += 1
            removed_blocks.append(_removed_block(lines[start:index], start + 1, "review-directed instruction"))
            continue
        if len(stripped) == 1:
            end = index
            characters: list[str] = []
            while end < len(lines) and (len(lines[end].strip()) <= 1):
                characters.append(lines[end].strip())
                end += 1
            reconstructed = "".join(characters).lower()
            if "include" in reconstructed and "review" in reconstructed and "phrase" in reconstructed:
                removed_blocks.append(_removed_block(lines[index:end], index + 1, "review phrase directive"))
                index = end
                continue
        if (
            stripped.startswith("===== PAGE")
            or stripped == "Submission and Formatting Instructions for ICML 2026"
            or lower.startswith("confidential reviewer copy.")
        ):
            index += 1
            continue
        cleaned.append(lines[index])
        index += 1
    sanitized = "\n".join(cleaned).strip() + "\n"
    audit = {
        "source_lines": len(lines),
        "sanitized_lines": len(cleaned),
        "removed_block_count": len(removed_blocks),
        "removed_blocks": removed_blocks,
        "sanitized_sha256": hashlib.sha256(sanitized.encode("utf-8")).hexdigest(),
    }
    return sanitized, audit


def _removed_block(lines: list[str], line: int, reason: str) -> dict[str, Any]:
    encoded = "\n".join(lines).encode("utf-8")
    return {
        "start_line": line,
        "line_count": len(lines),
        "reason": reason,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _macro(name: str, value: Any) -> str:
    text = str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
    return rf"\newcommand{{\{name}}}{{{text}}}"


def build_result_macros(root: Path, output_path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {
        "DataPosts": "not available",
        "ReconstructableCascades": "not available",
        "EmpiricalLeafDepth": "not available",
        "PaperLeafDepth": "18.4",
        "ExactReproductionRuns": "not available",
        "ExactBaselineLeafDepth": "not available",
        "ExactToxicLeafDepth": "not available",
        "ExactVolumeRatio": "not available",
        "FactorialBlocks": "not available",
        "FactorialPosts": "not available",
        "FactorialJointVolume": "not available",
        "FactorialJointVolumeLow": "not available",
        "FactorialJointVolumeHigh": "not available",
        "FactorialJointDepth": "not available",
        "FactorialJointDepthLow": "not available",
        "FactorialJointDepthHigh": "not available",
        "FactorialInteractionVolume": "not available",
        "HeldoutPosts": "not available",
        "HeldoutBlocks": "not available",
        "HeldoutVolumeAdditive": "not available",
        "HeldoutVolumeInteraction": "not available",
        "HeldoutVolumeGain": "not available",
        "HeldoutDepthAdditive": "not available",
        "HeldoutDepthInteraction": "not available",
        "HeldoutDepthGain": "not available",
        "InDomainWinner": "not available",
        "InDomainBestMean": "not available",
        "InDomainMedianWinner": "not available",
        "InDomainBestMedian": "not available",
        "LOCOWinnerMean": "not available",
        "LOCOWinnerMedian": "not available",
        "HNTargetCascades": "not available",
        "LemmyTargetCascades": "not available",
        "ExternalTargetCascades": "not available",
        "CrossPlatformSimulations": "not available",
        "CrossPlatformWinner": "not available",
        "HNCollectionStatus": "not collected",
        "HNDiscoveryMode": "not recorded",
        "AblationRuns": "not available",
        "ShallowSwarmRuns": "not available",
        "ShallowSwarmRate": "not available",
        "AblationMedianVolumeRatio": "not available",
        "AblationMedianDepthDelta": "not available",
        "PhaseLocalScenarios": "not available",
        "PhaseLocalSupport": "not available",
        "PhaseLocalVolumeMin": "not available",
        "PhaseLocalVolumeMax": "not available",
        "PhaseLocalDepthMin": "not available",
        "PhaseLocalDepthMax": "not available",
        "PhaseGlobalCells": "not available",
        "PhaseGlobalSupport": "not available",
        "HNAdapterBestBudget": "not available",
        "HNAdapterBestDistance": "not available",
        "HNTargetDefaultDistance": "not available",
        "HNRedditZeroShotDistance": "not available",
        "HNAdapterGainDefault": "not available",
        "HNAdapterGainZeroShot": "not available",
        "SemanticComments": "not available",
        "SemanticLabels": "not available",
        "SemanticCalls": "not available",
        "SemanticHeldoutPosts": "not available",
        "SemanticSupportedOutcomes": "not available",
        "SemanticAntagonismAlpha": "not available",
        "SemanticConflictAlpha": "not available",
        "APITaskStatus": "not completed",
        "APITaskCount": "not available",
        "TBBTStatus": "not completed",
        "TBBTRecords": "not available",
        "TBBTPanelRows": "not available",
        "TBBTOutcomeStatus": "not completed",
        "LemmyStatus": "not completed",
        "LemmyOutcomeStatus": "not completed",
        "LemmyInterventions": "not available",
        "LemmyRiskSetPairs": "not available",
        "LemmyEffectiveControls": "not available",
        "LemmyMaximumControlWeight": "not available",
        "LemmyReplyEffect": "not available",
        "LemmyReplyCILow": "not available",
        "LemmyReplyCIHigh": "not available",
        "LemmyAuthorEffect": "not available",
        "LemmyAuthorCILow": "not available",
        "LemmyAuthorCIHigh": "not available",
        "LemmyDepthEffect": "not available",
        "LemmyDepthCILow": "not available",
        "LemmyDepthCIHigh": "not available",
        "StoryExactMatches": "not available",
        "StorySemanticMatches": "not available",
        "StoryURLResolutionStatus": "not executed",
        "RedditExpansionStatus": "not completed",
        "RedditExpansionCommunities": "not available",
        "RedditExpansionCascades": "not available",
        "RedditExpansionComments": "not available",
        "RedditExpansionAdaptedDistance": "not available",
        "RedditExpansionLOCODistance": "not available",
        "AgentReplayTestInterventions": "not available",
        "AgentReplaySeeds": "not available",
        "AgentReplayEvents": "not available",
        "AgentReplayMAE": "not available",
        "AgentReplayMAELow": "not available",
        "AgentReplayMAEHigh": "not available",
        "AgentReplayDirection": "not available",
        "AgentReplayCoverage": "not available",
        "AgentReplayZeroGain": "not available",
        "AgentReplayZeroGainLow": "not available",
        "AgentReplayZeroGainHigh": "not available",
        "ContentExactPairs": "not available",
        "ContentUniqueURLs": "not available",
        "ContentStructuralPairs": "not available",
        "ContentSemanticPairs": "not available",
        "ContentAdaptedDistance": "not available",
        "ContentAdaptedLow": "not available",
        "ContentAdaptedHigh": "not available",
        "ContentZeroShotDistance": "not available",
        "ContentAdaptationGain": "not available",
        "ContentAdaptationGainLow": "not available",
        "ContentAdaptationGainHigh": "not available",
        "ContentCommentCorrelation": "not available",
        "ContentCommentCorrelationP": "not available",
        "NaturalExperimentStatus": "not completed",
        "HumanRCTStatus": "not completed",
        "ProspectiveStatus": "not frozen",
    }
    sources: list[dict[str, Any]] = []

    def load_json(relative: str) -> dict[str, Any] | None:
        path = root / relative
        if not path.is_file():
            return None
        sources.append({"path": relative, "sha256": sha256_file(path)})
        return json.loads(path.read_text(encoding="utf-8"))

    audit = load_json("artifacts/data_audit/data_audit.json")
    if audit:
        values.update(
            {
                "DataPosts": audit["n_posts"],
                "ReconstructableCascades": audit["n_reconstructable_cascades"],
                "EmpiricalLeafDepth": f"{audit['empirical_mean_leaf_depth']:.2f}",
            }
        )
    exact_status = load_json("run_outputs/paper_exact_full/run_status.json")
    exact_table_path = (
        root
        / "run_outputs"
        / "paper_exact_full"
        / "paper_tables"
        / "table1_aggregate_structural_effects.csv"
    )
    if exact_status and exact_table_path.is_file():
        exact_table = pd.read_csv(exact_table_path)
        leaf = exact_table[exact_table["metric"].eq("mean_leaf_depth")].iloc[0]
        volume = exact_table[exact_table["metric"].eq("comment_volume")].iloc[0]
        values.update(
            {
                "ExactReproductionRuns": exact_status[
                    "completed_unique_runs"
                ],
                "ExactBaselineLeafDepth": f"{leaf['baseline_mean']:.3f}",
                "ExactToxicLeafDepth": (
                    f"{leaf['toxic_controversial_mean']:.3f}"
                ),
                "ExactVolumeRatio": f"{volume['change_ratio']:.3f}",
            }
        )
        sources.append(
            {
                "path": exact_table_path.relative_to(root).as_posix(),
                "sha256": sha256_file(exact_table_path),
            }
        )
    factorial = load_json(
        "run_outputs/reviewer_confirmatory_factorial/"
        "reviewer_analysis/factorial_manifest.json"
    )
    factorial_summary_path = (
        root
        / "run_outputs"
        / "reviewer_confirmatory_factorial"
        / "reviewer_analysis"
        / "factorial_summary.csv"
    )
    if factorial and factorial_summary_path.is_file():
        frame = pd.read_csv(factorial_summary_path)
        frame = frame[frame["scope"].eq("all")]

        def factorial_row(metric: str, contrast: str) -> pd.Series:
            return frame[
                frame["metric"].eq(metric) & frame["contrast"].eq(contrast)
            ].iloc[0]

        joint_volume = factorial_row("comment_volume", "joint_log")
        joint_depth = factorial_row("mean_leaf_depth", "joint")
        interaction = factorial_row(
            "comment_volume", "core_ranking_interaction_log"
        )
        values.update(
            {
                "FactorialBlocks": factorial["complete_post_seed_blocks"],
                "FactorialPosts": factorial["complete_posts"],
                "FactorialJointVolume": f"{math.exp(joint_volume['mean']):.3f}",
                "FactorialJointVolumeLow": f"{math.exp(joint_volume['ci_low']):.3f}",
                "FactorialJointVolumeHigh": f"{math.exp(joint_volume['ci_high']):.3f}",
                "FactorialJointDepth": f"{joint_depth['mean']:.3f}",
                "FactorialJointDepthLow": f"{joint_depth['ci_low']:.3f}",
                "FactorialJointDepthHigh": f"{joint_depth['ci_high']:.3f}",
                "FactorialInteractionVolume": f"{math.exp(interaction['mean']):.3f}",
            }
        )
        sources.append(
            {
                "path": factorial_summary_path.relative_to(root).as_posix(),
                "sha256": sha256_file(factorial_summary_path),
            }
        )
    response_manifest = load_json(
        "artifacts/reviewer_validation/response_benchmark/"
        "response_benchmark_manifest.json"
    )
    response_summary_path = (
        root
        / "artifacts"
        / "reviewer_validation"
        / "response_benchmark"
        / "response_benchmark_summary.csv"
    )
    response_improvement_path = (
        root
        / "artifacts"
        / "reviewer_validation"
        / "response_benchmark"
        / "response_benchmark_improvements.csv"
    )
    if (
        response_manifest
        and response_summary_path.is_file()
        and response_improvement_path.is_file()
    ):
        summary = pd.read_csv(response_summary_path)
        gains = pd.read_csv(response_improvement_path)

        def response_mae(metric: str, model: str) -> float:
            return float(
                summary[
                    summary["metric"].eq(metric)
                    & summary["model"].eq(model)
                ].iloc[0]["mae"]
            )

        def response_gain(metric: str) -> float:
            return float(
                gains[
                    gains["metric"].eq(metric)
                    & gains["full_model"].eq("interaction_community")
                    & gains["reduced_model"].eq("additive_community")
                ].iloc[0]["mean_absolute_error_improvement"]
            )

        values.update(
            {
                "HeldoutPosts": response_manifest["n_test_posts"],
                "HeldoutBlocks": response_manifest["n_test_blocks"],
                "HeldoutVolumeAdditive": f"{response_mae('comment_volume', 'additive_community'):.3f}",
                "HeldoutVolumeInteraction": f"{response_mae('comment_volume', 'interaction_community'):.3f}",
                "HeldoutVolumeGain": f"{response_gain('comment_volume'):.3f}",
                "HeldoutDepthAdditive": f"{response_mae('mean_leaf_depth', 'additive_community'):.3f}",
                "HeldoutDepthInteraction": f"{response_mae('mean_leaf_depth', 'interaction_community'):.3f}",
                "HeldoutDepthGain": f"{response_gain('mean_leaf_depth'):.3f}",
            }
        )
        for path in (response_summary_path, response_improvement_path):
            sources.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
    loco = load_json("artifacts/external_validation/cross_community/summary.json")
    if loco:
        values["LOCOWinnerMean"] = loco["best_model_by_mean_normalized_wasserstein"].replace("zero_shot_", "")
        values["LOCOWinnerMedian"] = loco["best_model_by_median_normalized_wasserstein"].replace("zero_shot_", "")
    hn = load_json("artifacts/external_validation/cross_platform_transfer/summary.json")
    if hn:
        platform_counts = hn.get("n_target_cascades_by_platform", {})
        values["HNTargetCascades"] = platform_counts.get(
            "HackerNews",
            hn["n_target_cascades"],
        )
        values["LemmyTargetCascades"] = platform_counts.get("Lemmy", 0)
        values["ExternalTargetCascades"] = hn.get("n_target_cascades", "not available")
        values["CrossPlatformSimulations"] = hn.get("n_simulations", "not available")
        values["CrossPlatformWinner"] = (
            str(hn.get("best_model_by_median_normalized_wasserstein", "not available"))
            .replace("reddit_zero_shot_", "")
            .replace("_", " ")
        )
    hn_collection = load_json("data/external/hackernews/collection_manifest.json")
    if hn_collection:
        values["HNCollectionStatus"] = hn_collection.get("status", "unknown")
        values["HNDiscoveryMode"] = hn_collection.get("selection_mode", "unknown")
    api = load_json("artifacts/api/api_manifest.json")
    if api:
        values["APITaskStatus"] = api["status"]
        values["APITaskCount"] = api["expected_calls"]

    ranking_path = root / "artifacts" / "evaluation" / "model_ranking.csv"
    if ranking_path.is_file():
        ranking = pd.read_csv(ranking_path).sort_values("mean")
        values["InDomainWinner"] = str(ranking.iloc[0]["model"])
        values["InDomainBestMean"] = f"{float(ranking.iloc[0]['mean']):.3f}"
        median_row = ranking.sort_values("median").iloc[0]
        values["InDomainMedianWinner"] = str(median_row["model"])
        values["InDomainBestMedian"] = f"{float(median_row['median']):.3f}"
        sources.append({"path": str(ranking_path.relative_to(root)), "sha256": sha256_file(ranking_path)})
    ablation_path = root / "artifacts" / "runs" / "ablations" / "tradeoff_regions.csv"
    if ablation_path.is_file():
        ablations = pd.read_csv(ablation_path)
        values["AblationRuns"] = len(ablations)
        values["ShallowSwarmRuns"] = int(ablations["shallow_swarm"].astype(bool).sum())
        values["ShallowSwarmRate"] = (
            f"{100 * ablations['shallow_swarm'].astype(bool).mean():.1f}"
        )
        values["AblationMedianVolumeRatio"] = f"{ablations['volume_ratio'].median():.3f}"
        values["AblationMedianDepthDelta"] = (
            f"{ablations['leaf_depth_delta'].median():+.3f}"
        )
        sources.append({"path": str(ablation_path.relative_to(root)), "sha256": sha256_file(ablation_path)})
    phase_manifest = load_json(
        "artifacts/reviewer_validation/mechanism_phase_map/"
        "mechanism_phase_map_manifest.json"
    )
    phase_local_path = (
        root
        / "artifacts"
        / "reviewer_validation"
        / "mechanism_phase_map"
        / "confirmatory_local_scenarios.csv"
    )
    if phase_manifest and phase_local_path.is_file():
        local = pd.read_csv(phase_local_path)
        values.update(
            {
                "PhaseLocalScenarios": phase_manifest["confirmatory_local"][
                    "n_scenarios"
                ],
                "PhaseLocalSupport": phase_manifest["confirmatory_local"][
                    "supporting_scenarios"
                ],
                "PhaseLocalVolumeMin": f"{local['volume_ratio'].min():.3f}",
                "PhaseLocalVolumeMax": f"{local['volume_ratio'].max():.3f}",
                "PhaseLocalDepthMin": (
                    f"{local['leaf_depth_delta'].min():+.3f}"
                ),
                "PhaseLocalDepthMax": (
                    f"{local['leaf_depth_delta'].max():+.3f}"
                ),
                "PhaseGlobalCells": phase_manifest["global_stress"]["n_cells"],
                "PhaseGlobalSupport": phase_manifest["global_stress"][
                    "supporting_cells"
                ],
            }
        )
        sources.append(
            {
                "path": phase_local_path.relative_to(root).as_posix(),
                "sha256": sha256_file(phase_local_path),
            }
        )
    hn_curve_manifest = load_json(
        "artifacts/external_validation/platform_adapter_curve/"
        "platform_adapter_curve_manifest.json"
    )
    hn_curve_path = (
        root
        / "artifacts"
        / "external_validation"
        / "platform_adapter_curve"
        / "adapter_budget_curve.csv"
    )
    if hn_curve_manifest and hn_curve_path.is_file():
        curve = pd.read_csv(hn_curve_path)
        best = curve.sort_values(
            ["all_metric_median_distance", "validation_budget"]
        ).iloc[0]
        values.update(
            {
                "HNAdapterBestBudget": int(best["validation_budget"]),
                "HNAdapterBestDistance": (
                    f"{best['all_metric_median_distance']:.3f}"
                ),
                "HNTargetDefaultDistance": (
                    f"{best['target_default_median_distance']:.3f}"
                ),
                "HNRedditZeroShotDistance": (
                    f"{best['reddit_zero_shot_median_distance']:.3f}"
                ),
                "HNAdapterGainDefault": (
                    f"{100 * best['gain_over_target_default']:.1f}"
                ),
                "HNAdapterGainZeroShot": (
                    f"{100 * best['gain_over_reddit_zero_shot']:.1f}"
                ),
            }
        )
        sources.append(
            {
                "path": hn_curve_path.relative_to(root).as_posix(),
                "sha256": sha256_file(hn_curve_path),
            }
        )
    semantic_execution = load_json(
        "artifacts/reviewer_validation/semantic_calibration/"
        "semantic_execution_manifest.json"
    )
    semantic_analysis = load_json(
        "artifacts/reviewer_validation/semantic_calibration/"
        "semantic_analysis_manifest.json"
    )
    if semantic_execution and semantic_analysis:
        values.update(
            {
                "SemanticComments": semantic_analysis["consensus"][
                    "consensus_comments"
                ],
                "SemanticLabels": semantic_analysis["agreement"][
                    "parsed_labels"
                ],
                "SemanticCalls": semantic_execution[
                    "completed_unique_calls"
                ],
                "SemanticHeldoutPosts": semantic_analysis[
                    "heldout_analysis"
                ]["n_test"],
                "SemanticSupportedOutcomes": semantic_analysis[
                    "heldout_analysis"
                ]["supported_outcomes"],
            }
        )
        agreement_path = (
            root
            / "artifacts"
            / "reviewer_validation"
            / "semantic_calibration"
            / "semantic_model_agreement.csv"
        )
        if agreement_path.is_file():
            agreement = pd.read_csv(agreement_path).set_index("label")
            values["SemanticAntagonismAlpha"] = (
                f"{agreement.loc['antagonism', 'krippendorff_alpha_interval']:.3f}"
            )
            values["SemanticConflictAlpha"] = (
                f"{agreement.loc['conflict_amplifying', 'krippendorff_alpha_interval']:.3f}"
            )
            sources.append(
                {
                    "path": agreement_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(agreement_path),
                }
            )

    status_paths = {
        "TBBTStatus": "artifacts/interventions/tbbt/import_manifest.json",
        "LemmyStatus": "data/external/lemmy/collection_manifest.json",
        "RedditExpansionStatus": "artifacts/external_validation/reddit_expansion/expansion_manifest.json",
        "NaturalExperimentStatus": (
            "artifacts/interventions/"
            "natural_confirmatory_lemmy_calibrated/"
            "natural_experiment_summary.json"
        ),
        "HumanRCTStatus": "artifacts/human_rct/rct_summary.json",
        "ProspectiveStatus": "artifacts/prospective/prospective_manifest.json",
    }
    for macro_name, relative in status_paths.items():
        payload = load_json(relative)
        if payload:
            values[macro_name] = payload.get("status", "unknown")
            if macro_name == "TBBTStatus":
                values["TBBTRecords"] = payload.get("records_read", "not available")
                values["TBBTPanelRows"] = payload.get("panel_rows", "not available")

    tbbt_outcome = load_json("artifacts/interventions/tbbt/outcome_panel_manifest.json")
    if tbbt_outcome:
        values["TBBTOutcomeStatus"] = tbbt_outcome.get("status", "unknown")
    lemmy_outcome = load_json(
        "data/external/lemmy/outcomes_confirmatory/"
        "outcome_collection_manifest.json"
    )
    if lemmy_outcome:
        values["LemmyOutcomeStatus"] = lemmy_outcome.get("status", "unknown")
        values["LemmyInterventions"] = lemmy_outcome.get(
            "n_treated_interventions", "not available"
        )
        values["LemmyRiskSetPairs"] = lemmy_outcome.get("n_risk_set_pairs", "not available")
    lemmy_natural = load_json(
        "artifacts/interventions/natural_confirmatory_lemmy_calibrated/"
        "natural_experiment_summary.json"
    )
    if lemmy_natural:
        diagnostics = {
            item["outcome"]: item
            for item in lemmy_natural.get("primary_diagnostics", [])
        }
        aggregate = lemmy_natural.get(
            "aggregate_synthetic_control", {}
        )
        values["LemmyEffectiveControls"] = (
            f"{float(aggregate['effective_controls']):.1f}"
        )
        values["LemmyMaximumControlWeight"] = (
            f"{100 * float(aggregate['maximum_control_weight']):.2f}"
        )
        for outcome, prefix in (
            ("reply_count", "LemmyReply"),
            ("active_authors", "LemmyAuthor"),
            ("max_depth", "LemmyDepth"),
        ):
            row = diagnostics[outcome]
            values[prefix + "Effect"] = f"{float(row['effect']):.3f}"
            values[prefix + "CILow"] = f"{float(row['ci_low']):.3f}"
            values[prefix + "CIHigh"] = f"{float(row['ci_high']):.3f}"
    story = load_json("artifacts/story_matching/match_manifest.json")
    if story:
        values["StoryExactMatches"] = story.get("n_exact_url", "not available")
        values["StorySemanticMatches"] = story.get("n_semantic_event", "not available")
        values["StoryURLResolutionStatus"] = story.get("url_resolution", {}).get(
            "status",
            "not recorded",
        )

    reddit_expansion = load_json(
        "artifacts/external_validation/reddit_expansion/validation_summary.json"
    )
    if reddit_expansion:
        values.update(
            {
                "RedditExpansionCommunities": reddit_expansion["n_communities"],
                "RedditExpansionCascades": reddit_expansion["n_cascades"],
                "RedditExpansionComments": reddit_expansion["n_comments"],
                "RedditExpansionAdaptedDistance": (
                    f"{reddit_expansion['protocols']['target_adapted']['learned_bdmtf_median_normalized_wasserstein']:.3f}"
                ),
                "RedditExpansionLOCODistance": (
                    f"{reddit_expansion['protocols']['leave_one_community_out']['learned_bdmtf_median_normalized_wasserstein']:.3f}"
                ),
            }
        )

    agent_replay = load_json(
        "artifacts/interventions/agent_replay/agent_replay_manifest.json"
    )
    if agent_replay:
        result = agent_replay["bdmtf_agent_replay"]
        gain = agent_replay["bdmtf_vs_zero_effect_mae_improvement"]
        values.update(
            {
                "AgentReplayTestInterventions": agent_replay["test_interventions"],
                "AgentReplaySeeds": agent_replay["test_simulation_seeds"],
                "AgentReplayEvents": agent_replay["simulated_event_rows"],
                "AgentReplayMAE": f"{result['primary_mae']:.3f}",
                "AgentReplayMAELow": f"{result['primary_mae_ci'][0]:.3f}",
                "AgentReplayMAEHigh": f"{result['primary_mae_ci'][1]:.3f}",
                "AgentReplayDirection": f"{100 * result['direction_accuracy']:.1f}",
                "AgentReplayCoverage": f"{100 * result['interval_coverage']:.1f}",
                "AgentReplayZeroGain": f"{gain['mean']:.3f}",
                "AgentReplayZeroGainLow": f"{gain['ci'][0]:.3f}",
                "AgentReplayZeroGainHigh": f"{gain['ci'][1]:.3f}",
            }
        )

    content_matched = load_json(
        "artifacts/external_validation/lemmy_content_matched/"
        "content_matched_manifest.json"
    )
    if content_matched:
        matching = content_matched["matching"]
        result = content_matched["exact_url_model_result"]
        post_result = content_matched["post_level_result"]
        values.update(
            {
                "ContentExactPairs": matching["exact_post_pairs"],
                "ContentUniqueURLs": matching["exact_unique_urls"],
                "ContentStructuralPairs": matching["exact_structural_pairs"],
                "ContentSemanticPairs": matching["semantic_structural_pairs"],
                "ContentAdaptedDistance": f"{result['lemmy_structure_selected_mean_normalized_mae']:.3f}",
                "ContentAdaptedLow": f"{result['lemmy_structure_selected_ci'][0]:.3f}",
                "ContentAdaptedHigh": f"{result['lemmy_structure_selected_ci'][1]:.3f}",
                "ContentZeroShotDistance": f"{result['hackernews_zero_shot_mean_normalized_mae']:.3f}",
                "ContentAdaptationGain": f"{result['structure_selected_improvement_over_zero_shot']:.3f}",
                "ContentAdaptationGainLow": f"{result['structure_selected_improvement_ci'][0]:.3f}",
                "ContentAdaptationGainHigh": f"{result['structure_selected_improvement_ci'][1]:.3f}",
                "ContentCommentCorrelation": f"{post_result['spearman_comment_count']:.3f}",
                "ContentCommentCorrelationP": f"{post_result['spearman_p_value']:.3f}",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "% Generated from result artifacts. Do not edit by hand.\n"
        + "\n".join(_macro(name, value) for name, value in values.items())
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "status": "complete",
        "macros_sha256": sha256_file(output_path),
        "values": values,
        "sources": sources,
    }
    write_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest


MODEL_ORDER = (
    "empirical_bootstrap",
    "branching_process",
    "hawkes",
    "legacy_heuristic",
    "learned_bdmtf",
)
MODEL_LABELS = {
    "empirical_bootstrap": "Empirical resampling",
    "branching_process": "Branching process",
    "hawkes": "Hawkes",
    "legacy_heuristic": "Theory-specified BDMTF",
    "learned_bdmtf": "Learned BDMTF",
}


def _ranking_values(path: Path, prefix: str = "") -> dict[str, tuple[float, float]]:
    frame = pd.read_csv(path)
    frame["model"] = frame["model"].astype(str).str.removeprefix(prefix)
    return {
        str(row.model): (float(row.mean), float(row.median))
        for row in frame.itertuples(index=False)
    }


def _bold_min(value: float, values: list[float]) -> str:
    rendered = f"{value:.3f}"
    return rf"\textbf{{{rendered}}}" if value == min(values) else rendered


def _write_fidelity_table(root: Path, output_path: Path, wide: bool = False) -> None:
    in_domain = _ranking_values(root / "artifacts/evaluation/model_ranking.csv")
    loco = _ranking_values(
        root / "artifacts/external_validation/cross_community/evaluation/model_ranking.csv",
        "zero_shot_",
    )
    cross = pd.read_csv(
        root
        / "artifacts/external_validation/cross_platform_transfer/evaluation/fidelity_distances.csv"
    )
    cross["platform"] = cross["community"].astype(str).str.split(":").str[0]
    cross["model"] = cross["model"].astype(str).str.removeprefix("reddit_zero_shot_")
    platform_ranks = (
        cross.groupby(["platform", "model"])["normalized_wasserstein"]
        .agg(["mean", "median"])
        .reset_index()
    )

    columns: list[tuple[str, dict[str, float]]] = [
        ("Reddit mean", {model: values[0] for model, values in in_domain.items()}),
        ("Reddit med.", {model: values[1] for model, values in in_domain.items()}),
        ("LOCO mean", {model: values[0] for model, values in loco.items()}),
        ("LOCO med.", {model: values[1] for model, values in loco.items()}),
    ]
    for platform, statistic, label in (
        ("HackerNews", "mean", "HN mean"),
        ("HackerNews", "median", "HN med."),
        ("Lemmy", "median", "Lemmy med."),
    ):
        subset = platform_ranks[platform_ranks["platform"].eq(platform)]
        columns.append(
            (
                label,
                {
                    str(row.model): float(getattr(row, statistic))
                    for row in subset.itertuples(index=False)
                },
            )
        )

    table_environment = "table*" if wide else "table"
    lines = [
        rf"\begin{{{table_environment}}}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        "Model & " + " & ".join(label for label, _ in columns) + r" \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        cells = []
        for _, values in columns:
            all_values = [values[item] for item in MODEL_ORDER]
            cells.append(_bold_min(values[model], all_values))
        lines.append(MODEL_LABELS[model] + " & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            (
                r"\caption{Normalized Wasserstein ranking across held-out Reddit, "
                r"leave-one-community-out (LOCO), and zero-shot external tests. "
                r"Lower is better; bold marks the best entry in each column. "
                r"Reddit and LOCO aggregate 75 community--metric comparisons per model; "
                r"HN uses 15 metrics over 5,000 cascades; Lemmy uses 60 "
                r"community--metric comparisons over seven pilot cascades.}"
            ),
            r"\label{tab:fidelity-ranking}",
            rf"\end{{{table_environment}}}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_hn_structure_table(root: Path, output_path: Path, wide: bool = False) -> None:
    directory = root / "artifacts/external_validation/cross_platform_transfer"
    empirical = pd.read_parquet(directory / "empirical_metrics.parquet")
    simulated = pd.read_parquet(directory / "simulated_metrics.parquet")
    metrics = [
        ("size", "Size", 1.0),
        ("max_depth", "Max depth", 1.0),
        ("mean_leaf_depth", "Leaf depth", 1.0),
        ("root_reply_share", "Root share", 1.0),
        ("duration_minutes", "Duration (h)", 1 / 60),
    ]
    empirical_values = empirical[empirical["platform"].eq("HackerNews")][
        [name for name, _, _ in metrics]
    ].mean()
    simulated = simulated[simulated["platform"].eq("HackerNews")].copy()
    simulated["model"] = simulated["model"].astype(str).str.removeprefix("reddit_zero_shot_")
    simulated_values = simulated.groupby("model")[[name for name, _, _ in metrics]].mean()
    nearest: dict[str, str] = {}
    for metric, _, _ in metrics:
        nearest[metric] = min(
            MODEL_ORDER,
            key=lambda model: abs(
                float(simulated_values.loc[model, metric]) - float(empirical_values[metric])
            ),
        )

    table_environment = "table*" if wide else "table"
    lines = [
        rf"\begin{{{table_environment}}}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        "Source & " + " & ".join(label for _, label, _ in metrics) + r" \\",
        r"\midrule",
        "Observed HN & "
        + " & ".join(
            f"{float(empirical_values[name]) * scale:.2f}" for name, _, scale in metrics
        )
        + r" \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        cells = []
        for metric, _, scale in metrics:
            rendered = f"{float(simulated_values.loc[model, metric]) * scale:.2f}"
            cells.append(rf"\textbf{{{rendered}}}" if nearest[metric] == model else rendered)
        lines.append(MODEL_LABELS[model] + " & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Observed and zero-shot simulated Hacker News means. "
                r"Bold marks the closest simulated mean to the observed value in each column. "
                r"These point summaries complement, but do not replace, the distributional "
                r"distances in Table~\ref{tab:fidelity-ranking}.}"
            ),
            r"\label{tab:hn-structure}",
            rf"\end{{{table_environment}}}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_ablation_table(root: Path, output_path: Path, wide: bool = False) -> None:
    frame = pd.read_csv(root / "artifacts/runs/ablations/tradeoff_regions.csv")
    groups = [
        ("Reference", frame["scenario"].eq("reference"), "1"),
        ("Trait assignment", frame["scenario"].str.startswith("trait_mode="), "4"),
        ("Leader fraction", frame["scenario"].str.startswith("leader_fraction="), "4"),
        ("Viewport", frame["scenario"].str.startswith("viewport_k="), "4"),
        ("Population", frame["scenario"].str.startswith("num_agents="), "4"),
        ("Horizon", frame["scenario"].str.startswith("steps="), "3"),
        ("Ranking", frame["scenario"].str.startswith("ranking="), "5"),
        ("Depth preference", frame["scenario"].str.startswith("deep_drill_lambda="), "4"),
        ("Activation", frame["scenario"].str.startswith("activation_multiplier="), "3"),
        ("External traffic", frame["scenario"].str.startswith("external_traffic="), "2"),
        ("Moderation", frame["scenario"].str.startswith("moderation_probability="), "3"),
        ("Counterspeech", frame["scenario"].str.startswith("counterspeech_probability="), "3"),
        ("Hostile dropout", frame["scenario"].str.startswith("hostile_dropout_probability="), "3"),
        ("Trait--rank--viewport factorial", frame["scenario"].str.startswith("factorial|"), "80"),
    ]
    table_environment = "table*" if wide else "table"
    lines = [
        rf"\begin{{{table_environment}}}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Block & Levels & Tests & Shallow Swarm & Median $(V_1/V_0,\Delta D)$ \\",
        r"\midrule",
    ]
    for label, mask, levels in groups:
        subset = frame[mask]
        shallow = int(subset["shallow_swarm"].astype(bool).sum())
        lines.append(
            f"{label} & {levels} & {len(subset)} & {shallow} "
            + rf"({100 * shallow / max(1, len(subset)):.1f}\%) & "
            + f"({subset['volume_ratio'].median():.3f}, {subset['leaf_depth_delta'].median():+.3f})"
            + r" \\"
        )
    lines.extend(
        [
            r"\midrule",
            (
                f"All conditions & -- & {len(frame)} & "
                f"{int(frame['shallow_swarm'].astype(bool).sum())} "
                rf"({100 * frame['shallow_swarm'].astype(bool).mean():.1f}\%) & "
                f"({frame['volume_ratio'].median():.3f}, "
                f"{frame['leaf_depth_delta'].median():+.3f})"
                + r" \\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Ablation coverage and outcome. $V_1/V_0$ is conflict-to-baseline "
                r"volume and $\Delta D$ is the change in mean leaf depth. A Shallow Swarm "
                r"requires $V_1/V_0>1$ and $\Delta D<0$ within the same community. "
                r"Tests are scenario--community pairs, not independent human observations.}"
            ),
            r"\label{tab:ablation-summary}",
            rf"\end{{{table_environment}}}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_mechanism_phase_table(
    root: Path,
    output_path: Path,
    wide: bool = False,
) -> None:
    directory = (
        root
        / "artifacts"
        / "reviewer_validation"
        / "mechanism_phase_map"
    )
    local = pd.read_csv(directory / "confirmatory_local_scenarios.csv")
    broad = pd.read_csv(directory / "broad_phase_cells.csv")
    tier_labels = {
        "reference": "Global reference",
        "bounded_one_factor": "Bounded one-factor",
        "boundary_one_factor": "Boundary one-factor",
        "multifactor_stress": "Multifactor stress",
        "multifactor_boundary": "Multifactor boundary",
    }
    table_environment = "table*" if wide else "table"
    lines = [
        rf"\begin{{{table_environment}}}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrlrrr}",
        r"\toprule",
        (
            r"Evidence region & Unit & $N$ & Statistic & Shallow Swarm & "
            r"$V_1/V_0$ & $\Delta D$ \\"
        ),
        r"\midrule",
        (
            "Confirmatory local & Paired scenario & "
            f"{len(local)} & Range & "
            f"{int(local['shallow_swarm'].astype(bool).sum())} "
            rf"({100 * local['shallow_swarm'].astype(bool).mean():.1f}\%) & "
            f"[{local['volume_ratio'].min():.3f}, "
            f"{local['volume_ratio'].max():.3f}] & "
            f"[{local['leaf_depth_delta'].min():+.3f}, "
            f"{local['leaf_depth_delta'].max():+.3f}]"
            + r" \\"
        ),
        r"\midrule",
    ]
    for tier, label in tier_labels.items():
        subset = broad[broad["stress_tier"].astype(str).eq(tier)]
        shallow = int(subset["shallow_swarm"].astype(bool).sum())
        lines.append(
            f"{label} & Scenario--community & {len(subset)} & Median & "
            f"{shallow} "
            rf"({100 * shallow / max(1, len(subset)):.1f}\%) & "
            f"{subset['volume_ratio'].median():.3f} & "
            f"{subset['leaf_depth_delta'].median():+.3f}"
            + r" \\"
        )
    shallow = int(broad["shallow_swarm"].astype(bool).sum())
    lines.extend(
        [
            r"\midrule",
            (
                f"Global stress total & Scenario--community & {len(broad)} & Median & "
                f"{shallow} "
                rf"({100 * shallow / max(1, len(broad)):.1f}\%) & "
                f"{broad['volume_ratio'].median():.3f} & "
                f"{broad['leaf_depth_delta'].median():+.3f}"
                + r" \\"
            ),
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            (
                r"\caption{Two-tier mechanism phase analysis. The confirmatory "
                r"local tier perturbs the principal operating neighborhood "
                r"and reports scenario-level ranges. The global tier uses a "
                r"different learned simulator and intervention contrast to locate "
                r"phase boundaries; its unweighted cell share is not an estimate "
                r"of real-world prevalence. The two tiers answer local robustness "
                r"and global boundary questions, respectively.}"
            ),
            r"\label{tab:mechanism-phase-map}",
            rf"\end{{{table_environment}}}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_hn_adapter_curve_table(
    root: Path,
    output_path: Path,
    wide: bool = False,
) -> None:
    curve = pd.read_csv(
        root
        / "artifacts"
        / "external_validation"
        / "platform_adapter_curve"
        / "adapter_budget_curve.csv"
    )
    best_budget = int(
        curve.sort_values(
            ["all_metric_median_distance", "validation_budget"]
        ).iloc[0]["validation_budget"]
    )
    table_environment = "table*" if wide else "table"
    lines = [
        rf"\begin{{{table_environment}}}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{rrrlrr}",
        r"\toprule",
        (
            r"Val. $N$ & Test dist. & Gain vs. default & Ranking & "
            r"Viewport & Deep drill \\"
        ),
        r"\midrule",
    ]
    for row in curve.itertuples(index=False):
        budget = int(row.validation_budget)
        distance = f"{float(row.all_metric_median_distance):.3f}"
        if budget == best_budget:
            distance = rf"\textbf{{{distance}}}"
        viewport = (
            "all" if int(row.viewport_k) == -1 else str(int(row.viewport_k))
        )
        lines.append(
            f"{budget} & {distance} & "
            f"{100 * float(row.gain_over_target_default):+.1f}\\% & "
            f"{row.ranking} & {viewport} & "
            f"{float(row.deep_drill_lambda):.1f}"
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Hacker News adapter data-budget curve on the frozen "
                r"1,000-cascade temporal test split. The profile is fitted on "
                r"3,000 HN training cascades; Val.\ $N$ counts only the nested "
                r"validation subset used to select ranking, viewport, and "
                r"deep-drill parameters. Lower normalized Wasserstein distance "
                r"is better. Bold marks the best tested validation budget.}"
            ),
            r"\label{tab:hn-adapter-curve}",
            rf"\end{{{table_environment}}}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_lemmy_intervention_table(
    root: Path,
    output_path: Path,
    wide: bool = False,
) -> None:
    summary = json.loads(
        (
            root
            / "artifacts/interventions/"
            "natural_confirmatory_lemmy_calibrated/"
            "natural_experiment_summary.json"
        ).read_text(encoding="utf-8")
    )
    diagnostics = {
        item["outcome"]: item
        for item in summary["primary_diagnostics"]
    }
    labels = (
        ("reply_count", "Reply count"),
        ("active_authors", "Active authors"),
        ("max_depth", "Maximum depth"),
    )
    table_environment = "table*" if wide else "table"
    lines = [
        rf"\begin{{{table_environment}}}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Outcome & Calibrated ATT & 95\% CI & Pretrend slope \\",
        r"\midrule",
    ]
    for outcome, label in labels:
        row = diagnostics[outcome]
        lines.append(
            f"{label} & {float(row['effect']):.3f} & "
            f"[{float(row['ci_low']):.3f}, "
            f"{float(row['ci_high']):.3f}] & "
            f"{float(row['pretrend_slope']):.3f}"
            + r" \\"
        )
    aggregate = summary["aggregate_synthetic_control"]
    lines.extend(
        [
            r"\midrule",
            (
                "Effective controls & "
                f"{float(aggregate['effective_controls']):.1f} "
                r"& \multicolumn{2}{l}{maximum weight "
                f"{100 * float(aggregate['maximum_control_weight']):.2f}"
                r"\%} \\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Real Lemmy lock and removal interventions over 503 "
                r"non-reused risk-set pairs. Shared control weights use only "
                r"21 pre-intervention trajectory features. Intervals are "
                r"fixed-weight unit-cluster Bayesian bootstrap intervals.}"
            ),
            r"\label{tab:lemmy-intervention}",
            rf"\end{{{table_environment}}}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_cross_community_collapse_table(
    root: Path,
    output_path: Path,
    *,
    wide: bool = False,
) -> None:
    community_path = (
        root
        / "run_outputs"
        / "paper_exact_full"
        / "paper_tables"
        / "table2_cross_community_collapse.csv"
    )
    aggregate_path = (
        root
        / "run_outputs"
        / "paper_exact_full"
        / "paper_tables"
        / "table1_aggregate_structural_effects.csv"
    )
    communities = pd.read_csv(community_path).set_index("community")
    aggregate = pd.read_csv(aggregate_path).set_index("metric")
    order = ("worldnews", "science", "funny", "aww", "AskReddit")
    labels = {
        "worldnews": "WorldNews",
        "science": "Science",
        "funny": "Funny",
        "aww": "AWW",
        "AskReddit": "AskReddit",
    }
    environment = "table*" if wide else "table"
    lines = [
        rf"\begin{{{environment}}}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"Community & Baseline volume & Conflict volume & Amplification "
            r"& Baseline leaf depth & Conflict leaf depth & $\Delta$ leaf depth \\"
        ),
        r"\midrule",
    ]
    for community in order:
        row = communities.loc[community]
        lines.append(
            f"{labels[community]} & "
            f"{row['baseline_volume']:.1f} & "
            f"{row['toxic_volume']:.1f} & "
            f"{row['volume_amplification']:.2f}$\\times$ & "
            f"{row['baseline_mean_leaf_depth']:.2f} & "
            f"{row['toxic_mean_leaf_depth']:.2f} & "
            f"{row['delta_mean_leaf_depth']:.2f} \\\\"
        )
    volume = aggregate.loc["comment_volume"]
    leaf = aggregate.loc["mean_leaf_depth"]
    lines.extend(
        [
            r"\midrule",
            (
                r"\textbf{Pooled} & "
                f"\\textbf{{{volume['baseline_mean']:.1f}}} & "
                f"\\textbf{{{volume['toxic_controversial_mean']:.1f}}} & "
                f"\\textbf{{{volume['change_ratio']:.2f}$\\times$}} & "
                f"\\textbf{{{leaf['baseline_mean']:.2f}}} & "
                f"\\textbf{{{leaf['toxic_controversial_mean']:.2f}}} & "
                f"\\textbf{{{leaf['delta']:.2f}}} \\\\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            (
                r"\caption{Cross-community structural response under the paired "
                r"conflict-oriented Core and controversy-ranking intervention. "
                r"Each community contains 300 baseline and 300 intervention runs.}"
            ),
            r"\label{tab:cross-community-collapse}",
            rf"\end{{{environment}}}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_result_tables(
    root: Path,
    output_dir: Path,
    *,
    wide: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "table_cross_community_collapse.tex": _write_cross_community_collapse_table,
        "table_fidelity_ranking.tex": _write_fidelity_table,
        "table_hn_structure.tex": _write_hn_structure_table,
        "table_ablation_summary.tex": _write_ablation_table,
        "table_mechanism_phase_map.tex": _write_mechanism_phase_table,
        "table_hn_adapter_curve.tex": _write_hn_adapter_curve_table,
        "table_lemmy_intervention.tex": _write_lemmy_intervention_table,
    }
    for name, writer in outputs.items():
        writer(root, output_dir / name, wide=wide)
    paper_ready_dir = root / "manuscript" / "paper_ready_tables"
    copied_outputs = (
        "table_main_evidence.tex",
        "table_factorial_mechanism.tex",
        "table_robustness.tex",
        "table_heldout_prediction.tex",
        "table_fidelity_transfer.tex",
        "table_semantic_realdata.tex",
        "table_evidence_boundaries.tex",
        "table_reddit_expansion.tex",
        "table_lemmy_intervention_fidelity.tex",
        "table_lemmy_agent_replay.tex",
        "table_lemmy_content_matched.tex",
        "table_tbbt_qualified_controls.tex",
        "table_tbbt_intervention_fidelity.tex",
        "tbbt_intervention_fidelity_macros.tex",
        "table_api_model_robustness.tex",
        "api_model_robustness_macros.tex",
    )
    copied_names: list[str] = []
    for name in copied_outputs:
        source = paper_ready_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)
            copied_names.append(name)
    source_paths = [
        root / "artifacts/evaluation/model_ranking.csv",
        root / "artifacts/external_validation/cross_community/evaluation/model_ranking.csv",
        root / "artifacts/external_validation/cross_platform_transfer/evaluation/fidelity_distances.csv",
        root / "artifacts/external_validation/cross_platform_transfer/empirical_metrics.parquet",
        root / "artifacts/external_validation/cross_platform_transfer/simulated_metrics.parquet",
        root / "artifacts/runs/ablations/tradeoff_regions.csv",
        root / "artifacts/reviewer_validation/mechanism_phase_map/confirmatory_local_scenarios.csv",
        root / "artifacts/reviewer_validation/mechanism_phase_map/broad_phase_cells.csv",
        root / "artifacts/reviewer_validation/mechanism_phase_map/mechanism_phase_overview.pdf",
        root / "artifacts/external_validation/platform_adapter_curve/adapter_budget_curve.csv",
        root / "artifacts/external_validation/platform_adapter_curve/adapter_budget_curve.pdf",
        root
        / "artifacts/interventions/natural_confirmatory_lemmy_calibrated/"
        "natural_experiment_summary.json",
        root
        / "run_outputs/paper_exact_full/paper_tables/"
        "table2_cross_community_collapse.csv",
        root
        / "run_outputs/paper_exact_full/paper_tables/"
        "table1_aggregate_structural_effects.csv",
        root
        / "artifacts/interventions/tbbt_qualified_controls/"
        "tbbt_qualified_control_manifest.json",
        root
        / "artifacts/interventions/tbbt_qualified_controls/"
        "intervention_effects.csv",
        root
        / "artifacts/interventions/tbbt_intervention_fidelity/"
        "prediction_freeze_manifest.json",
        root
        / "artifacts/interventions/tbbt_intervention_fidelity/"
        "fidelity_summary.json",
        root
        / "artifacts/interventions/tbbt_intervention_fidelity/"
        "prediction_observed_comparison.csv",
        root / "artifacts/api/analysis/api_robustness_summary.json",
        root / "artifacts/api/intent_tasks.csv",
        root / "artifacts/api/frozen_intents_multimodel.jsonl",
        root / "artifacts/api/semantic_annotations_multimodel.jsonl",
        root / "artifacts/api/replay_metrics.csv",
        root / "artifacts/api/coupled_replay_metrics.csv",
    ]
    manifest = {
        "status": "complete",
        "layout": "two_column_wide" if wide else "single_column",
        "outputs": {
            name: sha256_file(output_dir / name)
            for name in (*outputs, *copied_names)
        },
        "sources": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in source_paths
        ],
    }
    write_json(output_dir / "results_tables.manifest.json", manifest)
    return manifest


def build_original_manuscript(root: Path) -> dict[str, Any]:
    source = _source_extract(root)
    text = _decode_extract(source)
    sanitized, audit = sanitize_original_extract(text)
    output_dir = root / "manuscript" / "original_reconstructed"
    output_dir.mkdir(parents=True, exist_ok=True)
    sanitized_path = output_dir / "source_extracted_sanitized.txt"
    sanitized_path.write_text(sanitized, encoding="utf-8")
    audit.update(
        {
            "source_path": str(source.resolve()),
            "source_sha256": sha256_file(source),
            "source_preserved": True,
            "tex_status": "reconstructed from extracted text; formulas and layout require comparison with original TeX/PDF",
        }
    )
    write_json(output_dir / "SOURCE_RECOVERY.json", audit)
    compile_result = compile_tex(output_dir / "paper.tex")
    result = {
        "status": "complete",
        "source_recovery": audit,
        "paper_tex_sha256": sha256_file(output_dir / "paper.tex"),
        "compile": compile_result,
    }
    write_json(output_dir / "BUILD_MANIFEST.json", result)
    return result


def build_reconstructed_revision_manuscript(root: Path) -> dict[str, Any]:
    original = build_original_manuscript(root)
    output_dir = root / "manuscript" / "revision_2026_v2"
    macros = build_result_macros(root, output_dir / "results_macros.tex")
    tables = build_result_tables(root, output_dir)
    compile_result = compile_tex(output_dir / "paper.tex")
    result = {
        "status": "complete",
        "base_original_sha256": original["paper_tex_sha256"],
        "paper_tex_sha256": sha256_file(output_dir / "paper.tex"),
        "result_macros": macros,
        "result_tables": tables,
        "compile": compile_result,
    }
    write_json(output_dir / "BUILD_MANIFEST.json", result)
    return result


def build_final_manuscript(root: Path) -> dict[str, Any]:
    source_pdf = (
        root
        / "manuscript"
        / "source_evidence"
        / "What_Makes_Content_Go_Vi_original.pdf"
    )
    if not source_pdf.is_file():
        return {
            "status": "source_missing",
            "source_pdf": str(source_pdf),
            "reason": "The user-designated original PDF has not been imported.",
        }

    output_dir = root / "manuscript" / "original_pdf_revision"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_sources = {
        "mechanism_phase_overview.pdf": (
            root
            / "artifacts/reviewer_validation/mechanism_phase_map/"
            "mechanism_phase_overview.pdf"
        ),
        "adapter_budget_curve.pdf": (
            root
            / "artifacts/external_validation/platform_adapter_curve/"
            "adapter_budget_curve.pdf"
        ),
    }
    for name, source in figure_sources.items():
        if source.is_file():
            shutil.copy2(source, figure_dir / name)
    macros = build_result_macros(root, output_dir / "results_macros.tex")
    tables = build_result_tables(root, output_dir, wide=True)
    compile_result = compile_tex(output_dir / "paper.tex")
    result = {
        "status": "complete" if compile_result.get("status") == "complete" else "failed",
        "base_original_pdf": source_pdf.relative_to(root).as_posix(),
        "base_original_pdf_sha256": sha256_file(source_pdf),
        "base_original_pdf_size": source_pdf.stat().st_size,
        "paper_tex_sha256": sha256_file(output_dir / "paper.tex"),
        "result_macros": macros,
        "result_tables": tables,
        "compile": compile_result,
    }
    fidelity_path = (
        root
        / "artifacts/interventions/tbbt_intervention_fidelity/"
        "fidelity_summary.json"
    )
    if fidelity_path.is_file():
        fidelity = json.loads(fidelity_path.read_text(encoding="utf-8"))
        result["tbbt_intervention_fidelity"] = {
            "status": fidelity["status"],
            "freeze_id": fidelity["freeze_id"],
            "n_interventions": fidelity["n_interventions"],
            "direction_accuracy": fidelity["direction_accuracy"],
            "mae_percentage_points": fidelity["mae_percentage_points"],
            "prediction_interval_coverage": fidelity[
                "observed_point_prediction_interval_coverage"
            ],
            "summary_sha256": sha256_file(fidelity_path),
        }
    api_robustness_path = (
        root / "artifacts/api/analysis/api_robustness_summary.json"
    )
    if api_robustness_path.is_file():
        api_robustness = json.loads(
            api_robustness_path.read_text(encoding="utf-8")
        )
        result["api_model_robustness"] = {
            "status": api_robustness["status"],
            "valid_calls": api_robustness["calls"]["total"],
            "reply_unanimity": api_robustness["intent_agreement"][
                "reply_unanimous"
            ]["mean"],
            "polarity_unanimity": api_robustness["intent_agreement"][
                "polarity_unanimous"
            ]["mean"],
            "toxicity_alpha": api_robustness["annotation_agreement"][
                "toxicity_interval_alpha"
            ],
            "strict_replay_maximum_range": api_robustness[
                "strict_rehydration"
            ]["maximum_structural_range"],
            "summary_sha256": sha256_file(api_robustness_path),
        }
    write_json(output_dir / "BUILD_MANIFEST.json", result)
    return result


def build_redline(root: Path) -> dict[str, Any]:
    build_reconstructed_revision_manuscript(root)
    original = root / "manuscript" / "original_reconstructed" / "paper.tex"
    revision = root / "manuscript" / "revision_2026_v2" / "paper.tex"
    output_dir = root / "manuscript" / "redline"
    output_dir.mkdir(parents=True, exist_ok=True)
    redline = output_dir / "paper_redline.tex"
    command_prefix = _latexdiff_command(root)
    if not command_prefix:
        result = {
            "status": "tool_unavailable",
            "resolved": False,
            "reason": "latexdiff or a Perl interpreter is not installed",
            "original_sha256": sha256_file(original),
            "revision_sha256": sha256_file(revision),
        }
        write_json(output_dir / "BUILD_MANIFEST.json", result)
        return result
    completed = subprocess.run(
        [*command_prefix, "--type=CFONT", str(original), str(revision)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=300,
    )
    redline.write_text(completed.stdout, encoding="utf-8")
    for support_name in (
        "results_macros.tex",
        "table_fidelity_ranking.tex",
        "table_hn_structure.tex",
        "table_ablation_summary.tex",
        "table_mechanism_phase_map.tex",
        "table_hn_adapter_curve.tex",
        "references.bib",
    ):
        support_file = revision.parent / support_name
        if support_file.is_file():
            shutil.copy2(support_file, output_dir / support_name)
    compile_result = compile_tex(redline)
    complete = completed.returncode == 0 and compile_result.get("status") == "complete"
    result = {
        "status": "complete" if complete else "failed",
        "redline_style": "CFONT",
        "latexdiff_returncode": completed.returncode,
        "stderr_tail": completed.stderr[-2000:],
        "redline_sha256": sha256_file(redline),
        "compile": compile_result,
    }
    write_json(output_dir / "BUILD_MANIFEST.json", result)
    return result


def _latexdiff_command(root: Path) -> list[str]:
    installed = shutil.which("latexdiff")
    if installed:
        return [installed]
    script_candidates = [
        root / "vendor" / "latexdiff" / "latexdiff-so",
        root / ".tools" / "latexdiff" / "latexdiff-so",
        root / ".tools" / "latexdiff" / "latexdiff",
        root / ".tools" / "latexdiff" / "source" / "latexdiff" / "latexdiff-so",
        root / ".tools" / "latexdiff" / "source" / "latexdiff" / "latexdiff",
    ]
    perl_candidates = [
        Path(value)
        for value in (
            shutil.which("perl"),
            root / ".tools" / "perl" / "runtime" / "perl" / "bin" / "perl.exe",
            root / ".tools" / "perl" / "perl" / "bin" / "perl.exe",
        )
        if value
    ]
    script = next((path for path in script_candidates if path.is_file()), None)
    perl = next((path for path in perl_candidates if path.is_file()), None)
    return [str(perl), str(script)] if script and perl else []


def compile_tex(tex_path: Path) -> dict[str, Any]:
    output_dir = tex_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    repository_root = tex_path.parents[2]
    bundled_bst = repository_root / "manuscript" / "plain.bst"
    local_bst = tex_path.parent / "plain.bst"
    if bundled_bst.is_file() and not local_bst.is_file():
        shutil.copy2(bundled_bst, local_bst)
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    bundled_tectonic = repository_root / ".tools" / "tectonic" / "tectonic.exe"
    tectonic = shutil.which("tectonic") or (str(bundled_tectonic) if bundled_tectonic.is_file() else "")
    if latexmk:
        command = [latexmk, "-pdf", "-interaction=nonstopmode", f"-outdir={output_dir}", tex_path.name]
    elif pdflatex:
        command = [pdflatex, "-interaction=nonstopmode", f"-output-directory={output_dir}", tex_path.name]
    elif tectonic:
        command = [tectonic, "-o", str(output_dir), tex_path.name]
    else:
        return {
            "status": "tool_unavailable",
            "reason": "latexmk, pdflatex, and tectonic are not installed",
        }
    completed = subprocess.run(
        command,
        cwd=tex_path.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=600,
    )
    pdf = output_dir / f"{tex_path.stem}.pdf"
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    return {
        "status": "complete" if completed.returncode == 0 and pdf.is_file() else "failed",
        "returncode": completed.returncode,
        "pdf": str(pdf) if pdf.is_file() else "",
        "pdf_sha256": sha256_file(pdf) if pdf.is_file() else "",
        "log_tail": (stdout + "\n" + stderr)[-4000:],
    }


def run_manuscript_command(root: Path, command: str) -> dict[str, Any]:
    if command == "build-original-manuscript":
        return build_original_manuscript(root)
    if command == "build-final-paper":
        return build_final_manuscript(root)
    if command == "build-paper-redline":
        return build_redline(root)
    raise ValueError(f"Unknown manuscript command: {command}")
