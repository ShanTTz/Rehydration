from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.reviewer_semantics import (
    build_semantic_consensus,
    fit_semantic_scorer,
    run_semantic_pattern_analysis,
    score_early_threads,
    summarize_semantic_api_usage,
    summarize_semantic_model_agreement,
)
from bdmtf.revision.provenance import sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate semantic measurement and rerun held-out real-pattern tests."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "reviewer_semantic_calibration.json"),
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT / "artifacts" / "reviewer_validation" / "semantic_calibration"
        ),
    )
    parser.add_argument(
        "--base-panel",
        default=str(
            ROOT
            / "artifacts"
            / "reviewer_validation"
            / "real_patterns"
            / "early_late_panel.parquet"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    cache = output / "semantic_annotations_multimodel.jsonl"
    if not cache.exists():
        raise SystemExit(
            f"Missing API cache: {cache}. Run prepare_reviewer_semantics.py --execute first."
        )
    consensus, consensus_manifest = build_semantic_consensus(
        cache,
        output / "selected_semantic_comments.parquet",
        int(config["annotation"]["require_distinct_families"]),
    )
    if consensus_manifest["status"] != "complete":
        write_json(output / "semantic_consensus_manifest.json", consensus_manifest)
        raise SystemExit(
            "Semantic cache is incomplete; no scorer or outcome analysis was run."
        )
    consensus_path = output / "semantic_consensus.parquet"
    consensus_csv_path = output / "semantic_consensus.csv"
    consensus.to_parquet(consensus_path, index=False)
    consensus.to_csv(consensus_csv_path, index=False)
    write_json(output / "semantic_consensus_manifest.json", consensus_manifest)

    agreement, agreement_manifest = summarize_semantic_model_agreement(cache)
    agreement_path = output / "semantic_model_agreement.csv"
    agreement.to_csv(agreement_path, index=False)
    usage = summarize_semantic_api_usage(cache)
    usage_path = output / "semantic_api_usage.csv"
    usage.to_csv(usage_path, index=False)

    model_path = output / "semantic_scorer.joblib"
    scorer_manifest = fit_semantic_scorer(consensus, model_path)
    write_json(output / "semantic_scorer_manifest.json", scorer_manifest)

    splits = pd.read_csv(ROOT / "artifacts" / "splits" / "post_splits.csv")
    thread_scores = score_early_threads(
        ROOT / "data" / "social_paper",
        splits,
        model_path,
        float(config["sampling"]["early_cutoff_minutes"]),
    )
    scores_path = output / "early_thread_semantic_scores.parquet"
    thread_scores.to_parquet(scores_path, index=False)
    results, analysis_manifest, panel = run_semantic_pattern_analysis(
        args.base_panel,
        thread_scores,
        args.bootstrap_samples,
        int(config.get("seed", 30371)),
    )
    results_path = output / "semantic_heldout_pattern_results.csv"
    panel_path = output / "semantic_early_late_panel.parquet"
    results.to_csv(results_path, index=False)
    panel.to_parquet(panel_path, index=False)
    final_manifest = {
        "schema_version": 1,
        "status": "complete",
        "consensus": consensus_manifest,
        "agreement": agreement_manifest,
        "scorer": scorer_manifest,
        "heldout_analysis": analysis_manifest,
        "files": {
            "consensus": str(consensus_path),
            "consensus_sha256": sha256_file(consensus_path),
            "consensus_csv": str(consensus_csv_path),
            "consensus_csv_sha256": sha256_file(consensus_csv_path),
            "agreement": str(agreement_path),
            "agreement_sha256": sha256_file(agreement_path),
            "api_usage": str(usage_path),
            "api_usage_sha256": sha256_file(usage_path),
            "thread_scores": str(scores_path),
            "thread_scores_sha256": sha256_file(scores_path),
            "results": str(results_path),
            "results_sha256": sha256_file(results_path),
            "analysis_panel": str(panel_path),
            "analysis_panel_sha256": sha256_file(panel_path),
        },
    }
    write_json(output / "semantic_analysis_manifest.json", final_manifest)
    print(json.dumps(final_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
