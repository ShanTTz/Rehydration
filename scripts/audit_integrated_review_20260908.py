"""Read-only input audit for the integrated-paper review; no simulation or API calls."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bdmtf.revision.lemmy_path_fidelity import (  # noqa: E402
    _empty_path,
    _observed_path,
    _path_errors,
    _simulated_path,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    paper = ROOT / "manuscript/iclr2027_overleaf_package_integrated_20260907"
    paths = {
        "paper": paper / "paper.pdf",
        "matches": ROOT / "data/external/lemmy/outcomes_confirmatory/risk_set_matches.csv",
        "splits": ROOT / "artifacts/interventions/fidelity/intervention_splits.csv",
        "events": ROOT / "data/external/lemmy/outcomes_confirmatory/lemmy_thread_events.parquet",
        "predictions": ROOT / "artifacts/interventions/path_fidelity/path_calibrated_simulated_events.parquet",
        "scores": ROOT / "artifacts/interventions/path_fidelity/path_intervention_scores.csv",
        "manifest": ROOT / "artifacts/interventions/path_fidelity/path_fidelity_manifest.json",
        "config": ROOT / "configs/lemmy_agent_intervention_replay.json",
    }
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    horizon = config["simulation"]["horizon_days"]
    seeds = config["simulation"]["test_seeds"]
    scales = manifest["scales_fitted_on_training_only"]
    matches = pd.read_csv(paths["matches"])
    splits = pd.read_csv(paths["splits"])
    matches = matches.merge(splits[["intervention_id", "split"]], on="intervention_id", validate="one_to_one")
    selected = matches[matches["split"] == "test"].copy()
    selected["event_time"] = pd.to_datetime(selected["event_time"], utc=True, format="mixed")
    events = pd.read_parquet(paths["events"])
    events["created_at"] = pd.to_datetime(events["created_at"], utc=True, format="mixed")
    groups = {str(key): group for key, group in events.groupby("content_id")}
    predicted = pd.read_parquet(paths["predictions"])
    rows = []
    for row in selected.itertuples(index=False):
        observed = _observed_path(groups, str(row.treated_content_id), row.event_time, horizon)
        zero_scores = _path_errors(
            observed, _empty_path(horizon), horizon_days=horizon,
            count_scale=scales["cumulative_count_q95"], depth_scale=scales["depth_q95"],
        )
        draws = [_simulated_path(predicted, str(row.intervention_id), seed, horizon) for seed in seeds]
        rows.append({
            "intervention_id": row.intervention_id,
            "intervention_type": row.intervention_type,
            "observed_events_in_saved_window": len(observed.times),
            "nonempty_predicted_draws": sum(len(draw.times) > 0 for draw in draws),
            "both_nonempty_draws": sum(len(draw.times) > 0 and len(observed.times) > 0 for draw in draws),
            "always_empty_composite_error": zero_scores["composite_path_error"],
        })
    detail = pd.DataFrame(rows)
    scores = pd.read_csv(paths["scores"])
    composite = scores[scores["metric"] == "composite_path_error"].pivot(
        index="intervention_id", columns="model", values="error",
    )
    detail = detail.merge(composite, on="intervention_id", validate="one_to_one")
    detail["bdmtf_minus_always_empty_error"] = (
        detail["bdmtf_training_path"] - detail["always_empty_composite_error"]
    )
    rng = np.random.default_rng(20260908)
    bootstrap_sum = np.zeros(10000)
    for _, group in detail.groupby("intervention_type"):
        values = group["bdmtf_minus_always_empty_error"].to_numpy()
        bootstrap_sum += values[rng.integers(len(values), size=(10000, len(values)))].sum(axis=1)
    interval = np.quantile(bootstrap_sum / len(detail), [0.025, 0.975]).tolist()
    by_type = detail.groupby("intervention_type").agg(
        interventions=("intervention_id", "count"),
        observed_nonempty=("observed_events_in_saved_window", lambda values: int((values > 0).sum())),
        observed_events=("observed_events_in_saved_window", "sum"),
        bdmtf_error=("bdmtf_training_path", "mean"),
        empirical_error=("empirical_nearest_path", "mean"),
        always_empty_error=("always_empty_composite_error", "mean"),
        both_nonempty_draws=("both_nonempty_draws", "sum"),
    ).reset_index()
    labels = Counter(re.findall(r"\\newlabel\{([^}]+)\}", (paper / "paper.aux").read_text()))
    hashes_after = {name: sha256(path) for name, path in paths.items()}
    assert hashes_after == hashes_before, "Review inputs changed during the audit"
    result = {
        "kind": "recomputed_review_diagnostic_not_a_new_simulation",
        "paper_pdf_sha256": hashes_before["paper"],
        "input_sha256": hashes_before,
        "inputs_unchanged": True,
        "test_interventions": len(detail),
        "horizon_days": horizon,
        "observed_empty_interventions": int((detail["observed_events_in_saved_window"] == 0).sum()),
        "observed_total_events": int(detail["observed_events_in_saved_window"].sum()),
        "predicted_draws": len(detail) * len(seeds),
        "nonempty_predicted_draws": int(detail["nonempty_predicted_draws"].sum()),
        "both_nonempty_draws": int(detail["both_nonempty_draws"].sum()),
        "bdmtf_composite_error": float(detail["bdmtf_training_path"].mean()),
        "empirical_composite_error": float(detail["empirical_nearest_path"].mean()),
        "always_empty_composite_error": float(detail["always_empty_composite_error"].mean()),
        "bdmtf_minus_always_empty_error": float(detail["bdmtf_minus_always_empty_error"].mean()),
        "difference_95pct_stratified_intervention_bootstrap": interval,
        "duplicate_latex_labels": {key: count for key, count in labels.items() if count > 1},
        "interpretation": "Counts refer to saved observed event windows. Missingness versus genuine silence requires separate data-collection verification.",
    }
    output = ROOT / "artifacts/reviewer_validation/review_20260908"
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "lemmy_empty_path_diagnostic.csv", index=False)
    by_type.to_csv(output / "lemmy_empty_path_by_type.csv", index=False)
    (output / "review_diagnostic.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()
