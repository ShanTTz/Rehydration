from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import energy_distance, ks_2samp, wasserstein_distance

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bdmtf.revision.data_pipeline import METRIC_COLUMNS
from bdmtf.revision.fitted_models import CommunityModel
from bdmtf.revision.provenance import run_manifest, write_json
from bdmtf.revision.simulation import MODEL_NAMES, RevisionSimulationConfig, event_metrics, nodes_to_records, simulate_cascade


def run_model_suite(
    empirical_metrics: pd.DataFrame,
    models: dict[str, CommunityModel],
    output_dir: Path,
    config: RevisionSimulationConfig,
    model_names: Iterable[str] = MODEL_NAMES,
    seeds: Iterable[int] = (0,),
    max_posts_per_community: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = empirical_metrics[empirical_metrics["split"] == "train"]
    test = empirical_metrics[empirical_metrics["split"] == "test"]
    metric_records: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    for community, test_rows in test.groupby("community"):
        test_rows = test_rows.sort_values("post_id")
        if max_posts_per_community > 0:
            test_rows = test_rows.head(max_posts_per_community)
        train_rows = train[train["community"] == community]
        for test_row in test_rows.itertuples(index=False):
            for model_name in model_names:
                for seed in seeds:
                    run_seed = int(seed) + _stable_seed(str(test_row.post_id), str(model_name))
                    nodes = simulate_cascade(model_name, models[community], train_rows, str(test_row.post_id), run_seed, config)
                    context = {
                        "community": community,
                        "post_id": str(test_row.post_id),
                        "model": model_name,
                        "seed": int(seed),
                        "split": "test",
                    }
                    metric_records.append({**context, **event_metrics(nodes)})
                    event_records.extend(nodes_to_records(nodes, context))
    metrics = pd.DataFrame(metric_records)
    events = pd.DataFrame(event_records)
    metrics.to_csv(output_dir / "simulated_metrics.csv", index=False)
    metrics.to_parquet(output_dir / "simulated_metrics.parquet", index=False)
    events.to_parquet(output_dir / "events.parquet", index=False)
    write_json(output_dir / "run_manifest.json", run_manifest("run-model-suite", {"simulation": config.__dict__, "models": list(model_names), "seeds": list(seeds)}))
    return metrics, events


def _stable_seed(*parts: str) -> int:
    import hashlib

    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def evaluate_fidelity(empirical: pd.DataFrame, simulated: pd.DataFrame, output_dir: Path, bootstrap_samples: int = 400, seed: int = 30371) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    test = empirical[empirical["split"] == "test"]
    for (model, community), simulation_group in simulated.groupby(["model", "community"]):
        real_group = test[test["community"] == community]
        for metric in METRIC_COLUMNS:
            if metric not in simulation_group or metric not in real_group:
                continue
            real = real_group[metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            generated = simulation_group[metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            if not len(real) or not len(generated):
                continue
            scale = max(float(np.std(real)), abs(float(np.mean(real))) * 0.1, 1e-9)
            normalized = float(wasserstein_distance(real, generated) / scale)
            boot = []
            for _ in range(bootstrap_samples):
                a = rng.choice(real, size=len(real), replace=True)
                b = rng.choice(generated, size=len(generated), replace=True)
                boot.append(wasserstein_distance(a, b) / scale)
            ks = ks_2samp(real, generated)
            rows.append(
                {
                    "model": model,
                    "community": community,
                    "metric": metric,
                    "n_empirical": len(real),
                    "n_simulated": len(generated),
                    "empirical_mean": float(real.mean()),
                    "simulated_mean": float(generated.mean()),
                    "wasserstein": float(wasserstein_distance(real, generated)),
                    "normalized_wasserstein": normalized,
                    "normalized_wasserstein_ci_low": float(np.quantile(boot, 0.025)),
                    "normalized_wasserstein_ci_high": float(np.quantile(boot, 0.975)),
                    "ks_statistic": float(ks.statistic),
                    "ks_pvalue": float(ks.pvalue),
                    "energy_distance": float(energy_distance(real, generated)),
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "fidelity_distances.csv", index=False)
    aggregate = results.groupby(["model", "metric"])["normalized_wasserstein"].agg(["mean", "median", "std", "count"]).reset_index()
    aggregate.to_csv(output_dir / "fidelity_summary.csv", index=False)
    _plot_fidelity(aggregate, output_dir / "fidelity_by_model.png")
    _plot_volume_depth(test, simulated, output_dir / "volume_depth_holdout.png")
    return results


def _plot_fidelity(summary: pd.DataFrame, path: Path) -> None:
    pivot = summary.pivot(index="metric", columns="model", values="median")
    ax = pivot.plot(kind="bar", figsize=(14, 7), width=0.85)
    ax.set_ylabel("Median normalized Wasserstein distance (lower is better)")
    ax.set_xlabel("")
    ax.legend(fontsize=8, ncol=2)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=180)
    plt.close(ax.figure)


def _plot_volume_depth(empirical: pd.DataFrame, simulated: pd.DataFrame, path: Path) -> None:
    figure, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(empirical["size"], empirical["mean_leaf_depth"], alpha=0.65, label="empirical test", color="#202020")
    for model, group in simulated.groupby("model"):
        ax.scatter(group["size"], group["mean_leaf_depth"], alpha=0.35, s=20, label=model)
    ax.set_xscale("log")
    ax.set_xlabel("Cascade size (log scale)")
    ax.set_ylabel("Mean leaf depth")
    ax.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def summarize_model_ranking(fidelity: pd.DataFrame) -> pd.DataFrame:
    return fidelity.groupby("model")["normalized_wasserstein"].agg(["mean", "median", "std", "count"]).sort_values("median").reset_index()
