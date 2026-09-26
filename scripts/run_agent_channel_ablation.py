from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.data.social_loader import (  # noqa: E402
    load_comments,
    load_population,
    load_posts,
    resolve_community_paths,
    select_posts,
)
from bdmtf.experiments import (  # noqa: E402
    _post_early_engagement_signal,
    _stable_seed,
    _with_community_calibration,
    config_from_dict,
    interventions_from_config,
)
from bdmtf.intent_pool import FrozenIntentPool  # noqa: E402
from bdmtf.metrics import compute_metrics  # noqa: E402
from bdmtf.simulator import BDMTFSimulator  # noqa: E402


SCENARIOS: dict[str, dict[str, Any]] = {
    "reference": {
        "label": "All channels",
        "changes": {},
        "definition": "Frozen principal configuration with all mechanisms enabled.",
    },
    "no_conflict": {
        "label": "No conflict channel",
        "changes": {"enable_conflict_channel": False},
        "definition": (
            "Removes conflict-conditioned activation, polarity, reply, target, "
            "and external-feedback paths."
        ),
    },
    "no_heat": {
        "label": "No heat channel",
        "changes": {"enable_heat_channel": False},
        "definition": "Removes heat-conditioned activation, reply, and target paths.",
    },
    "no_depth_preference": {
        "label": "No depth exponent",
        "changes": {
            "constructive_depth_lambda": 0.0,
            "antagonistic_depth_lambda": 0.0,
            "baseline_depth_lambda_floor": 0.0,
            "toxic_depth_lambda_floor": 0.0,
        },
        "definition": (
            "Sets the explicit depth-preference exponent to zero while retaining "
            "the separately declared fatigue, collapse, root, and cap mechanisms."
        ),
    },
    "no_depth_targeting": {
        "label": "No complete depth channel",
        "changes": {"enable_depth_targeting": False},
        "definition": (
            "Removes explicit depth exponents, fatigue, collapse, root bias, "
            "and depth caps; ranking still determines the visible set."
        ),
    },
}

BASELINE = "BASELINE"
TREATED = "CORE_TOXIC_CONTROVERSIAL"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired mechanism-deletion ablations for the Agent Modeling section."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "paper_exact_reproduction.json"),
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT / "artifacts" / "reviewer_validation" / "agent_channel_ablation"
        ),
    )
    parser.add_argument("--posts-per-community", type=int, default=5)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument(
        "--paper-output",
        default=str(
            ROOT
            / "manuscript"
            / "iclr2026_revision_staging"
            / "generated"
            / "table_agent_channel_ablation.tex"
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    source = json.loads(config_path.read_text(encoding="utf-8"))
    source["conditions"] = [
        item
        for item in source["conditions"]
        if item["name"] in {BASELINE, TREATED}
    ]
    if {item["name"] for item in source["conditions"]} != {BASELINE, TREATED}:
        raise ValueError("Source config does not contain both ablation conditions")

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "runs.jsonl"
    completed = _completed_keys(records_path)
    expected = (
        len(SCENARIOS)
        * len(source["communities"])
        * args.posts_per_community
        * len(seeds)
        * 2
    )
    _write_status(output, expected, len(completed), "starting")

    started = time.time()
    new_runs = 0
    with records_path.open("a", encoding="utf-8") as handle:
        for community in source["communities"]:
            paths = resolve_community_paths(
                ROOT / "data" / "social_paper", community
            )
            posts = select_posts(
                load_posts(paths, enriched=True),
                args.posts_per_community,
                seed=0,
            )
            comments = load_comments(paths)
            frozen_path = paths.base_dir / "frozen_intents.jsonl"
            pool_template = (
                FrozenIntentPool.from_jsonl(frozen_path)
                if frozen_path.exists()
                else FrozenIntentPool.from_comments(comments, seed=0)
            )

            for scenario, definition in SCENARIOS.items():
                raw = copy.deepcopy(source)
                raw["simulation"].update(definition["changes"])
                config = _with_community_calibration(
                    config_from_dict(raw), community
                )
                interventions = interventions_from_config(raw)
                for seed in seeds:
                    agents = load_population(paths, config, seed=seed)
                    for post in posts.itertuples(index=False):
                        post_id = str(getattr(post, "post_id"))
                        sim_seed = _stable_seed(
                            "agent_channel_ablation",
                            community,
                            post_id,
                            seed,
                        )
                        run_config = replace(
                            config,
                            initial_engagement_signal=_post_early_engagement_signal(post),
                        )
                        for intervention in interventions:
                            key = (
                                scenario,
                                community,
                                post_id,
                                seed,
                                intervention.name,
                            )
                            if key in completed:
                                continue
                            simulator = BDMTFSimulator(
                                agents,
                                copy.deepcopy(pool_template),
                                config=run_config,
                                seed=sim_seed,
                            )
                            state, traces = simulator.run(
                                post_id=post_id,
                                title=str(getattr(post, "title", "")),
                                initial_text=str(getattr(post, "full_text", "")),
                                intervention=intervention,
                            )
                            record: dict[str, Any] = {
                                "scenario": scenario,
                                "scenario_label": definition["label"],
                                "community": community,
                                "post_id": post_id,
                                "seed": seed,
                                "condition": intervention.name,
                                "core": intervention.core,
                                "ranking": intervention.ranking.value,
                                "enable_conflict_channel": (
                                    run_config.enable_conflict_channel
                                ),
                                "enable_heat_channel": run_config.enable_heat_channel,
                                "enable_depth_targeting": (
                                    run_config.enable_depth_targeting
                                ),
                            }
                            record.update(compute_metrics(state, traces))
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            completed.add(key)
                            new_runs += 1
                            if new_runs % args.checkpoint_every == 0:
                                handle.flush()
                                _write_status(
                                    output,
                                    expected,
                                    len(completed),
                                    "running",
                                    time.time() - started,
                                )
                                print(f"checkpoint: {len(completed)}/{expected}")

    state = "complete" if len(completed) == expected else "incomplete"
    _write_status(output, expected, len(completed), state, time.time() - started)
    if state != "complete":
        raise RuntimeError(f"Ablation run incomplete: {len(completed)}/{expected}")

    summary = _analyze(records_path, args.bootstrap)
    summary.to_csv(output / "channel_ablation_summary.csv", index=False)
    latex_path = output / "table_agent_channel_ablation.tex"
    _write_latex(summary, latex_path)
    paper_output = Path(args.paper_output)
    paper_output.parent.mkdir(parents=True, exist_ok=True)
    paper_output.write_text(latex_path.read_text(encoding="utf-8"), encoding="utf-8")
    _write_report(summary, output / "AGENT_CHANNEL_ABLATION_REPORT.md")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "design": {
            "communities": source["communities"],
            "posts_per_community": args.posts_per_community,
            "seeds": seeds,
            "conditions": [BASELINE, TREATED],
            "pairing": "same post, seed, frozen intent pool, and simulator seed",
            "bootstrap": "post-level stratified by community",
            "bootstrap_replicates": args.bootstrap,
        },
        "scenarios": SCENARIOS,
        "source_config": str(config_path.relative_to(ROOT)),
        "source_config_sha256": _sha256(config_path),
        "records_sha256": _sha256(records_path),
        "summary_sha256": _sha256(output / "channel_ablation_summary.csv"),
        "code_scope": (
            "Mechanism deletion around the legacy paper-aligned configuration; "
            "not independent parameter calibration."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


def _analyze(records_path: Path, bootstrap_replicates: int) -> pd.DataFrame:
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frame = pd.DataFrame(records)
    keys = ["scenario", "community", "post_id", "seed"]
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(keys, sort=True):
        by_condition = group.set_index("condition")
        if BASELINE not in by_condition.index or TREATED not in by_condition.index:
            continue
        baseline = by_condition.loc[BASELINE]
        treated = by_condition.loc[TREATED]
        rows.append(
            {
                **dict(zip(keys, key)),
                "log_volume_effect": math.log1p(float(treated["comment_volume"]))
                - math.log1p(float(baseline["comment_volume"])),
                "leaf_depth_delta": float(treated["mean_leaf_depth"])
                - float(baseline["mean_leaf_depth"]),
                "max_depth_delta": float(treated["max_depth"])
                - float(baseline["max_depth"]),
            }
        )
    blocks = pd.DataFrame(rows)
    blocks["shallow_swarm"] = (blocks["log_volume_effect"] > 0) & (
        blocks["leaf_depth_delta"] < 0
    )

    summaries: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        group = blocks[blocks["scenario"] == scenario]
        if group.empty:
            continue
        post_level = (
            group.groupby(["community", "post_id"], as_index=False)[
                [
                    "log_volume_effect",
                    "leaf_depth_delta",
                    "max_depth_delta",
                    "shallow_swarm",
                ]
            ]
            .mean()
        )
        rng = np.random.default_rng(
            int(hashlib.sha256(scenario.encode()).hexdigest()[:8], 16)
        )
        boot_log: list[float] = []
        boot_leaf: list[float] = []
        communities = list(post_level.groupby("community", sort=True))
        for _ in range(bootstrap_replicates):
            samples = []
            for _, community_rows in communities:
                draw = rng.integers(0, len(community_rows), len(community_rows))
                samples.append(community_rows.iloc[draw])
            sampled = pd.concat(samples, ignore_index=True)
            boot_log.append(float(sampled["log_volume_effect"].mean()))
            boot_leaf.append(float(sampled["leaf_depth_delta"].mean()))

        mean_log = float(post_level["log_volume_effect"].mean())
        leaf_delta = float(post_level["leaf_depth_delta"].mean())
        definition = SCENARIOS[scenario]
        summaries.append(
            {
                "scenario": scenario,
                "label": definition["label"],
                "n_posts": int(len(post_level)),
                "n_blocks": int(len(group)),
                "geometric_volume_ratio": math.exp(mean_log),
                "volume_ratio_ci_low": math.exp(float(np.quantile(boot_log, 0.025))),
                "volume_ratio_ci_high": math.exp(float(np.quantile(boot_log, 0.975))),
                "mean_leaf_depth_delta": leaf_delta,
                "leaf_depth_ci_low": float(np.quantile(boot_leaf, 0.025)),
                "leaf_depth_ci_high": float(np.quantile(boot_leaf, 0.975)),
                "mean_max_depth_delta": float(post_level["max_depth_delta"].mean()),
                "shallow_swarm_block_share": float(group["shallow_swarm"].mean()),
                "supports_shallow_swarm": bool(mean_log > 0 and leaf_delta < 0),
            }
        )
    return pd.DataFrame(summaries)


def _write_latex(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Mechanism-deletion ablations under paired frozen-intent replay. "
        r"Intervals are 95\% post-level stratified bootstrap intervals.}",
        r"\label{tab:agent-channel-ablation}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Setting & Posts & Blocks & Volume ratio [95\% CI] & "
        r"$\Delta$ leaf depth [95\% CI] & Shallow Swarm \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        support = "Yes" if row.supports_shallow_swarm else "No"
        lines.append(
            f"{row.label} & {row.n_posts} & {row.n_blocks} & "
            f"{row.geometric_volume_ratio:.3f} "
            f"[{row.volume_ratio_ci_low:.3f}, {row.volume_ratio_ci_high:.3f}] & "
            f"{row.mean_leaf_depth_delta:.3f} "
            f"[{row.leaf_depth_ci_low:.3f}, {row.leaf_depth_ci_high:.3f}] & "
            f"{support} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Agent Channel Ablation",
        "",
        "This diagnostic removes complete dynamic paths while holding posts, "
        "frozen intent pools, conditions, and paired simulator seeds fixed.",
        "",
        "| Setting | Volume ratio (95% CI) | Delta leaf depth (95% CI) | Support |",
        "|---|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        support = "Yes" if row.supports_shallow_swarm else "No"
        lines.append(
            f"| {row.label} | {row.geometric_volume_ratio:.3f} "
            f"[{row.volume_ratio_ci_low:.3f}, {row.volume_ratio_ci_high:.3f}] | "
            f"{row.mean_leaf_depth_delta:.3f} "
            f"[{row.leaf_depth_ci_low:.3f}, {row.leaf_depth_ci_high:.3f}] | "
            f"{support} |"
        )
    lines.extend(
        [
            "",
            "The experiment is a mechanism deletion around the frozen legacy "
            "paper-aligned configuration. It tests which implemented pathways are "
            "necessary for the simulated response; it is not an independent fit to "
            "real-platform outcomes.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _completed_keys(path: Path) -> set[tuple[str, str, str, int, str]]:
    completed: set[tuple[str, str, str, int, str]] = set()
    if not path.is_file():
        return completed
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        key = (
            str(record["scenario"]),
            str(record["community"]),
            str(record["post_id"]),
            int(record["seed"]),
            str(record["condition"]),
        )
        if key in completed:
            raise ValueError(f"Duplicate ablation run: {key}")
        completed.add(key)
    return completed


def _write_status(
    output: Path,
    expected: int,
    completed: int,
    state: str,
    elapsed: float = 0.0,
) -> None:
    payload = {
        "state": state,
        "expected_runs": expected,
        "completed_unique_runs": completed,
        "missing_runs": max(0, expected - completed),
        "elapsed_seconds_this_invocation": round(elapsed, 3),
    }
    (output / "run_status.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
