from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.data.social_loader import (
    load_comments,
    load_population,
    load_posts,
    resolve_community_paths,
    select_posts,
)
from bdmtf.experiments import (
    _post_early_engagement_signal,
    _with_community_calibration,
    config_from_dict,
    interventions_from_config,
    simulation_seed,
)
from bdmtf.intent_pool import FrozenIntentPool
from bdmtf.metrics import compute_metrics
from bdmtf.reviewer_trait_coupling import (
    make_post_core_trait_transform,
    write_trait_analysis,
)
from bdmtf.simulator import BDMTFSimulator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the reviewer trait-coupling robustness experiment."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "reviewer_trait_coupling.json"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "run_outputs" / "reviewer_trait_coupling"),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()

    experiment = json.loads(Path(args.config).read_text(encoding="utf-8"))
    simulation_source = ROOT / experiment["simulation_config"]
    simulation_raw = json.loads(simulation_source.read_text(encoding="utf-8"))
    simulation_raw["conditions"] = experiment["conditions"]
    if args.quick:
        experiment["posts_per_community"] = 2
        experiment["seeds"] = [0]
    config = config_from_dict(simulation_raw)
    interventions = interventions_from_config(simulation_raw)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "runs.jsonl"
    records, completed = _load_existing(records_path)
    expected = (
        len(experiment["communities"])
        * int(experiment["posts_per_community"])
        * len(experiment["seeds"])
        * len(experiment["trait_modes"])
        * len(interventions)
    )
    _write_status(output, expected, len(completed), "starting")
    start = time.time()
    new_runs = 0
    with records_path.open("a", encoding="utf-8") as handle:
        for community in experiment["communities"]:
            paths = resolve_community_paths(ROOT / "data" / "social_paper", community)
            posts = select_posts(
                load_posts(paths, enriched=True),
                int(experiment["posts_per_community"]),
                seed=0,
            )
            comments = load_comments(paths)
            intent_path = paths.base_dir / "frozen_intents.jsonl"
            intent_pool = (
                FrozenIntentPool.from_jsonl(intent_path)
                if intent_path.exists()
                else FrozenIntentPool.from_comments(comments, seed=0)
            )
            community_config = _with_community_calibration(config, community)
            for seed in map(int, experiment["seeds"]):
                agents = load_population(paths, community_config, seed=seed)
                for trait_mode in experiment["trait_modes"]:
                    transform = make_post_core_trait_transform(
                        agents,
                        trait_mode,
                        simulation_seed(
                            community,
                            f"traits::{trait_mode}",
                            "traits",
                            seed,
                            seed_mode="paired_by_post_seed",
                        ),
                    )
                    for post in posts.itertuples(index=False):
                        post_id = str(post.post_id)
                        run_config = replace(
                            community_config,
                            initial_engagement_signal=_post_early_engagement_signal(
                                post
                            ),
                        )
                        for intervention in interventions:
                            key = (
                                community,
                                post_id,
                                seed,
                                trait_mode,
                                intervention.name,
                            )
                            if key in completed:
                                continue
                            sim_seed = simulation_seed(
                                community,
                                f"{post_id}::{trait_mode}",
                                intervention.name,
                                seed,
                                seed_mode="paired_by_post_seed",
                            )
                            simulator = BDMTFSimulator(
                                agents,
                                intent_pool,
                                config=run_config,
                                seed=sim_seed,
                                post_core_agent_transform=transform,
                            )
                            state, traces = simulator.run(
                                post_id=post_id,
                                title=str(getattr(post, "title", "")),
                                initial_text=str(getattr(post, "full_text", "")),
                                intervention=intervention,
                            )
                            record: dict[str, Any] = {
                                "community": community,
                                "post_id": post_id,
                                "seed": seed,
                                "trait_mode": trait_mode,
                                "condition": intervention.name,
                                "core": intervention.core,
                                "ranking": intervention.ranking.value,
                            }
                            record.update(compute_metrics(state, traces))
                            handle.write(
                                json.dumps(record, ensure_ascii=False) + "\n"
                            )
                            handle.flush()
                            records.append(record)
                            completed.add(key)
                            new_runs += 1
                            if (
                                args.checkpoint_every > 0
                                and new_runs % args.checkpoint_every == 0
                            ):
                                _write_status(
                                    output,
                                    expected,
                                    len(completed),
                                    "running",
                                    time.time() - start,
                                )
                            if args.max_runs and new_runs >= args.max_runs:
                                _finish(
                                    output,
                                    records_path,
                                    expected,
                                    completed,
                                    start,
                                )
                                return
    _finish(output, records_path, expected, completed, start)


def _load_existing(path: Path):
    records = []
    completed = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = (
                str(record["community"]),
                str(record["post_id"]),
                int(record["seed"]),
                str(record["trait_mode"]),
                str(record["condition"]),
            )
            if key not in completed:
                records.append(record)
                completed.add(key)
    return records, completed


def _finish(
    output: Path,
    records_path: Path,
    expected: int,
    completed: set,
    start: float,
) -> None:
    state = "complete" if len(completed) == expected else "incomplete"
    _write_status(output, expected, len(completed), state, time.time() - start)
    if state == "complete":
        write_trait_analysis(records_path, output / "analysis")
    print(f"Trait coupling {state}: {len(completed)}/{expected}")


def _write_status(
    output: Path,
    expected: int,
    completed: int,
    state: str,
    elapsed: float = 0.0,
) -> None:
    (output / "run_status.json").write_text(
        json.dumps(
            {
                "state": state,
                "expected_runs": expected,
                "completed_unique_runs": completed,
                "missing_runs": max(0, expected - completed),
                "elapsed_seconds_this_invocation": round(elapsed, 3),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
