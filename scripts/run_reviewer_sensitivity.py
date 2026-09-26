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
    _stable_seed,
    _with_community_calibration,
    config_from_dict,
    interventions_from_config,
)
from bdmtf.intent_pool import FrozenIntentPool
from bdmtf.metrics import compute_metrics
from bdmtf.reviewer_sensitivity import (
    apply_parameter_draw,
    latin_hypercube_draws,
    write_sensitivity_analysis,
)
from bdmtf.simulator import BDMTFSimulator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run resumable local-uncertainty sensitivity experiments."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "reviewer_local_sensitivity.json"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "run_outputs" / "reviewer_local_sensitivity"),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-runs", type=int, default=0)
    args = parser.parse_args()

    sensitivity = json.loads(Path(args.config).read_text(encoding="utf-8"))
    source_path = ROOT / sensitivity["source_config"]["path"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    design = dict(sensitivity["design"])
    parameters = list(sensitivity["parameters"])
    if args.quick:
        design["draws"] = 2
        design["posts_per_community"] = 1
        design["seeds"] = [0]

    draws = latin_hypercube_draws(
        parameters,
        int(design["draws"]),
        int(design["lhs_seed"]),
    )
    condition_names = set(design["conditions"])
    source["conditions"] = [
        item for item in source["conditions"] if item["name"] in condition_names
    ]
    if {item["name"] for item in source["conditions"]} != condition_names:
        raise ValueError("Sensitivity config refers to an unknown source condition")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runs_path = output / "runs.jsonl"
    completed = _completed_keys(runs_path)
    expected = (
        len(draws)
        * len(source["communities"])
        * int(design["posts_per_community"])
        * len(design["seeds"])
        * len(source["conditions"])
    )
    _write_status(output, expected, len(completed), "starting")

    draw_manifest = {
        "schema_version": 1,
        "design": design,
        "parameters": parameters,
        "draws": [
            {"scenario": f"lhs_{index:03d}", **draw}
            for index, draw in enumerate(draws)
        ],
        "source_config": sensitivity["source_config"],
        "evidence_scope": sensitivity["evidence_scope"],
    }
    (output / "draw_manifest.json").write_text(
        json.dumps(draw_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    start = time.time()
    new_runs = 0
    with runs_path.open("a", encoding="utf-8") as handle:
        for scenario_index, draw in enumerate(draws):
            scenario = f"lhs_{scenario_index:03d}"
            raw = apply_parameter_draw(source, parameters, draw)
            config = config_from_dict(raw)
            interventions = interventions_from_config(raw)
            for community in raw["communities"]:
                paths = resolve_community_paths(ROOT / "data" / "social_paper", community)
                posts = select_posts(
                    load_posts(paths, enriched=True),
                    int(design["posts_per_community"]),
                    seed=0,
                )
                comments = load_comments(paths)
                community_config = _with_community_calibration(config, community)
                frozen_path = paths.base_dir / "frozen_intents.jsonl"
                intent_pool = (
                    FrozenIntentPool.from_jsonl(frozen_path)
                    if frozen_path.exists()
                    else FrozenIntentPool.from_comments(comments, seed=0)
                )
                for seed_value in design["seeds"]:
                    seed = int(seed_value)
                    agents = load_population(paths, community_config, seed=seed)
                    for post in posts.itertuples(index=False):
                        post_id = str(getattr(post, "post_id"))
                        sim_seed = _stable_seed(
                            "reviewer_sensitivity",
                            scenario,
                            community,
                            post_id,
                            seed,
                        )
                        run_config = replace(
                            community_config,
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
                                intent_pool,
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
                                "community": community,
                                "post_id": post_id,
                                "seed": seed,
                                "condition": intervention.name,
                                "core": intervention.core,
                                "ranking": intervention.ranking.value,
                            }
                            record.update(
                                {f"parameter__{name}": value for name, value in draw.items()}
                            )
                            record.update(compute_metrics(state, traces))
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            completed.add(key)
                            new_runs += 1
                            if (
                                args.checkpoint_every > 0
                                and new_runs % args.checkpoint_every == 0
                            ):
                                handle.flush()
                                _write_status(
                                    output,
                                    expected,
                                    len(completed),
                                    "running",
                                    time.time() - start,
                                )
                                print(f"checkpoint: {len(completed)}/{expected}")
                            if args.max_runs > 0 and new_runs >= args.max_runs:
                                _write_status(
                                    output,
                                    expected,
                                    len(completed),
                                    "incomplete",
                                    time.time() - start,
                                )
                                return

    state = "complete" if len(completed) == expected else "incomplete"
    _write_status(output, expected, len(completed), state, time.time() - start)
    if state == "complete":
        manifest = write_sensitivity_analysis(
            runs_path,
            output / "analysis",
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _completed_keys(path: Path) -> set[tuple[str, str, str, int, str]]:
    completed: set[tuple[str, str, str, int, str]] = set()
    if not path.is_file():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
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
                raise ValueError(f"Duplicate sensitivity run: {key}")
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
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
