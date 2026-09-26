from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.data.social_loader import load_comments, load_population, load_posts, resolve_community_paths, select_posts
from bdmtf.experiments import (
    _post_early_engagement_signal,
    _with_community_calibration,
    config_from_dict,
    interventions_from_config,
    simulation_seed,
    write_summary,
)
from bdmtf.intent_pool import FrozenIntentPool
from bdmtf.metrics import compute_metrics
from bdmtf.paper_outputs import write_paper_outputs
from bdmtf.simulator import BDMTFSimulator


RecordKey = Tuple[str, str, int, str]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or resume the complete 9000-run paper reproduction.")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "social_paper"))
    parser.add_argument("--output", default=str(ROOT / "run_outputs" / "paper_reproduction_resumable"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper_reproduction.json"))
    parser.add_argument("--quick", action="store_true", help="Small validation run: 2 posts, seed 0 only.")
    parser.add_argument("--posts-per-community", type=int, default=0, help="Override the configured post count.")
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        help="Run only the named condition; may be supplied more than once.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Write status every N newly completed runs.")
    parser.add_argument("--max-runs", type=int, default=0, help="Stop after N new runs; useful for smoke tests.")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if args.quick:
        raw["posts_per_community"] = 2
        raw["seeds"] = [0]
    elif args.posts_per_community > 0:
        raw["posts_per_community"] = args.posts_per_community

    if args.condition:
        requested = set(args.condition)
        raw["conditions"] = [item for item in raw.get("conditions", []) if item.get("name") in requested]
        missing = requested - {item.get("name") for item in raw["conditions"]}
        if missing:
            raise SystemExit(f"Unknown condition(s): {', '.join(sorted(missing))}")

    config = config_from_dict(raw)
    interventions = interventions_from_config(raw)
    seed_mode = str(raw.get("randomization", {}).get("seed_mode", "condition_specific"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runs_path = output / "runs.jsonl"

    records, completed, duplicate_count = _load_existing_records(runs_path)
    expected_total = _count_expected_runs(
        social_root=Path(args.data_root),
        communities=raw["communities"],
        posts_per_community=int(raw["posts_per_community"]),
        seeds=raw["seeds"],
        interventions=interventions,
    )
    _write_status(output, expected_total, len(completed), duplicate_count, "starting")

    start = time.time()
    new_runs = 0
    with runs_path.open("a", encoding="utf-8") as handle:
        for record in _run_missing_records(
            social_root=Path(args.data_root),
            communities=raw["communities"],
            posts_per_community=int(raw["posts_per_community"]),
            seeds=raw["seeds"],
            config=config,
            interventions=interventions,
            completed=completed,
            seed_mode=seed_mode,
        ):
            key = _record_key(record)
            if key in completed:
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            completed.add(key)
            records.append(record)
            new_runs += 1

            if args.checkpoint_every > 0 and new_runs % args.checkpoint_every == 0:
                handle.flush()
                _write_status(output, expected_total, len(completed), duplicate_count, "running", elapsed=time.time() - start)
                print(f"checkpoint: {len(completed)}/{expected_total} complete")

            if args.max_runs > 0 and new_runs >= args.max_runs:
                break

    if records:
        write_summary(records, output)
        write_paper_outputs(records, output)

    state = "complete" if len(completed) >= expected_total else "incomplete"
    _write_status(output, expected_total, len(completed), duplicate_count, state, elapsed=time.time() - start)
    print(f"Paper reproduction {state}: {len(completed)}/{expected_total} runs")
    print(f"New runs this invocation: {new_runs}")
    print(f"Output: {output}")


def _load_existing_records(path: Path) -> Tuple[List[Dict[str, Any]], Set[RecordKey], int]:
    records: List[Dict[str, Any]] = []
    completed: Set[RecordKey] = set()
    duplicate_count = 0
    if not path.exists():
        return records, completed, duplicate_count
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Cannot resume because {path} line {line_number} is not valid JSON: {exc}") from exc
            key = _record_key(record)
            if key in completed:
                duplicate_count += 1
                continue
            completed.add(key)
            records.append(record)
    return records, completed, duplicate_count


def _count_expected_runs(
    social_root: Path,
    communities: Iterable[str],
    posts_per_community: int,
    seeds: Iterable[int],
    interventions: Iterable[Any],
) -> int:
    seed_count = len(list(seeds))
    intervention_count = len(list(interventions))
    total = 0
    for community in communities:
        paths = resolve_community_paths(social_root, community)
        posts_df = load_posts(paths, enriched=True)
        total += len(select_posts(posts_df, posts_per_community, seed=0)) * seed_count * intervention_count
    return total


def _run_missing_records(
    social_root: Path,
    communities: Iterable[str],
    posts_per_community: int,
    seeds: Iterable[int],
    config: Any,
    interventions: Iterable[Any],
    completed: Set[RecordKey],
    seed_mode: str = "condition_specific",
) -> Iterable[Dict[str, Any]]:
    intervention_list = list(interventions)
    seed_list = [int(seed) for seed in seeds]
    for community in communities:
        paths = resolve_community_paths(social_root, community)
        posts_df = load_posts(paths, enriched=True)
        comments_df = load_comments(paths)
        selected_posts = select_posts(posts_df, posts_per_community, seed=0)
        community_config = _with_community_calibration(config, community)
        frozen_path = paths.base_dir / "frozen_intents.jsonl"
        intent_pool = (
            FrozenIntentPool.from_jsonl(frozen_path)
            if frozen_path.exists()
            else FrozenIntentPool.from_comments(comments_df, seed=0)
        )

        for seed in seed_list:
            agents = load_population(paths, community_config, seed=seed)
            for post in selected_posts.itertuples(index=False):
                title = str(getattr(post, "title", ""))
                post_id = str(getattr(post, "post_id"))
                for intervention in intervention_list:
                    key = (community, post_id, seed, intervention.name)
                    if key in completed:
                        continue
                    sim_seed = simulation_seed(
                        community,
                        post_id,
                        intervention.name,
                        seed,
                        seed_mode=seed_mode,
                    )
                    run_config = replace(
                        community_config,
                        initial_engagement_signal=_post_early_engagement_signal(post),
                    )
                    sim = BDMTFSimulator(agents, intent_pool, config=run_config, seed=sim_seed)
                    state, traces = sim.run(
                        post_id=post_id,
                        title=title,
                        initial_text=str(getattr(post, "full_text", "")),
                        intervention=intervention,
                    )
                    metrics = compute_metrics(state, traces)
                    record: Dict[str, Any] = {
                        "community": community,
                        "post_id": post_id,
                        "seed": seed,
                        "condition": intervention.name,
                        "core": intervention.core,
                        "ranking": intervention.ranking.value,
                        "context": intervention.context,
                        "intent_pool_capacity_multiplier": int(
                            run_config.intent_pool_capacity_multiplier
                        ),
                        "intent_pool_exhaustion_policy": str(
                            run_config.intent_pool_exhaustion_policy
                        ),
                    }
                    record.update(metrics)
                    yield record


def _record_key(record: Dict[str, Any]) -> RecordKey:
    return (
        str(record["community"]),
        str(record["post_id"]),
        int(record["seed"]),
        str(record["condition"]),
    )


def _write_status(
    output: Path,
    expected_total: int,
    completed_total: int,
    duplicate_count: int,
    state: str,
    elapsed: float = 0.0,
) -> None:
    payload = {
        "state": state,
        "expected_runs": expected_total,
        "completed_unique_runs": completed_total,
        "missing_runs": max(0, expected_total - completed_total),
        "duplicate_existing_lines": duplicate_count,
        "elapsed_seconds_this_invocation": round(elapsed, 3),
    }
    (output / "run_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
