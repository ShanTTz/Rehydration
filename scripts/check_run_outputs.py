from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.data.social_loader import load_posts, resolve_community_paths, select_posts
from bdmtf.experiments import interventions_from_config


RecordKey = Tuple[str, str, int, str]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether a reproduction output directory is complete.")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "social_paper"))
    parser.add_argument("--output", default=str(ROOT / "run_outputs" / "paper_reproduction"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper_reproduction.json"))
    parser.add_argument("--quick", action="store_true", help="Evaluate against the quick-run shape: 2 posts, seed 0.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable status.")
    parser.add_argument("--require-complete", action="store_true", help="Return non-zero if any expected run is missing.")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if args.quick:
        raw["posts_per_community"] = 2
        raw["seeds"] = [0]

    expected = _expected_keys(
        social_root=Path(args.data_root),
        communities=raw["communities"],
        posts_per_community=int(raw["posts_per_community"]),
        seeds=raw["seeds"],
        conditions=[item.name for item in interventions_from_config(raw)],
    )
    status = _inspect_output(Path(args.output), expected)

    if args.json:
        print(json.dumps(status, indent=2, ensure_ascii=False))
    else:
        _print_status(status)

    if args.require_complete and status["missing_runs"] > 0:
        return 1
    return 0


def _expected_keys(
    social_root: Path,
    communities: Iterable[str],
    posts_per_community: int,
    seeds: Iterable[int],
    conditions: Iterable[str],
) -> Set[RecordKey]:
    keys: Set[RecordKey] = set()
    seed_list = [int(seed) for seed in seeds]
    condition_list = [str(condition) for condition in conditions]
    for community in communities:
        paths = resolve_community_paths(social_root, community)
        posts_df = load_posts(paths, enriched=True)
        selected_posts = select_posts(posts_df, posts_per_community, seed=0)
        for post_id in selected_posts["post_id"].astype(str):
            for seed in seed_list:
                for condition in condition_list:
                    keys.add((str(community), str(post_id), seed, condition))
    return keys


def _inspect_output(output: Path, expected: Set[RecordKey]) -> Dict[str, Any]:
    runs_path = output / "runs.jsonl"
    completed: Set[RecordKey] = set()
    duplicate_lines = 0
    invalid_lines: List[int] = []
    total_lines = 0
    if runs_path.exists():
        with runs_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                total_lines += 1
                try:
                    record = json.loads(line)
                    key = (
                        str(record["community"]),
                        str(record["post_id"]),
                        int(record["seed"]),
                        str(record["condition"]),
                    )
                except Exception:
                    invalid_lines.append(line_number)
                    continue
                if key in completed:
                    duplicate_lines += 1
                completed.add(key)

    missing = sorted(expected - completed)
    unexpected = sorted(completed - expected)
    paper_dir = output / "paper_tables"
    return {
        "output": str(output),
        "runs_jsonl": str(runs_path),
        "runs_jsonl_exists": runs_path.exists(),
        "expected_runs": len(expected),
        "total_nonempty_lines": total_lines,
        "completed_unique_runs": len(completed & expected),
        "missing_runs": len(missing),
        "unexpected_runs": len(unexpected),
        "duplicate_lines": duplicate_lines,
        "invalid_json_lines": invalid_lines[:20],
        "summary_csv_exists": (output / "summary.csv").exists(),
        "volume_depth_csv_exists": (output / "volume_depth.csv").exists(),
        "paper_tables_exists": paper_dir.exists(),
        "paper_table_files": sorted(path.name for path in paper_dir.glob("*")) if paper_dir.exists() else [],
        "missing_sample": [
            {"community": k[0], "post_id": k[1], "seed": k[2], "condition": k[3]}
            for k in missing[:10]
        ],
        "unexpected_sample": [
            {"community": k[0], "post_id": k[1], "seed": k[2], "condition": k[3]}
            for k in unexpected[:10]
        ],
    }


def _print_status(status: Dict[str, Any]) -> None:
    print(f"Output: {status['output']}")
    print(f"Expected runs: {status['expected_runs']}")
    print(f"Completed unique runs: {status['completed_unique_runs']}")
    print(f"Missing runs: {status['missing_runs']}")
    print(f"Unexpected runs: {status['unexpected_runs']}")
    print(f"Duplicate lines: {status['duplicate_lines']}")
    print(f"Invalid JSON lines: {len(status['invalid_json_lines'])}")
    print(f"summary.csv: {'yes' if status['summary_csv_exists'] else 'no'}")
    print(f"paper_tables: {'yes' if status['paper_tables_exists'] else 'no'}")
    if status["missing_sample"]:
        print("First missing keys:")
        for item in status["missing_sample"]:
            print(f"  {item['community']} {item['post_id']} seed={item['seed']} condition={item['condition']}")


if __name__ == "__main__":
    raise SystemExit(main())
