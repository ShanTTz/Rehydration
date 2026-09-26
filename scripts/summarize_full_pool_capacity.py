"""Combine a five-community intent-opportunity capacity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.revision.intent_pool_sensitivity import (
    _effect_summary,
    _exhaustion_summary,
    _factorial_contrasts,
)


DEFAULT_INPUT = ROOT / "artifacts" / "reviewer_validation" / "intent_pool_full_256"
DEFAULT_OUTPUT = ROOT / "artifacts" / "reviewer_validation" / "intent_pool_full_256_summary"
COMMUNITIES = ("AskReddit", "aww", "funny", "science", "worldnews")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _catalog_stats(capacity_multiplier: int) -> pd.DataFrame:
    rows = []
    for community in COMMUNITIES:
        path = ROOT / "data" / "social_paper" / f"{community}_data" / "frozen_intents.jsonl"
        identifiers = set()
        sources: dict[str, int] = {}
        polarities: dict[str, int] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            identifiers.add(str(item["intent_id"]))
            source = str(item.get("metadata", {}).get("source", "unknown"))
            sources[source] = sources.get(source, 0) + 1
            polarity = str(item.get("polarity", "unknown"))
            polarities[polarity] = polarities.get(polarity, 0) + 1
        rows.append(
            {
                "community": community,
                "unique_intents": len(identifiers),
                "opportunity_capacity": len(identifiers) * capacity_multiplier,
                "catalog_bytes": path.stat().st_size,
                "catalog_sha256": _sha256(path),
                "source_counts": json.dumps(sources, sort_keys=True),
                "polarity_counts": json.dumps(polarities, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def summarize(
    input_dir: Path,
    output_dir: Path,
    bootstrap_samples: int,
    seed: int,
    capacity_multiplier: int = 256,
) -> dict[str, object]:
    frames = []
    manifests = []
    run_file_birthtimes = []
    manifest_mtimes = []
    for community in COMMUNITIES:
        directory = input_dir / community.lower()
        runs_path = directory / "capacity_runs.csv"
        manifest_path = directory / "manifest.json"
        if not runs_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"incomplete community output: {directory}")
        frame = pd.read_csv(runs_path)
        if sorted(frame["community"].astype(str).unique()) != [community]:
            raise ValueError(f"unexpected community content in {runs_path}")
        if sorted(frame["intent_pool_capacity_multiplier"].astype(int).unique()) != [
            capacity_multiplier
        ]:
            raise ValueError(f"unexpected capacity content in {runs_path}")
        frames.append(frame)
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        raw_runs_path = directory / f"capacity_{capacity_multiplier}x" / "runs.jsonl"
        if raw_runs_path.exists():
            run_file_birthtimes.append(raw_runs_path.stat().st_ctime)
        manifest_mtimes.append(manifest_path.stat().st_mtime)

    runs = pd.concat(frames, ignore_index=True)
    cells = {
        "baseline_best": "BASELINE",
        "baseline_controversial": "BASELINE_CONTROVERSIAL",
        "toxic_best": "CORE_TOXIC_BEST",
        "toxic_controversial": "CORE_TOXIC_CONTROVERSIAL",
    }
    contrasts = _factorial_contrasts(runs, cells)
    effects = _effect_summary(contrasts, bootstrap_samples, seed)
    exhaustion = _exhaustion_summary(runs)
    catalogs = _catalog_stats(capacity_multiplier)

    output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_dir / "full_capacity_runs.csv", index=False)
    contrasts.to_csv(output_dir / "full_capacity_contrasts.csv", index=False)
    effects.to_csv(output_dir / "full_capacity_effects.csv", index=False)
    exhaustion.to_csv(output_dir / "full_capacity_exhaustion.csv", index=False)
    catalogs.to_csv(output_dir / "catalog_capacity.csv", index=False)

    source_counts = catalogs["source_counts"].map(json.loads)
    all_empirical = all(set(counts) == {"comments_csv"} for counts in source_counts)
    manifest = {
        "status": "complete",
        "communities": list(COMMUNITIES),
        "posts": int(runs[["community", "post_id"]].drop_duplicates().shape[0]),
        "reconstructable_posts_in_source": 482,
        "seeds": int(runs["seed"].nunique()),
        "conditions": int(runs["condition"].nunique()),
        "runs": int(len(runs)),
        "capacity_multiplier": capacity_multiplier,
        "maximum_exhaustion_rate": float(exhaustion["exhaustion_rate"].max()),
        "unique_intents_total": int(catalogs["unique_intents"].sum()),
        "unique_intents_range": [
            int(catalogs["unique_intents"].min()),
            int(catalogs["unique_intents"].max()),
        ],
        "opportunity_capacity_total": int(catalogs["opportunity_capacity"].sum()),
        "catalog_bytes_total": int(catalogs["catalog_bytes"].sum()),
        "capacity_semantics": (
            "maximum branch-local realizations per immutable abstract intent; "
            "not additional generated texts"
        ),
        "additional_llm_calls_for_multiplier": 0,
        "additional_llm_tokens_for_multiplier": 0,
        "all_catalog_entries_empirical_comments": all_empirical,
        "bootstrap_samples": bootstrap_samples,
        "parallel_community_workers": len(COMMUNITIES),
        "approx_wall_clock_seconds": float(
            max(manifest_mtimes) - min(run_file_birthtimes)
        )
        if run_file_birthtimes
        else None,
        "wall_clock_source": "first raw-run file creation to last community manifest",
        "community_manifests": manifests,
    }
    (output_dir / "full_capacity_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    effect_lookup = effects.set_index("metric")["estimate"]
    macros = [
        "% Auto-generated by summarize_full_pool_capacity.py.",
        f"\\newcommand{{\\FullPoolCapacity}}{{{capacity_multiplier}}}",
        f"\\newcommand{{\\FullPoolPosts}}{{{manifest['posts']}}}",
        f"\\newcommand{{\\FullPoolRuns}}{{{manifest['runs']:,}}}",
        f"\\newcommand{{\\FullPoolUniqueIntents}}{{{manifest['unique_intents_total']:,}}}",
        f"\\newcommand{{\\FullPoolIntentMin}}{{{manifest['unique_intents_range'][0]:,}}}",
        f"\\newcommand{{\\FullPoolIntentMax}}{{{manifest['unique_intents_range'][1]:,}}}",
        f"\\newcommand{{\\FullPoolOpportunities}}{{{manifest['opportunity_capacity_total']:,}}}",
        f"\\newcommand{{\\FullPoolCatalogMB}}{{{manifest['catalog_bytes_total'] / 1_000_000:.1f}}}",
        f"\\newcommand{{\\FullPoolWallMinutes}}{{{manifest['approx_wall_clock_seconds'] / 60:.1f}}}",
        f"\\newcommand{{\\FullPoolMaxExhaustionPct}}{{{100 * manifest['maximum_exhaustion_rate']:.2f}}}",
        f"\\newcommand{{\\FullPoolJointVolume}}{{{effect_lookup['joint_volume_ratio']:.3f}}}",
        f"\\newcommand{{\\FullPoolJointDepth}}{{{effect_lookup['joint_mean_leaf_depth_delta']:.3f}}}",
    ]
    (output_dir / "full_pool_capacity_macros.tex").write_text(
        "\n".join(macros) + "\n", encoding="utf-8"
    )

    report = [
        f"# Full-Dataset {capacity_multiplier}x Intent-Opportunity Audit",
        "",
        f"- Posts: {manifest['posts']} across five communities ({manifest['runs']:,} policy replays).",
        f"- Maximum exhaustion: {100 * manifest['maximum_exhaustion_rate']:.2f}%.",
        f"- Unique immutable intents: {manifest['unique_intents_total']:,} total, "
        f"{manifest['unique_intents_range'][0]:,}--{manifest['unique_intents_range'][1]:,} per community.",
        f"- Branch-local realization opportunities at {capacity_multiplier}x: {manifest['opportunity_capacity_total']:,}.",
        "- The multiplier changes allowed realizations of an already frozen abstract frame; it does not call an LLM or create 256 new texts.",
        "- Additional API calls and tokens caused by the multiplier: 0 and 0.",
        f"- Approximate five-worker wall clock: {manifest['approx_wall_clock_seconds'] / 60:.1f} minutes.",
        "",
        "The 25-post, three-seed panel remains the capacity-convergence analysis. This full-dataset run is the coverage audit for the selected zero-exhaustion setting.",
        "",
    ]
    (output_dir / "FULL_POOL_CAPACITY_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=30371)
    parser.add_argument("--capacity", type=int, default=256)
    args = parser.parse_args()
    result = summarize(
        args.input_dir,
        args.output_dir,
        args.bootstrap_samples,
        args.seed,
        args.capacity,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
