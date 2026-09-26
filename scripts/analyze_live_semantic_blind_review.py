"""Analyze completed three-arm dynamic-parent semantic blind ratings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = (
    ROOT
    / "artifacts"
    / "reviewer_validation"
    / "live_semantic_ceiling"
    / "blind_review_package"
)
SOURCES = (
    "frozen_frame_renderer",
    "parent_aware_intent_renderer",
    "dynamic_parent_live_generation",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_completed(completed_dir: Path) -> pd.DataFrame:
    paths = sorted(completed_dir.glob("SEM-*.csv"))
    if not paths:
        raise ValueError(f"No completed assignment CSV files in {completed_dir}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["assignment_id", "item_number"]).any():
        raise ValueError("Duplicate assignment/item rows")
    return result


def long_ratings(master: pd.DataFrame, completed: pd.DataFrame) -> pd.DataFrame:
    keys = ["assignment_id", "item_number"]
    mapping_columns = keys + [
        "item_id",
        "community",
        "condition",
        "depth_bin",
        *[f"candidate_{label}_source" for label in "ABC"],
    ]
    mapped = completed.merge(
        master[mapping_columns], on=keys, validate="one_to_one"
    )
    rows = []
    for row in mapped.itertuples(index=False):
        for label in "ABC":
            source = getattr(row, f"candidate_{label}_source")
            rows.append(
                {
                    "assignment_id": row.assignment_id,
                    "item_number": int(row.item_number),
                    "item_id": row.item_id,
                    "community": row.community,
                    "condition": row.condition,
                    "depth_bin": row.depth_bin,
                    "candidate_label": label,
                    "source": source,
                    "relevance": getattr(row, f"relevance_{label}_1_to_5"),
                    "coherence": getattr(row, f"coherence_{label}_1_to_5"),
                    "intent_preservation": getattr(
                        row, f"intent_preservation_{label}_1_to_5"
                    ),
                    "mismatch": getattr(row, f"obvious_mismatch_{label}_0_or_1"),
                    "best_label": row.best_direct_response_A_B_C_or_tie,
                }
            )
    result = pd.DataFrame(rows)
    for column in ("relevance", "coherence", "intent_preservation", "mismatch"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["relevance", "coherence", "intent_preservation", "mismatch"]].isna().any().any():
        raise ValueError("All numeric ratings must be complete")
    if not result["relevance"].between(1, 5).all() or not result["coherence"].between(1, 5).all():
        raise ValueError("Relevance and coherence must be in [1, 5]")
    if not result["intent_preservation"].between(1, 5).all():
        raise ValueError("Intent preservation must be in [1, 5]")
    if not result["mismatch"].isin([0, 1]).all():
        raise ValueError("Mismatch must be 0 or 1")
    result["direct_acceptable"] = (
        (result["relevance"] >= 3)
        & (result["coherence"] >= 3)
        & (result["mismatch"] == 0)
    ).astype(float)
    result["intent_compatible"] = (
        (result["direct_acceptable"] == 1)
        & (result["intent_preservation"] >= 3)
    ).astype(float)
    result["best"] = (
        result["candidate_label"] == result["best_label"].astype(str).str.upper()
    ).astype(float)
    return result


def clustered_summary(ratings: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    metrics = [
        "relevance",
        "coherence",
        "intent_preservation",
        "mismatch",
        "direct_acceptable",
        "intent_compatible",
        "best",
    ]
    item = ratings.groupby(["item_id", "source"], as_index=False)[metrics].mean()
    item_ids = sorted(item["item_id"].unique())
    rng = np.random.default_rng(seed)
    rows = []
    for source in SOURCES:
        group = item[item["source"] == source].set_index("item_id").loc[item_ids]
        for metric in metrics:
            values = group[metric].to_numpy(float)
            samples = np.empty(draws)
            for draw in range(draws):
                samples[draw] = values[rng.integers(0, len(values), len(values))].mean()
            low, high = np.quantile(samples, [0.025, 0.975])
            rows.append(
                {
                    "source": source,
                    "metric": metric,
                    "estimate": float(values.mean()),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "items": len(values),
                }
            )
    return pd.DataFrame(rows)


def paired_gaps(ratings: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    metrics = ["relevance", "coherence", "intent_preservation", "direct_acceptable", "intent_compatible"]
    item = ratings.groupby(["item_id", "source"], as_index=False)[metrics].mean()
    pairs = [
        ("frozen_minus_live", "frozen_frame_renderer", "dynamic_parent_live_generation"),
        ("intent_renderer_minus_live", "parent_aware_intent_renderer", "dynamic_parent_live_generation"),
        ("intent_renderer_minus_frozen", "parent_aware_intent_renderer", "frozen_frame_renderer"),
    ]
    rng = np.random.default_rng(seed)
    rows = []
    for label, left, right in pairs:
        merged = item[item.source == left].merge(
            item[item.source == right], on="item_id", suffixes=("_left", "_right"), validate="one_to_one"
        )
        for metric in metrics:
            values = merged[f"{metric}_left"].to_numpy(float) - merged[f"{metric}_right"].to_numpy(float)
            samples = np.empty(draws)
            for draw in range(draws):
                samples[draw] = values[rng.integers(0, len(values), len(values))].mean()
            low, high = np.quantile(samples, [0.025, 0.975])
            rows.append({"contrast": label, "metric": metric, "estimate": float(values.mean()), "ci_low": float(low), "ci_high": float(high), "items": len(values)})
    return pd.DataFrame(rows)


def write_table(summary: pd.DataFrame, path: Path) -> None:
    pivot = summary.set_index(["source", "metric"])
    labels = {
        "frozen_frame_renderer": "Frozen frame",
        "parent_aware_intent_renderer": "Intent-preserving",
        "dynamic_parent_live_generation": "Live generation",
    }
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Method & Relevance & Coherence & Intent & Acceptable (\%) \\", r"\midrule"]
    for source in SOURCES:
        rel = pivot.loc[(source, "relevance")]
        coh = pivot.loc[(source, "coherence")]
        intent = pivot.loc[(source, "intent_preservation")]
        acceptable = pivot.loc[(source, "direct_acceptable")]
        lines.append(
            f"{labels[source]} & {rel.estimate:.2f} & {coh.estimate:.2f} & "
            f"{intent.estimate:.2f} & {100 * acceptable.estimate:.1f} "
            f"[{100 * acceptable.ci_low:.1f}, {100 * acceptable.ci_high:.1f}] "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--completed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "reviewer_validation" / "live_semantic_ceiling" / "analysis")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=30371)
    args = parser.parse_args()
    master_path = args.package / "researcher_only" / "RESEARCHER_MASTER_DO_NOT_SHARE.csv"
    master = pd.read_csv(master_path)
    completed = load_completed(args.completed_dir)
    ratings = long_ratings(master, completed)
    summary = clustered_summary(ratings, args.bootstrap, args.seed)
    gaps = paired_gaps(ratings, args.bootstrap, args.seed + 1)
    args.output.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(args.output / "ratings_long.csv", index=False)
    summary.to_csv(args.output / "semantic_ceiling_summary.csv", index=False)
    gaps.to_csv(args.output / "semantic_ceiling_paired_gaps.csv", index=False)
    write_table(summary, args.output / "table_semantic_ceiling.tex")
    manifest = {
        "status": "complete",
        "items": int(ratings["item_id"].nunique()),
        "assignments": int(ratings["assignment_id"].nunique()),
        "ratings_per_source": int(len(ratings) / len(SOURCES)),
        "sources": list(SOURCES),
        "master_sha256": sha256(master_path),
        "completed_file_hashes": {path.name: sha256(path) for path in sorted(args.completed_dir.glob("SEM-*.csv"))},
        "analysis_seed": args.seed,
        "bootstrap_draws": args.bootstrap,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(summary.to_string(index=False))
    print(gaps.to_string(index=False))


if __name__ == "__main__":
    main()
