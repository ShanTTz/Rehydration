from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from bdmtf.data.social_loader import load_comments, load_posts, resolve_community_paths, strip_reddit_prefix
from bdmtf.features import DARK_SIGNAL_LEXICONS, POSITIVE_LEXICON
from bdmtf.revision.provenance import sha256_file, write_json


COMMUNITIES = ("AskReddit", "aww", "funny", "science", "worldnews")
METRIC_COLUMNS = (
    "size",
    "max_depth",
    "mean_leaf_depth",
    "mean_depth",
    "mean_branching_factor",
    "root_reply_share",
    "width_gini",
    "time_to_50_minutes",
    "time_to_90_minutes",
    "duration_minutes",
    "mean_interarrival_minutes",
    "repeat_author_share",
    "toxicity_density",
    "counterspeech_rate",
    "removed_rate",
)


def parse_timestamp(series: pd.Series) -> pd.Series:
    """Parse mixed ISO timestamps and infer the unit of numeric Unix epochs."""
    values = pd.Series(series, index=series.index)
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_mask = numeric.notna()
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    if numeric_mask.any():
        magnitude = float(numeric.loc[numeric_mask].abs().median())
        unit = "ns" if magnitude >= 1e17 else "us" if magnitude >= 1e14 else "ms" if magnitude >= 1e11 else "s"
        parsed.loc[numeric_mask] = pd.to_datetime(
            numeric.loc[numeric_mask],
            unit=unit,
            utc=True,
            errors="coerce",
        )
    if (~numeric_mask).any():
        textual = values.loc[~numeric_mask]
        try:
            parsed.loc[~numeric_mask] = pd.to_datetime(
                textual,
                format="mixed",
                utc=True,
                errors="coerce",
            )
        except TypeError:
            parsed.loc[~numeric_mask] = textual.map(
                lambda value: pd.to_datetime(value, utc=True, errors="coerce")
            )
    return parsed


def build_splits(
    social_root: Path,
    output_dir: Path,
    communities: Iterable[str] = COMMUNITIES,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> pd.DataFrame:
    rows = []
    for community in communities:
        paths = resolve_community_paths(social_root, community)
        posts = load_posts(paths, enriched=False).copy()
        posts["community"] = community
        posts["created_at"] = parse_timestamp(posts["created_utc"])
        posts["is_viral"] = pd.to_numeric(posts.get("is_viral", 0), errors="coerce").fillna(0).astype(int)
        for _, group in posts.groupby("is_viral", dropna=False):
            group = group.sort_values(["created_at", "post_id"], na_position="first").reset_index(drop=True)
            count = len(group)
            train_end = int(np.floor(count * train_fraction))
            validation_end = train_end + int(np.floor(count * validation_fraction))
            assignments = np.full(count, "test", dtype=object)
            assignments[:train_end] = "train"
            assignments[train_end:validation_end] = "validation"
            for index, post in group.iterrows():
                rows.append(
                    {
                        "community": community,
                        "post_id": str(post["post_id"]),
                        "is_viral": int(post["is_viral"]),
                        "created_at": post["created_at"],
                        "split": str(assignments[index]),
                    }
                )
    splits = pd.DataFrame(rows).sort_values(["community", "split", "created_at", "post_id"])
    if splits.duplicated(["community", "post_id"]).any():
        raise ValueError("A post was assigned to more than one split")
    output_dir.mkdir(parents=True, exist_ok=True)
    splits.to_csv(output_dir / "post_splits.csv", index=False)
    splits.to_parquet(output_dir / "post_splits.parquet", index=False)
    counts = splits.groupby(["community", "is_viral", "split"]).size().rename("n").reset_index()
    write_json(
        output_dir / "split_manifest.json",
        {
            "method": "chronological within community and viral label",
            "fractions": {"train": train_fraction, "validation": validation_fraction, "test": 1 - train_fraction - validation_fraction},
            "n_posts": int(len(splits)),
            "duplicate_assignments": 0,
            "counts": counts.to_dict(orient="records"),
        },
    )
    return splits


def load_author_history_before(authors_csv: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_csv(authors_csv, encoding="utf-8", encoding_errors="replace")
    if "created_utc" not in frame:
        return frame.iloc[0:0].copy()
    timestamps = parse_timestamp(frame["created_utc"])
    return frame.loc[timestamps < cutoff].copy()


def _text_flags(text: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    clean = text.fillna("").astype(str).str.lower()
    dark_phrases = sorted({phrase for lexicon in DARK_SIGNAL_LEXICONS.values() for phrase in lexicon})
    positive_phrases = sorted(set(POSITIVE_LEXICON))
    toxic = pd.Series(False, index=clean.index)
    positive = pd.Series(False, index=clean.index)
    for phrase in dark_phrases:
        toxic |= clean.str.contains(phrase, regex=False)
    for phrase in positive_phrases:
        positive |= clean.str.contains(phrase, regex=False)
    removed = clean.str.strip().isin({"[removed]", "[deleted]", "removed", "deleted"})
    counterspeech = positive & ~toxic
    return toxic, counterspeech, removed


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or np.all(values == 0):
        return 0.0
    values = np.sort(np.maximum(values, 0.0))
    index = np.arange(1, len(values) + 1)
    return float(np.sum((2 * index - len(values) - 1) * values) / (len(values) * values.sum()))


def cascade_metrics(comments: pd.DataFrame, post_id: str, community: str) -> dict[str, Any] | None:
    if comments.empty:
        return None
    frame = comments.copy()
    frame["comment_id"] = frame["comment_id"].map(strip_reddit_prefix).astype(str)
    frame["parent_clean"] = frame["parent_id"].map(strip_reddit_prefix).astype(str)
    ids = set(frame["comment_id"])
    if not ids:
        return None
    depth = pd.to_numeric(frame.get("depth", 0), errors="coerce").fillna(0).clip(lower=0).astype(int) + 1
    frame["depth_reconstructed"] = depth
    child_counts = frame["parent_clean"].value_counts()
    leaves = frame.loc[~frame["comment_id"].isin(child_counts.index)]
    root_reply = ~frame["parent_clean"].isin(ids)
    width = frame.groupby("depth_reconstructed").size().to_numpy(dtype=float)
    non_leaf_children = child_counts[child_counts.index.isin(ids)].to_numpy(dtype=float)
    minutes = pd.to_numeric(frame.get("minutes_since_post", np.nan), errors="coerce").dropna().sort_values().to_numpy(dtype=float)
    minutes = minutes[np.isfinite(minutes) & (minutes >= 0)]
    authors = frame.get("author", pd.Series("", index=frame.index)).fillna("").astype(str)
    author_counts = authors[~authors.isin({"", "[deleted]", "None", "nan"})].value_counts()
    toxic, counterspeech, removed = _text_flags(frame.get("comment_text", pd.Series("", index=frame.index)))
    size = len(frame)
    return {
        "community": community,
        "post_id": str(post_id),
        "size": float(size),
        "max_depth": float(depth.max()),
        "mean_leaf_depth": float(leaves["depth_reconstructed"].mean()) if len(leaves) else 0.0,
        "mean_depth": float(depth.mean()),
        "mean_branching_factor": float(np.mean(non_leaf_children)) if len(non_leaf_children) else 0.0,
        "root_reply_share": float(root_reply.mean()),
        "width_gini": _gini(width),
        "time_to_50_minutes": float(np.quantile(minutes, 0.5)) if len(minutes) else 0.0,
        "time_to_90_minutes": float(np.quantile(minutes, 0.9)) if len(minutes) else 0.0,
        "duration_minutes": float(minutes.max() - minutes.min()) if len(minutes) > 1 else 0.0,
        "mean_interarrival_minutes": float(np.diff(minutes).mean()) if len(minutes) > 1 else 0.0,
        "repeat_author_share": float(author_counts[author_counts > 1].sum() / size) if size else 0.0,
        "toxicity_density": float(toxic.mean()),
        "counterspeech_rate": float(counterspeech.mean()),
        "removed_rate": float(removed.mean()),
        "n_leaf_nodes": int(len(leaves)),
    }


def audit_dataset(social_root: Path, output_dir: Path, splits: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if splits is None:
        split_path = output_dir.parent / "splits" / "post_splits.csv"
        splits = pd.read_csv(split_path) if split_path.exists() else build_splits(social_root, output_dir.parent / "splits")
    all_metrics = []
    inventory = []
    for community in COMMUNITIES:
        paths = resolve_community_paths(social_root, community)
        posts = load_posts(paths, enriched=False)
        comments = load_comments(paths)
        comments["post_id"] = comments["post_id"].astype(str)
        groups = {str(key): value for key, value in comments.groupby("post_id", sort=False)}
        community_splits = splits[splits["community"] == community]
        for row in community_splits.itertuples(index=False):
            metrics = cascade_metrics(groups.get(str(row.post_id), pd.DataFrame()), str(row.post_id), community)
            if metrics:
                metrics.update({"split": row.split, "is_viral": int(row.is_viral)})
                all_metrics.append(metrics)
        inventory.append(
            {
                "community": community,
                "posts": int(len(posts)),
                "comments": int(len(comments)),
                "reconstructable_cascades": int(sum(item["community"] == community for item in all_metrics)),
                "posts_sha256": sha256_file(paths.posts_csv),
                "comments_sha256": sha256_file(paths.comments_csv),
            }
        )
    metrics_frame = pd.DataFrame(all_metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(output_dir / "empirical_cascade_metrics.csv", index=False)
    metrics_frame.to_parquet(output_dir / "empirical_cascade_metrics.parquet", index=False)
    split_counts = splits.groupby("split").size().to_dict()
    empirical_leaf = metrics_frame.groupby(["community", "split"])["mean_leaf_depth"].agg(["count", "mean", "std"]).reset_index()
    audit = {
        "n_posts": int(len(splits)),
        "n_reconstructable_cascades": int(len(metrics_frame)),
        "n_unreconstructable_posts": int(len(splits) - len(metrics_frame)),
        "split_counts": {key: int(value) for key, value in split_counts.items()},
        "inventory": inventory,
        "metric_definition": "Leaf depth counts top-level comments as depth 1.",
        "paper_mean_leaf_depth_claim": 18.4,
        "empirical_mean_leaf_depth": float(metrics_frame["mean_leaf_depth"].mean()),
        "paper_metric_reconstructed": bool(np.isclose(metrics_frame["mean_leaf_depth"].mean(), 18.4, rtol=0.05)),
        "empirical_leaf_depth_by_community_split": empirical_leaf.fillna(0).to_dict(orient="records"),
    }
    write_json(output_dir / "data_audit.json", audit)
    return metrics_frame, audit
