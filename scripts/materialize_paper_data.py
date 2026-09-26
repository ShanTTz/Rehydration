from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, Set

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.data.social_loader import (
    extract_cascade_features,
    load_comments,
    load_posts,
    resolve_community_paths,
    select_posts,
)
from bdmtf.intent_pool import FrozenIntentPool


DEFAULT_COMMUNITIES = ["AskReddit", "aww", "funny", "science", "worldnews"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the paper reproduction dataset inside this repo.")
    parser.add_argument("--social-root", required=True, help="Path to the old social folder.")
    parser.add_argument("--output", default=str(ROOT / "data" / "social_paper"))
    parser.add_argument("--communities", nargs="+", default=DEFAULT_COMMUNITIES)
    parser.add_argument("--posts-per-community", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=200000)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_social_root": str(Path(args.social_root).resolve()),
        "posts_per_community": args.posts_per_community,
        "seed": args.seed,
        "communities": {},
    }

    for community in args.communities:
        print(f"\n== {community} ==")
        src = resolve_community_paths(args.social_root, community)
        dst = output / f"{community}_data"
        dst.mkdir(parents=True, exist_ok=True)

        posts = load_posts(src, enriched=False)
        selected = select_posts(posts, args.posts_per_community, seed=args.seed)
        post_ids = set(selected["post_id"].astype(str))
        print(f"selected_posts={len(post_ids)}")

        _write_selected_posts(src, dst, community, post_ids)
        comment_rows = _write_selected_comments(src.comments_csv, dst / f"comments_data_{community}.csv", post_ids, args.chunk_size)
        _copy_optional(src.calibration_json, dst / "calibration_summary.json")
        _copy_optional(src.population_json, dst / "population.json")

        comments = load_comments(resolve_community_paths(output, community))
        FrozenIntentPool.from_comments(comments, max_intents=5000, seed=args.seed).save_jsonl(dst / "frozen_intents.jsonl")
        _write_cascades(comments, post_ids, dst / "empirical_cascades_fixed.jsonl", dst / "empirical_summary_fixed.json")

        manifest["communities"][community] = {
            "selected_posts": len(post_ids),
            "comment_rows": comment_rows,
            "files": sorted(p.name for p in dst.glob("*")),
        }

    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"\nMaterialized paper data at {output}")


def _write_selected_posts(src, dst: Path, community: str, post_ids: Set[str]) -> None:
    for name in [
        f"posts_features_{community}.csv",
        f"posts_features_final_enriched_safe_{community}.csv",
    ]:
        source = src.base_dir / name
        if not source.exists():
            continue
        df = pd.read_csv(source, encoding="utf-8", encoding_errors="replace")
        df["post_id"] = df["post_id"].astype(str)
        df[df["post_id"].isin(post_ids)].to_csv(dst / name, index=False, encoding="utf-8")


def _write_selected_comments(source: Path, dest: Path, post_ids: Set[str], chunk_size: int) -> int:
    total = 0
    first = True
    for chunk in pd.read_csv(source, chunksize=chunk_size, encoding="utf-8", encoding_errors="replace"):
        chunk["post_id"] = chunk["post_id"].astype(str)
        out = chunk[chunk["post_id"].isin(post_ids)]
        if out.empty:
            continue
        out.to_csv(dest, mode="w" if first else "a", index=False, header=first, encoding="utf-8")
        first = False
        total += len(out)
        print(f"  comments_written={total}")
    if first:
        pd.DataFrame().to_csv(dest, index=False)
    return total


def _copy_optional(source: Path | None, dest: Path) -> None:
    if source and source.exists():
        shutil.copy2(source, dest)


def _write_cascades(comments: pd.DataFrame, post_ids: Iterable[str], out_jsonl: Path, out_summary: Path) -> None:
    features = []
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for post_id in post_ids:
            item = extract_cascade_features(comments, post_id)
            if not item:
                continue
            features.append(item)
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    if not features:
        summary = {"n_cascades": 0}
    else:
        import numpy as np

        summary = {
            "n_cascades": len(features),
            "size_mean": float(np.mean([x["size"] for x in features])),
            "max_depth_mean": float(np.mean([x["max_depth"] for x in features])),
            "mean_leaf_depth_mean": float(np.mean([x["mean_leaf_depth"] for x in features])),
            "depth_variance_mean": float(np.mean([x["depth_variance"] for x in features])),
        }
    with out_summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
