from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from .metrics import cohen_d


def write_paper_outputs(records: List[Dict[str, Any]], output: Path) -> None:
    paper_dir = output / "paper_tables"
    paper_dir.mkdir(parents=True, exist_ok=True)
    _write_table1(records, paper_dir / "table1_aggregate_structural_effects.csv")
    _write_table2(records, paper_dir / "table2_cross_community_collapse.csv")
    _write_volume_depth(records, paper_dir / "figure_volume_depth_points.csv")
    _write_readme(paper_dir / "README.txt")


def _write_table1(records: List[Dict[str, Any]], path: Path) -> None:
    baseline = [r for r in records if r["condition"] == "BASELINE"]
    toxic = [r for r in records if r["condition"] == "CORE_TOXIC_CONTROVERSIAL"]
    rows = []
    for metric in ["comment_volume", "max_depth", "mean_leaf_depth", "engagement_volume"]:
        b = [float(r[metric]) for r in baseline]
        t = [float(r[metric]) for r in toxic]
        b_mean = _mean(b)
        t_mean = _mean(t)
        rows.append(
            {
                "metric": metric,
                "baseline_mean": b_mean,
                "toxic_controversial_mean": t_mean,
                "change_ratio": _safe_div(t_mean, b_mean),
                "delta": t_mean - b_mean,
                "cohen_d_toxic_minus_baseline": cohen_d(t, b),
                "n_baseline": len(b),
                "n_toxic": len(t),
            }
        )
    _write_csv(path, rows)


def _write_table2(records: List[Dict[str, Any]], path: Path) -> None:
    rows = []
    by_community = defaultdict(list)
    for record in records:
        by_community[record["community"]].append(record)
    for community, items in sorted(by_community.items()):
        baseline = [r for r in items if r["condition"] == "BASELINE"]
        toxic = [r for r in items if r["condition"] == "CORE_TOXIC_CONTROVERSIAL"]
        rows.append(
            {
                "community": community,
                "baseline_volume": _mean(r["comment_volume"] for r in baseline),
                "toxic_volume": _mean(r["comment_volume"] for r in toxic),
                "volume_amplification": _safe_div(
                    _mean(r["comment_volume"] for r in toxic),
                    _mean(r["comment_volume"] for r in baseline),
                ),
                "baseline_max_depth": _mean(r["max_depth"] for r in baseline),
                "toxic_max_depth": _mean(r["max_depth"] for r in toxic),
                "delta_max_depth": _mean(r["max_depth"] for r in toxic)
                - _mean(r["max_depth"] for r in baseline),
                "baseline_mean_leaf_depth": _mean(r["mean_leaf_depth"] for r in baseline),
                "toxic_mean_leaf_depth": _mean(r["mean_leaf_depth"] for r in toxic),
                "delta_mean_leaf_depth": _mean(r["mean_leaf_depth"] for r in toxic)
                - _mean(r["mean_leaf_depth"] for r in baseline),
            }
        )
    _write_csv(path, rows)


def _write_volume_depth(records: List[Dict[str, Any]], path: Path) -> None:
    rows = [
        {
            "community": r["community"],
            "post_id": r["post_id"],
            "seed": r["seed"],
            "condition": r["condition"],
            "comment_volume": r["comment_volume"],
            "engagement_volume": r["engagement_volume"],
            "mean_leaf_depth": r["mean_leaf_depth"],
            "max_depth": r["max_depth"],
        }
        for r in records
        if r["condition"] in {"BASELINE", "CORE_TOXIC_CONTROVERSIAL"}
    ]
    _write_csv(path, rows)


def _write_readme(path: Path) -> None:
    path.write_text(
        "Generated paper reproduction outputs.\n"
        "table1_aggregate_structural_effects.csv compares BASELINE vs CORE_TOXIC_CONTROVERSIAL.\n"
        "table2_cross_community_collapse.csv reports the same comparison by community.\n"
        "figure_volume_depth_points.csv contains plotting data for the volume-depth trade-off.\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[Any]) -> float:
    vals = [float(v) for v in values]
    return float(np.mean(vals)) if vals else 0.0


def _safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else float(a / b)
