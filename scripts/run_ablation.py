from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.experiments import config_from_dict, interventions_from_config, run_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run motivational-channel ablations.")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "social_paper"))
    parser.add_argument("--output", default=str(ROOT / "run_outputs" / "ablation"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper_reproduction.json"))
    parser.add_argument("--posts-per-community", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = parser.parse_args()

    raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
    base_config = config_from_dict(raw)
    toxic = [item for item in interventions_from_config(raw) if item.name == "CORE_TOXIC_CONTROVERSIAL"]
    variants = {
        "ALL": base_config,
        "NO_CONFLICT": replace(base_config, alpha_conflict=0.0),
        "NO_HEAT": replace(base_config, beta_heat=0.0),
    }
    all_rows = []
    for label, config in variants.items():
        out = Path(args.output) / label
        records = run_batch(
            social_root=args.data_root,
            output_dir=out,
            communities=raw["communities"],
            posts_per_community=args.posts_per_community,
            seeds=args.seeds,
            config=config,
            interventions=toxic,
        )
        for record in records:
            record["ablation"] = label
            all_rows.append(record)
    _write_rows(Path(args.output) / "ablation_runs.csv", all_rows)
    print(f"Ablation complete: {Path(args.output)}")


def _write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
