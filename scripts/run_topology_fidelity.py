from __future__ import annotations

import argparse
import json
from pathlib import Path

from bdmtf.revision.topology_fidelity import (
    freeze_topology_design,
    run_topology_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run held-out topology fidelity and learned interaction replication."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/topology_fidelity_learned_interaction.json"),
    )
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--force-freeze", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    frozen = freeze_topology_design(root, config_path, force=args.force_freeze)
    result = frozen if args.freeze_only else run_topology_experiment(root, config_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
