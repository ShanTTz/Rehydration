from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bdmtf.revision.reference_policy_strengthening import freeze_protocol, run_community, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reference_policy_strengthening.json")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    path = ROOT / args.config
    config, manifest = freeze_protocol(ROOT, path)
    print(json.dumps({"posts": manifest["post_count"], "runs": manifest["expected_runs"], "status": "frozen"}), flush=True)
    if args.prepare_only:
        return
    if not args.summarize_only:
        communities = sorted({p["community"] for p in manifest["evaluation_ids"]})
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_community, ROOT, path, c) for c in communities]
            for community, future in zip(communities, futures):
                print(json.dumps({"community": community, "runs": future.result()}), flush=True)
    print(json.dumps(summarize(ROOT, path), indent=2), flush=True)


if __name__ == "__main__":
    main()
