from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdmtf.intent_pool import FrozenIntentPool
from bdmtf.metrics import compute_metrics
from bdmtf.schema import AgentProfile, Intervention, RankingPolicy, SimulationConfig
from bdmtf.simulator import BDMTFSimulator


def main() -> None:
    config = SimulationConfig(num_agents=20, steps=12, viewport_k=5, enable_external_traffic=True)
    agents = [
        AgentProfile.from_dark_tetrad(
            i,
            {
                "machiavellianism": 2 + (i % 4 == 0),
                "narcissism": 2 + (i % 5 == 0),
                "psychopathy": 2 + (i % 6 == 0),
                "sadism": 2 + (i % 7 == 0),
            },
            threshold_base=config.threshold_base,
            is_leader=i < 4,
        )
        for i in range(config.num_agents)
    ]
    pool = FrozenIntentPool([])
    sim = BDMTFSimulator(agents, pool, config=config, seed=7)
    state, traces = sim.run(
        post_id="synthetic",
        title="Synthetic post",
        intervention=Intervention(
            name="core_toxic_controversial",
            core="toxic",
            ranking=RankingPolicy.CONTROVERSIAL,
            context="hostile",
        ),
    )
    print(compute_metrics(state, traces))


if __name__ == "__main__":
    main()
