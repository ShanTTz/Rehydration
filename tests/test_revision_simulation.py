import unittest

import pandas as pd

from bdmtf.revision.fitted_models import CommunityModel
from bdmtf.revision.policies import EventNode
from bdmtf.revision.simulation import RevisionSimulationConfig, event_metrics, simulate_cascade


PROFILE = CommunityModel(
    community="test",
    n_train_cascades=10,
    log_size_mean=3.0,
    log_size_std=0.2,
    mean_branching_factor=1.5,
    root_reply_share=0.5,
    depth_decay=0.4,
    time_decay_minutes=100,
    repeat_author_share=0.2,
    toxicity_density=0.1,
    toxicity_log_size_effect=0.2,
    toxicity_leaf_depth_effect=-0.5,
    counterspeech_rate=0.1,
    removed_rate=0.05,
)


class RevisionSimulationTest(unittest.TestCase):
    def test_tree_invariants_and_determinism(self):
        config = RevisionSimulationConfig(max_comments=100, counterspeech_probability=0.2)
        train = pd.DataFrame([{"size": 20}])
        first = simulate_cascade("learned_bdmtf", PROFILE, train, "p", 7, config)
        second = simulate_cascade("learned_bdmtf", PROFILE, train, "p", 7, config)
        self.assertEqual([(n.node_id, n.parent_id) for n in first], [(n.node_id, n.parent_id) for n in second])
        ids = {node.node_id for node in first}
        self.assertEqual(len(ids), len(first))
        for node in first[1:]:
            self.assertIn(node.parent_id, ids)
            parent = next(item for item in first if item.node_id == node.parent_id)
            self.assertEqual(node.depth, parent.depth + 1)

    def test_metrics_are_complete(self):
        nodes = simulate_cascade("branching_process", PROFILE, pd.DataFrame(), "p", 2, RevisionSimulationConfig(max_comments=50))
        metrics = event_metrics(nodes)
        self.assertGreater(metrics["size"], 0)
        self.assertIn("time_to_90_minutes", metrics)
        self.assertIn("repeat_author_share", metrics)

    def test_branching_factor_excludes_root_post(self):
        nodes = [
            EventNode("post", None, 0, 0, metadata={"root_post": True}),
            EventNode("a", "post", 1, 1),
            EventNode("b", "post", 1, 2),
            EventNode("c", "post", 1, 3),
            EventNode("d", "a", 2, 4),
            EventNode("e", "a", 2, 5),
        ]
        self.assertEqual(event_metrics(nodes)["mean_branching_factor"], 2.0)


if __name__ == "__main__":
    unittest.main()
