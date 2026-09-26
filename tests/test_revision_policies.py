import random
import unittest

from bdmtf.revision.policies import EventNode, NullConnectivityGraph, RedditRankingPolicy, TraitMode, sample_traits


class RevisionPolicyTest(unittest.TestCase):
    def test_rankings_are_deterministic_and_distinct(self):
        nodes = [
            EventNode("old", "post", 1, 0, likes=20, dislikes=1),
            EventNode("new", "post", 1, 100, likes=2, dislikes=1),
            EventNode("split", "post", 1, 50, likes=10, dislikes=9),
        ]
        self.assertEqual(RedditRankingPolicy("new").rank(nodes, 120)[0].node_id, "new")
        self.assertEqual(RedditRankingPolicy("top").rank(nodes, 120)[0].node_id, "old")
        self.assertEqual(RedditRankingPolicy("controversial").rank(nodes, 120)[0].node_id, "split")
        self.assertEqual(
            [node.node_id for node in RedditRankingPolicy("hot").rank(nodes, 120)],
            [node.node_id for node in RedditRankingPolicy("hot").rank(nodes, 120)],
        )

    def test_trait_modes_include_aggressive_constructive_agents(self):
        profiles = sample_traits(50, TraitMode.AGGRESSIVE_CONSTRUCTIVE, 4)
        self.assertTrue(any(item.antagonism >= 0.75 and item.prosocial >= 0.75 for item in profiles))
        coupled = sample_traits(20, TraitMode.COUPLED, 4)
        self.assertTrue(all(abs(item.prosocial - max(0.0, 1 - 1.5 * item.antagonism)) < 1e-12 for item in coupled))

    def test_null_connectivity_does_not_invent_edges(self):
        self.assertEqual(NullConnectivityGraph().neighbors("a"), frozenset())


if __name__ == "__main__":
    unittest.main()
