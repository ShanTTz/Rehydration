import math
import unittest
from dataclasses import replace

from bdmtf.intent_pool import FrozenIntentPool
from bdmtf.metrics import compute_metrics
from bdmtf.schema import AgentProfile, Intent, Intervention, RankingPolicy, SimulationConfig, ThreadState
from bdmtf.simulator import BDMTFSimulator


class SimulatorTest(unittest.TestCase):
    def test_intent_sessions_do_not_reuse_and_are_branch_independent(self):
        pool = FrozenIntentPool(
            [
                Intent("s1", None, "supportive", "supportive one"),
                Intent("s2", None, "supportive", "supportive two"),
            ]
        )
        first = pool.start_session()
        ids = [first.retrieve(1, "supportive").intent_id for _ in range(2)]
        self.assertEqual(ids, ["s1", "s2"])
        self.assertIsNone(first.retrieve(1, "supportive"))
        self.assertEqual(first.diagnostics()["exhausted"], 1)

        second = pool.start_session()
        self.assertEqual(second.retrieve(1, "supportive").intent_id, "s1")

    def test_capacity_multiplier_creates_finite_opportunities(self):
        pool = FrozenIntentPool(
            [Intent("s1", None, "supportive", "supportive one")]
        )
        session = pool.start_session(capacity_multiplier=2)
        first = session.retrieve(1, "supportive")
        second = session.retrieve(1, "supportive")
        self.assertEqual(first.metadata["base_intent_id"], "s1")
        self.assertEqual(second.metadata["opportunity_index"], 2)
        self.assertIsNone(session.retrieve(1, "supportive"))

    def test_behavioral_drives_follow_paper_mapping(self):
        agent = AgentProfile.from_dark_tetrad(
            1,
            {"machiavellianism": 2.5, "narcissism": 2.5, "psychopathy": 2.5, "sadism": 2.5},
        )
        self.assertAlmostEqual(agent.antagonism, 0.5)
        self.assertAlmostEqual(agent.attention, 0.5)
        self.assertAlmostEqual(agent.prosocial, 0.25)
        expected_threshold = 0.52 - 0.15 * 0.5
        self.assertAlmostEqual(agent.threshold, expected_threshold)

        low = AgentProfile.from_dark_tetrad(
            2,
            {"machiavellianism": 1, "narcissism": 1, "psychopathy": 1, "sadism": 1},
        )
        expected_drive = 1.0 / (1.0 + math.exp(0.6))
        self.assertAlmostEqual(low.antagonism, expected_drive)
        self.assertAlmostEqual(low.prosocial, max(0.0, 1.0 - 1.5 * expected_drive))

    def test_simulator_runs_paper_condition(self):
        config = SimulationConfig(num_agents=12, steps=6, enable_external_traffic=False)
        agents = [
            AgentProfile.from_dark_tetrad(
                i,
                {"machiavellianism": 2, "narcissism": 2, "psychopathy": 2, "sadism": 2},
                threshold_base=config.threshold_base,
                is_leader=i < 3,
            )
            for i in range(config.num_agents)
        ]
        sim = BDMTFSimulator(agents, FrozenIntentPool([]), config=config, seed=1)
        state, traces = sim.run(
            post_id="p",
            title="post",
            intervention=Intervention(
                name="toxic_cont",
                core="toxic",
                ranking=RankingPolicy.CONTROVERSIAL,
                context="hostile",
            ),
        )
        metrics = compute_metrics(state, traces)
        self.assertGreaterEqual(metrics["comment_volume"], 3)
        self.assertIn("mean_leaf_depth", metrics)
        self.assertIn("toxic_density", metrics)

    def test_neutral_thread_starts_from_post_root_without_counting_root(self):
        config = SimulationConfig(
            num_agents=20,
            steps=8,
            threshold_base=0.73,
            enable_external_traffic=False,
        )
        agents = [
            AgentProfile.from_dark_tetrad(
                i,
                {"machiavellianism": 1, "narcissism": 2, "psychopathy": 1, "sadism": 1},
                threshold_base=config.threshold_base,
                is_leader=i < 4,
            )
            for i in range(config.num_agents)
        ]
        sim = BDMTFSimulator(agents, FrozenIntentPool([]), config=config, seed=4)
        state, traces = sim.run(
            post_id="p",
            title="neutral post",
            initial_text="A neutral thread starter.",
            intervention=Intervention(name="baseline", core="baseline", ranking=RankingPolicy.BEST),
        )
        metrics = compute_metrics(state, traces)
        self.assertIn("post", state.comments)
        self.assertEqual(state.comments["post"].depth, 0)
        self.assertGreater(metrics["comment_volume"], 0)
        self.assertLess(metrics["comment_volume"], len(state.comments))

    def test_visibility_is_bounded_and_keeps_ranked_and_frontier_nodes(self):
        config = SimulationConfig(num_agents=1, steps=1, viewport_k=3, enable_external_traffic=False)
        sim = BDMTFSimulator([], FrozenIntentPool([]), config=config, seed=1)
        state = ThreadState(post_id="p", title="post")
        state.add_root_post()
        self.assertEqual([node.node_id for node in sim.visible_nodes(state, RankingPolicy.TOP)], ["post"])

        comments = [state.add_comment(i, str(i), "post", i, 0.0) for i in range(3)]
        comments[0].likes = 1
        comments[1].likes = 7
        comments[2].likes = 4
        visible = sim.visible_nodes(state, RankingPolicy.TOP)
        visible_ids = [node.node_id for node in visible]
        self.assertLessEqual(len(visible), config.viewport_k + 1)
        self.assertIn("post", visible_ids)
        self.assertIn(comments[1].node_id, visible_ids)
        self.assertIn(comments[2].node_id, visible_ids)

    def test_external_traffic_changes_only_root_post_reactions(self):
        config = SimulationConfig(
            num_agents=1,
            steps=1,
            external_lambda=10.0,
            initial_engagement_signal=100.0,
            early_engagement_median=100.0,
        )
        sim = BDMTFSimulator([], FrozenIntentPool([]), config=config, seed=2)
        state = ThreadState(post_id="p", title="post")
        root = state.add_root_post()
        comment = state.add_comment(1, "reply", "post", 0, 0.0)
        injected = sim._inject_external_traffic(state, 0, likes_this_step=2, replies_this_step=1)
        self.assertGreater(injected, 0)
        self.assertEqual(root.likes + root.dislikes, injected)
        self.assertEqual(comment.likes + comment.dislikes, 0)

    def test_channel_switches_remove_their_dynamic_paths(self):
        agent = AgentProfile(
            agent_id=1,
            antagonism=1.0,
            attention=1.0,
            prosocial=0.0,
            threshold=0.5,
        )
        state = ThreadState(post_id="p", title="post")
        node = state.add_root_post()
        node.likes = 5
        node.dislikes = 5
        node.reply_count = 20

        full = BDMTFSimulator(
            [agent],
            FrozenIntentPool([]),
            config=SimulationConfig(
                base_impulse=0.0,
                alpha_conflict=1.0,
                beta_heat=1.0,
                gamma_consensus=0.0,
                noise_width=0.0,
            ),
            seed=1,
        )
        no_conflict = BDMTFSimulator(
            [agent],
            FrozenIntentPool([]),
            config=SimulationConfig(
                base_impulse=0.0,
                alpha_conflict=1.0,
                beta_heat=1.0,
                gamma_consensus=0.0,
                noise_width=0.0,
                enable_conflict_channel=False,
            ),
            seed=1,
        )
        no_heat = BDMTFSimulator(
            [agent],
            FrozenIntentPool([]),
            config=SimulationConfig(
                base_impulse=0.0,
                alpha_conflict=1.0,
                beta_heat=1.0,
                gamma_consensus=0.0,
                noise_width=0.0,
                enable_heat_channel=False,
            ),
            seed=1,
        )

        full_impulse, signals = full._compute_impulse(agent, [node])
        conflict_off_impulse, _ = no_conflict._compute_impulse(agent, [node])
        heat_off_impulse, _ = no_heat._compute_impulse(agent, [node])
        self.assertAlmostEqual(
            full_impulse - conflict_off_impulse,
            signals["controversy"],
        )
        self.assertAlmostEqual(full_impulse - heat_off_impulse, signals["heat"])

        no_depth = BDMTFSimulator(
            [agent],
            FrozenIntentPool([]),
            config=SimulationConfig(enable_depth_targeting=False),
            seed=1,
        )
        self.assertEqual(
            no_depth._max_comment_depth(
                Intervention(name="baseline", core="baseline")
            ),
            0,
        )

    def test_frame_rendering_changes_text_but_not_structural_metrics(self):
        base = SimulationConfig(
            num_agents=12,
            steps=8,
            enable_external_traffic=False,
            intent_pool_capacity_multiplier=20,
        )
        agents = [
            AgentProfile.from_dark_tetrad(
                i,
                {
                    "machiavellianism": 2,
                    "narcissism": 2,
                    "psychopathy": 2,
                    "sadism": 2,
                },
                is_leader=i < 3,
            )
            for i in range(base.num_agents)
        ]
        pool = FrozenIntentPool(
            [
                Intent("s1", None, "supportive", "A source would help this claim."),
                Intent("a1", None, "antagonistic", "This ignores the evidence."),
            ]
        )
        intervention = Intervention(
            name="toxic",
            core="toxic",
            ranking=RankingPolicy.CONTROVERSIAL,
        )
        verbatim = BDMTFSimulator(agents, pool, config=base, seed=9)
        rendered = BDMTFSimulator(
            agents,
            pool,
            config=replace(base, semantic_payload_mode="frame_rendered"),
            seed=9,
        )
        state_a, traces_a = verbatim.run("p", "Evidence policy", intervention)
        state_b, traces_b = rendered.run("p", "Evidence policy", intervention)
        self.assertEqual(compute_metrics(state_a, traces_a), compute_metrics(state_b, traces_b))
        comments_a = [node for node in state_a.comments.values() if node.node_id != "post"]
        comments_b = [node for node in state_b.comments.values() if node.node_id != "post"]
        self.assertEqual(len(comments_a), len(comments_b))
        self.assertTrue(any(a.content != b.content for a, b in zip(comments_a, comments_b)))
        self.assertTrue(all("semantic_frame" in node.metadata for node in comments_b))


if __name__ == "__main__":
    unittest.main()
