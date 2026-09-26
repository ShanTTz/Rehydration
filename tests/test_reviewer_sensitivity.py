import unittest

from bdmtf.reviewer_sensitivity import (
    apply_parameter_draw,
    latin_hypercube_draws,
    summarize_sensitivity_records,
)


class ReviewerSensitivityTest(unittest.TestCase):
    def test_latin_hypercube_is_deterministic_and_bounded(self):
        parameters = [
            {"name": "scale", "low": 0.8, "high": 1.2},
            {"name": "count", "low": 4, "high": 6, "integer": True},
        ]
        first = latin_hypercube_draws(parameters, 8, 7)
        second = latin_hypercube_draws(parameters, 8, 7)
        self.assertEqual(first, second)
        self.assertTrue(all(0.8 <= item["scale"] <= 1.2 for item in first))
        self.assertTrue(all(item["count"] in {4, 5, 6} for item in first))

    def test_draw_updates_global_and_community_override(self):
        source = {
            "simulation": {"action_probability": 0.3, "viewport_k": 5},
            "community_calibration": {
                "science": {"action_probability": 0.2},
                "aww": {},
            },
        }
        parameters = [
            {
                "name": "action_scale",
                "target": "action_probability",
                "community_field": "action_probability",
                "mode": "scale",
                "low": 0.8,
                "high": 1.2,
            },
            {
                "name": "viewport",
                "target": "viewport_k",
                "mode": "value",
                "integer": True,
                "low": 4,
                "high": 6,
            },
        ]
        result = apply_parameter_draw(
            source,
            parameters,
            {"action_scale": 1.1, "viewport": 6},
        )
        self.assertAlmostEqual(result["simulation"]["action_probability"], 0.33)
        self.assertAlmostEqual(
            result["community_calibration"]["science"]["action_probability"],
            0.22,
        )
        self.assertEqual(result["simulation"]["viewport_k"], 6)
        self.assertEqual(source["simulation"]["viewport_k"], 5)

    def test_summary_detects_robust_scenarios(self):
        records = []
        for scenario, treated_volume, treated_leaf in (
            ("lhs_000", 30, 4.0),
            ("lhs_001", 8, 9.0),
        ):
            for condition, volume, leaf in (
                ("BASELINE", 10, 8.0),
                ("CORE_TOXIC_CONTROVERSIAL", treated_volume, treated_leaf),
            ):
                records.append(
                    {
                        "scenario": scenario,
                        "community": "science",
                        "post_id": "p1",
                        "seed": 0,
                        "condition": condition,
                        "comment_volume": volume,
                        "mean_leaf_depth": leaf,
                        "max_depth": leaf + 2,
                        "parameter__x": 1.0,
                    }
                )
        blocks, scenarios, manifest = summarize_sensitivity_records(records)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(scenarios), 2)
        self.assertEqual(manifest["supporting_scenarios"], 1)
        self.assertEqual(manifest["supporting_scenario_share"], 0.5)


if __name__ == "__main__":
    unittest.main()
