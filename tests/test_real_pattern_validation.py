import unittest

import pandas as pd

from bdmtf.real_pattern_validation import _thread_windows, analyze_heldout_patterns


class RealPatternValidationTest(unittest.TestCase):
    def test_thread_windows_do_not_mix_future_text_into_exposure(self):
        comments = pd.DataFrame(
            [
                {
                    "comment_id": "t1_a",
                    "parent_id": "t3_p",
                    "minutes_since_post": 10,
                    "depth": 0,
                    "author": "u1",
                    "comment_text": "troll",
                },
                {
                    "comment_id": "t1_b",
                    "parent_id": "t3_p",
                    "minutes_since_post": 20,
                    "depth": 0,
                    "author": "u2",
                    "comment_text": "thanks",
                },
                {
                    "comment_id": "t1_c",
                    "parent_id": "t1_a",
                    "minutes_since_post": 30,
                    "depth": 1,
                    "author": "u3",
                    "comment_text": "neutral",
                },
                {
                    "comment_id": "t1_d",
                    "parent_id": "t1_c",
                    "minutes_since_post": 90,
                    "depth": 2,
                    "author": "u4",
                    "comment_text": "troll troll",
                },
                {
                    "comment_id": "t1_e",
                    "parent_id": "t3_p",
                    "minutes_since_post": 120,
                    "depth": 0,
                    "author": "u5",
                    "comment_text": "troll",
                },
                {
                    "comment_id": "t1_f",
                    "parent_id": "t1_d",
                    "minutes_since_post": 180,
                    "depth": 3,
                    "author": "u6",
                    "comment_text": "troll",
                },
            ]
        )
        row, reason = _thread_windows(
            comments,
            "science",
            "p",
            "test",
            60,
            720,
            3,
            3,
        )
        self.assertEqual(reason, "")
        self.assertEqual(row["early_comment_count"], 3)
        self.assertEqual(row["late_comment_count"], 3)
        self.assertAlmostEqual(row["early_toxicity_density"], 1 / 3)
        self.assertEqual(row["depth_growth"], 2)

    def test_heldout_analysis_uses_declared_splits(self):
        rows = []
        communities = ["science", "worldnews"]
        for index in range(80):
            split = "train" if index < 50 else "test"
            toxicity = (index % 10) / 10
            late = 1.0 + 0.8 * toxicity
            rows.append(
                {
                    "community": communities[index % 2],
                    "post_id": f"p{index}",
                    "split": split,
                    "early_toxicity_density": toxicity,
                    "log1p_early_comments": 2.0 + (index % 3) * 0.1,
                    "early_max_depth": 2.0,
                    "early_root_reply_share": 0.5,
                    "early_removed_rate": 0.0,
                    "log1p_late_comments": late,
                    "late_mean_leaf_depth": 4.0 - toxicity,
                    "depth_growth": 2.0 - toxicity,
                    "late_root_reply_share": 0.4 + toxicity * 0.2,
                }
            )
        results, manifest = analyze_heldout_patterns(
            pd.DataFrame(rows),
            bootstrap_samples=100,
            seed=7,
        )
        self.assertEqual(manifest["n_train"], 50)
        self.assertEqual(manifest["n_test"], 30)
        volume = results[results["outcome"] == "log1p_late_comments"].iloc[0]
        self.assertGreater(volume["heldout_partial_slope"], 0)
        self.assertTrue(volume["supports_expected_direction"])


if __name__ == "__main__":
    unittest.main()
