import tempfile
import unittest
from pathlib import Path

from bdmtf.experiments import simulation_seed
from bdmtf.reviewer_validation import factorial_contrasts, write_factorial_outputs


def _records():
    cells = [
        ("BASELINE", 10, 8.0, 10),
        ("BASELINE_CONTROVERSIAL", 12, 7.0, 9),
        ("CORE_TOXIC_BEST", 20, 6.0, 8),
        ("CORE_TOXIC_CONTROVERSIAL", 30, 4.0, 7),
    ]
    return [
        {
            "community": "science",
            "post_id": "p1",
            "seed": 0,
            "condition": condition,
            "comment_volume": volume,
            "mean_leaf_depth": leaf,
            "max_depth": depth,
            "engagement_volume": volume * 2,
        }
        for condition, volume, leaf, depth in cells
    ]


class ReviewerValidationTest(unittest.TestCase):
    def test_paired_seed_is_shared_across_conditions(self):
        first = simulation_seed("science", "p1", "BASELINE", 0, "paired_by_post_seed")
        second = simulation_seed(
            "science", "p1", "CORE_TOXIC_CONTROVERSIAL", 0, "paired_by_post_seed"
        )
        self.assertEqual(first, second)
        self.assertNotEqual(
            simulation_seed("science", "p1", "BASELINE", 0),
            simulation_seed("science", "p1", "CORE_TOXIC_CONTROVERSIAL", 0),
        )

    def test_factorial_contrasts_recover_main_and_interaction(self):
        frame = factorial_contrasts(_records())
        self.assertEqual(len(frame), 1)
        row = frame.iloc[0]
        self.assertEqual(row["comment_volume__joint"], 20)
        self.assertEqual(row["comment_volume__core_at_best"], 10)
        self.assertEqual(row["comment_volume__ranking_at_baseline"], 2)
        self.assertEqual(row["comment_volume__core_ranking_interaction"], 8)
        self.assertEqual(row["mean_leaf_depth__joint"], -4)
        self.assertTrue(row["shallow_swarm_joint"])

    def test_incomplete_block_is_excluded(self):
        frame = factorial_contrasts(_records()[:-1])
        self.assertTrue(frame.empty)

    def test_outputs_include_manifest_and_clustered_ci(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs.jsonl"
            runs.write_text(
                "\n".join(__import__("json").dumps(item) for item in _records()) + "\n",
                encoding="utf-8",
            )
            manifest = write_factorial_outputs(
                runs,
                root / "analysis",
                bootstrap_samples=20,
                seed=7,
            )
            self.assertEqual(manifest["complete_post_seed_blocks"], 1)
            self.assertTrue((root / "analysis" / "factorial_summary.csv").is_file())
            self.assertTrue((root / "analysis" / "factorial_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
