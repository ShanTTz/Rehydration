import unittest

import pandas as pd

from bdmtf.revision.external_data import external_cascade_metrics, normalize_events
from bdmtf.revision.external_validation import summarize_cross_platform_fidelity
from bdmtf.revision.fitted_models import fit_pooled_model
from bdmtf.revision.interventions import estimate_matched_difference_in_differences, intervention_readiness


class ExternalValidityTest(unittest.TestCase):
    def test_cross_platform_summary_reports_platform_counts(self):
        fidelity = pd.DataFrame(
            {
                "model": ["m", "m"],
                "normalized_wasserstein": [0.2, 0.3],
            }
        )
        targets = pd.DataFrame(
            {
                "platform": ["HackerNews", "HackerNews", "Lemmy"],
            }
        )

        summary = summarize_cross_platform_fidelity(fidelity, targets, 3)

        self.assertEqual(
            summary["n_target_cascades_by_platform"],
            {"HackerNews": 2, "Lemmy": 1},
        )

    def test_external_event_normalization_and_depth(self):
        source = pd.DataFrame(
            [
                {"thread": "t", "id": "root", "parent": "", "time": "2025-01-01T00:00:00Z", "kind": "post"},
                {"thread": "t", "id": "a", "parent": "root", "time": "2025-01-01T00:01:00Z", "kind": "comment"},
                {"thread": "t", "id": "b", "parent": "a", "time": "2025-01-01T00:02:00Z", "kind": "comment"},
            ]
        )
        events = normalize_events(
            source,
            {"content_id": "thread", "event_id": "id", "parent_event_id": "parent", "created_at": "time", "event_type": "kind"},
            platform="example",
            community="forum",
        )
        self.assertEqual(events["depth"].tolist(), [1, 2, 3])
        metrics = external_cascade_metrics(events).iloc[0]
        self.assertEqual(metrics["size"], 2.0)
        self.assertEqual(metrics["max_depth"], 2.0)
        self.assertEqual(metrics["mean_leaf_depth"], 2.0)

    def test_event_cycle_is_rejected(self):
        source = pd.DataFrame(
            [
                {"content_id": "t", "event_id": "a", "parent_event_id": "b", "created_at": "2025-01-01T00:00:00Z"},
                {"content_id": "t", "event_id": "b", "parent_event_id": "a", "created_at": "2025-01-01T00:01:00Z"},
            ]
        )
        with self.assertRaisesRegex(ValueError, "Cycle"):
            normalize_events(source)

    def test_missing_parent_below_explicit_root_is_rejected(self):
        source = pd.DataFrame(
            [
                {"content_id": "t", "event_id": "root", "parent_event_id": "", "created_at": "2025-01-01T00:00:00Z", "event_type": "post"},
                {"content_id": "t", "event_id": "a", "parent_event_id": "missing", "created_at": "2025-01-01T00:01:00Z", "event_type": "comment"},
            ]
        )
        with self.assertRaisesRegex(ValueError, "missing parent"):
            normalize_events(source)

    def test_pooled_fit_excludes_target_community(self):
        rows = []
        for community, size in (("source_a", 10), ("source_b", 20), ("target", 1000)):
            rows.append(
                {
                    "community": community,
                    "split": "train",
                    "size": size,
                    "mean_depth": 2,
                    "mean_branching_factor": 1.5,
                    "root_reply_share": 0.5,
                    "time_to_90_minutes": 100,
                    "repeat_author_share": 0.2,
                    "toxicity_density": 0.1,
                    "mean_leaf_depth": 2,
                    "counterspeech_rate": 0.05,
                    "removed_rate": 0.01,
                }
            )
        profile = fit_pooled_model(pd.DataFrame(rows), ("target",), "zero_shot")
        self.assertEqual(profile.n_train_cascades, 2)
        self.assertLess(profile.log_size_mean, 4.0)

    def test_matched_did_recovers_known_effect(self):
        rows = []
        for index in range(20):
            treated = int(index >= 10)
            pre = float(index % 10)
            rows.append(
                {
                    "unit_id": f"u{index}",
                    "domain": "d",
                    "treatment": treated,
                    "event_time": "2025-01-01T00:00:00Z",
                    "pre_outcome": pre,
                    "post_outcome": pre + 1.0 + 2.0 * treated,
                    "baseline_size": pre,
                }
            )
        frame = pd.DataFrame(rows)
        self.assertTrue(intervention_readiness(frame)["claim_allowed"])
        result, pairs = estimate_matched_difference_in_differences(frame, ["baseline_size"], bootstrap_samples=100, seed=7)
        self.assertAlmostEqual(result["effect"], 2.0)
        self.assertEqual(len(pairs), 10)

    def test_removed_text_without_event_time_is_not_ready(self):
        readiness = intervention_readiness(pd.DataFrame([{"unit_id": "x", "treatment": 1}]))
        self.assertFalse(readiness["claim_allowed"])
        self.assertIn("event_time", readiness["missing_columns"])


if __name__ == "__main__":
    unittest.main()
