from __future__ import annotations

import unittest

import numpy as np

from bdmtf.revision.ground_truth_scm import OutcomeSpec, simulate_trial


class GroundTruthSCMTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = OutcomeSpec(
            name="test",
            treatment_effect=0.6,
            semantic_main=0.8,
            semantic_interaction=0.35,
            structural_noise=1.0,
            structural_noise_interaction=0.3,
        )

    def test_rehydration_recovers_truth_in_large_sample(self) -> None:
        result = simulate_trial(
            np.random.default_rng(7),
            units=200_000,
            semantic_shift=0.8,
            proposal_noise=1.0,
            spec=self.spec,
        )
        self.assertAlmostEqual(
            result["rehydration"]["estimate"],
            self.spec.treatment_effect,
            delta=0.005,
        )

    def test_common_seed_does_not_remove_semantic_shift_bias(self) -> None:
        shift = 0.8
        result = simulate_trial(
            np.random.default_rng(11),
            units=200_000,
            semantic_shift=shift,
            proposal_noise=1.0,
            spec=self.spec,
        )
        expected_bias = (
            self.spec.semantic_main + self.spec.semantic_interaction
        ) * shift
        observed_bias = (
            result["common_seed_live"]["estimate"]
            - self.spec.treatment_effect
        )
        self.assertAlmostEqual(observed_bias, expected_bias, delta=0.01)


if __name__ == "__main__":
    unittest.main()

