from __future__ import annotations

import pandas as pd

from scripts.analyze_live_semantic_blind_review import SOURCES, long_ratings


def test_long_ratings_decodes_three_blinded_candidates() -> None:
    master = pd.DataFrame(
        [
            {
                "assignment_id": "SEM-001",
                "item_number": 1,
                "item_id": "item-1",
                "community": "science",
                "condition": "BASELINE",
                "depth_bin": "shallow",
                "candidate_A_source": SOURCES[0],
                "candidate_B_source": SOURCES[1],
                "candidate_C_source": SOURCES[2],
            }
        ]
    )
    completed = pd.DataFrame(
        [
            {
                "assignment_id": "SEM-001",
                "item_number": 1,
                "relevance_A_1_to_5": 3,
                "coherence_A_1_to_5": 4,
                "intent_preservation_A_1_to_5": 3,
                "obvious_mismatch_A_0_or_1": 0,
                "relevance_B_1_to_5": 4,
                "coherence_B_1_to_5": 4,
                "intent_preservation_B_1_to_5": 5,
                "obvious_mismatch_B_0_or_1": 0,
                "relevance_C_1_to_5": 5,
                "coherence_C_1_to_5": 5,
                "intent_preservation_C_1_to_5": 2,
                "obvious_mismatch_C_0_or_1": 0,
                "best_direct_response_A_B_C_or_tie": "C",
            }
        ]
    )
    ratings = long_ratings(master, completed)
    assert list(ratings["source"]) == list(SOURCES)
    assert ratings["direct_acceptable"].tolist() == [1.0, 1.0, 1.0]
    assert ratings["intent_compatible"].tolist() == [1.0, 1.0, 0.0]
    assert ratings["best"].tolist() == [0.0, 0.0, 1.0]
