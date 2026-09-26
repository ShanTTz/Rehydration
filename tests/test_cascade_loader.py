import unittest

import pandas as pd

from bdmtf.data.social_loader import extract_cascade_features, strip_reddit_prefix


class CascadeLoaderTest(unittest.TestCase):
    def test_strip_reddit_prefix(self):
        self.assertEqual(strip_reddit_prefix("t1_abc"), "abc")
        self.assertEqual(strip_reddit_prefix("t3_post"), "post")
        self.assertEqual(strip_reddit_prefix("plain"), "plain")

    def test_extract_cascade_depth_with_prefixed_parent(self):
        df = pd.DataFrame(
            [
                {"post_id": "p1", "comment_id": "a", "parent_id": "t3_p1", "score": 1},
                {"post_id": "p1", "comment_id": "b", "parent_id": "t1_a", "score": 1},
                {"post_id": "p1", "comment_id": "c", "parent_id": "t1_b", "score": 1},
            ]
        )
        features = extract_cascade_features(df, "p1")
        self.assertEqual(features["max_depth"], 3)
        self.assertEqual(features["mean_leaf_depth"], 3.0)


if __name__ == "__main__":
    unittest.main()
