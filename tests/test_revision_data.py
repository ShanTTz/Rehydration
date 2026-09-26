import tempfile
import unittest
from pathlib import Path

import pandas as pd

from bdmtf.revision.data_pipeline import load_author_history_before, parse_timestamp


class RevisionDataTest(unittest.TestCase):
    def test_author_history_is_truncated_before_post(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "authors.csv"
            pd.DataFrame(
                [
                    {"created_utc": "2025-01-01T00:00:00Z", "text": "past"},
                    {"created_utc": "2025-02-01T00:00:00Z", "text": "future"},
                ]
            ).to_csv(path, index=False)
            frame = load_author_history_before(path, pd.Timestamp("2025-01-15", tz="UTC"))
            self.assertEqual(frame["text"].tolist(), ["past"])

    def test_parse_timestamp_handles_mixed_iso_precision_and_unix_seconds(self):
        mixed = parse_timestamp(
            pd.Series(
                [
                    "2026-05-24T15:22:36.950296Z",
                    "2026-05-24T04:18:40Z",
                    1_700_000_000,
                    "1700000001",
                ]
            )
        )

        self.assertTrue(mixed.notna().all())
        self.assertEqual(mixed.iloc[0].year, 2026)
        self.assertEqual(mixed.iloc[1].hour, 4)
        self.assertEqual(mixed.iloc[2].year, 2023)
        self.assertEqual((mixed.iloc[3] - mixed.iloc[2]).total_seconds(), 1)


if __name__ == "__main__":
    unittest.main()
