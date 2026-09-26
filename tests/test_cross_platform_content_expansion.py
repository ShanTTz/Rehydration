from __future__ import annotations

import json
import zipfile

import pandas as pd

from bdmtf.revision.cross_platform_content_expansion import (
    _flatten_hn_comments,
    _merge_events,
    analyze_voat_author_content_transfer,
)


def test_flatten_hn_comments_preserves_nested_nodes() -> None:
    tree = {
        "children": [
            {
                "id": 2,
                "children": [{"id": 3, "children": []}],
            },
            {"id": 4, "children": []},
        ]
    }
    assert {item["id"] for item in _flatten_hn_comments(tree)} == {2, 3, 4}


def test_merge_events_deduplicates_platform_tree_nodes() -> None:
    first = pd.DataFrame(
        [
            {
                "platform": "HackerNews",
                "content_id": "1",
                "event_id": "1",
                "event_type": "post",
            }
        ]
    )
    second = pd.concat([first, first], ignore_index=True)
    merged = _merge_events(first, second)
    assert len(merged) == 1


def test_voat_transfer_outputs_only_hashed_authors(tmp_path) -> None:
    archive_path = tmp_path / "migration.zip"
    before_name = (
        "migration/2015-fatpeoplehate/"
        "2015-fatpeoplehate-in-before"
    )
    after_name = (
        "migration/2015-fatpeoplehate/"
        "2015-fatpeoplehate-out-after"
    )
    before = []
    after = []
    for index in range(12):
        author = f"raw-author-{index}"
        topic = f"topic{index}"
        for repeat in range(2):
            before.append(
                json.dumps(
                    {
                        "author": author,
                        "selftext": f"{topic} shared phrase before {repeat}",
                    }
                )
            )
            after.append(
                json.dumps(
                    {
                        "author": author,
                        "selftext": f"{topic} shared phrase after {repeat}",
                    }
                )
            )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(before_name, "\n".join(before))
        archive.writestr(after_name, "\n".join(after))

    result = analyze_voat_author_content_transfer(
        archive_path,
        tmp_path / "output",
        permutations=50,
        bootstrap_samples=50,
    )

    pairs = pd.read_csv(tmp_path / "output" / "voat_author_content_pairs.csv")
    assert result["status"] == "complete"
    assert len(pairs) == 12
    assert all(pairs["author_hash"].str.len().eq(20))
    assert not any(pairs.astype(str).apply(lambda column: column.str.contains("raw-author")).any())
