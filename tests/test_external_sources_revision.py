from __future__ import annotations

import io
import json
import zipfile

import pandas as pd
import pytest

from bdmtf.revision.contracts import event_from_mapping
from bdmtf.revision import hackernews
from bdmtf.revision.lemmy_outcomes import (
    _match_lemmy_risk_sets,
    build_lemmy_cascade_events,
)
from bdmtf.revision.external_sources import (
    TBBT_INTERVENTIONS,
    build_story_matches,
    canonicalize_url,
    import_tbbt,
    make_prospective_split,
    parse_lemmy_modlog,
    resolve_story_urls,
)
from bdmtf.revision.reddit_archives import import_reddit_archive_bundles
from bdmtf.revision.reddit_expansion_validation import build_expanded_reddit_metrics
from bdmtf.revision.workflows import make_prospective_split_workflow


def test_nested_reddit_archive_import_builds_complete_reply_tree(tmp_path) -> None:
    posts = (
        "post_id,title,author,created_utc,subreddit,final_score,url,full_text,is_viral\n"
        "p1,Fixture post,author0,1700000000,testgroup,10,https://example.test/p1,body,True\n"
        "pbad,Bad timestamp,author0,1e999,testgroup,0,,,False\n"
    )
    comments = (
        "post_id,comment_id,author,score,created_utc,depth,parent_id,comment_text\n"
        "p1,c1,author1,2,1700000010,0,t3_p1,root reply\n"
        "p1,c2,author2,1,1700000020,1,t1_c1,nested reply\n"
    )
    nested_buffer = io.BytesIO()
    with zipfile.ZipFile(nested_buffer, "w", zipfile.ZIP_DEFLATED) as nested:
        nested.writestr("posts_features_testgroup.csv", posts)
        nested.writestr("comments_data_testgroup.csv", comments)

    outer_path = tmp_path / "reddit.zip"
    with zipfile.ZipFile(outer_path, "w", zipfile.ZIP_DEFLATED) as outer:
        outer.writestr("reddit/testgroup_data.zip", nested_buffer.getvalue())

    output = tmp_path / "output"
    manifest = import_reddit_archive_bundles(
        [outer_path],
        output,
        ["testgroup"],
        minimum_communities=1,
        minimum_cascades=1,
    )
    events = pd.read_parquet(output / "reddit_expanded_events.parquet").sort_values("depth")

    assert manifest["status"] == "complete"
    assert manifest["n_communities"] == 1
    assert manifest["n_cascades"] == 1
    assert manifest["community_manifests"][0]["cascades_excluded"] == 1
    assert events["depth"].tolist() == [1, 2, 3]
    assert events["parent_event_id"].tolist() == [
        "",
        "reddit:post:p1",
        "reddit:comment:c1",
    ]

    metrics, metrics_manifest = build_expanded_reddit_metrics(
        output / "communities",
        output / "metrics",
    )
    assert metrics_manifest["n_cascades"] == 1
    assert metrics.iloc[0]["size"] == 2
    assert metrics.iloc[0]["max_depth"] == 2
    assert metrics.iloc[0]["mean_leaf_depth"] == 2


def test_tbbt_catalog_has_25_real_interventions() -> None:
    assert len(TBBT_INTERVENTIONS) == 25
    assert {item["type"] for item in TBBT_INTERVENTIONS} == {
        "ban",
        "post_removal",
        "quarantine",
        "migration",
    }


def test_hackernews_corrupt_cache_is_refetched_atomically(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "7.json").write_text('{"id":', encoding="utf-8")
    monkeypatch.setattr(
        hackernews,
        "_request_json",
        lambda _url: {"id": 7, "type": "story", "time": 1},
    )

    item = hackernews._fetch_item(7, cache, "https://example.test")

    assert item == {"id": 7, "type": "story", "time": 1}
    assert json.loads((cache / "7.json").read_text(encoding="utf-8")) == item
    assert not (cache / "7.json.tmp").exists()


def test_hackernews_global_traversal_excludes_incomplete_story(tmp_path, monkeypatch) -> None:
    items = {
        1: {"id": 1, "type": "story", "time": 1, "descendants": 1, "kids": [11]},
        2: {"id": 2, "type": "story", "time": 1, "descendants": 1, "kids": [21]},
        3: {"id": 3, "type": "story", "time": 1, "descendants": 1, "kids": [31]},
        11: {"id": 11, "type": "comment", "time": 2, "parent": 1},
        21: None,
        31: {"id": 31, "type": "comment", "time": 2, "parent": 3},
    }

    def fake_request(url: str):
        if url.endswith("/topstories.json"):
            return [1, 2, 3]
        return items[int(url.rsplit("/", 1)[1].split(".", 1)[0])]

    monkeypatch.setattr(hackernews, "_request_json", fake_request)
    frame, manifest = hackernews.collect_hackernews(
        tmp_path,
        story_count=2,
        max_comments_per_story=5,
        workers=2,
        base_url="https://example.test",
    )

    assert set(frame["content_id"]) == {"1", "3"}
    assert manifest["target_met"] is True
    assert manifest["incomplete_stories_excluded"] == 1
    assert manifest["structurally_complete_stories"] == 2


def test_hackernews_records_single_item_failure_without_aborting(tmp_path, monkeypatch) -> None:
    items = {
        1: {"id": 1, "type": "story", "time": 1, "descendants": 1, "kids": [11]},
        3: {"id": 3, "type": "story", "time": 1, "descendants": 1, "kids": [31]},
        4: {"id": 4, "type": "story", "time": 1, "descendants": 1, "kids": [41]},
        11: {"id": 11, "type": "comment", "time": 2, "parent": 1},
        31: {"id": 31, "type": "comment", "time": 2, "parent": 3},
        41: {"id": 41, "type": "comment", "time": 2, "parent": 4},
    }

    def fake_request(url: str):
        if url.endswith("/topstories.json"):
            return [1, 2, 3, 4]
        item_id = int(url.rsplit("/", 1)[1].split(".", 1)[0])
        if item_id == 2:
            raise RuntimeError("fixture timeout")
        return items[item_id]

    monkeypatch.setattr(hackernews, "_request_json", fake_request)
    frame, manifest = hackernews.collect_hackernews(
        tmp_path,
        story_count=2,
        max_comments_per_story=5,
        workers=2,
        base_url="https://example.test",
    )

    assert frame["content_id"].nunique() == 2
    assert manifest["target_met"] is True
    assert manifest["status"] == "partial"
    assert manifest["request_failure_count"] == 1


def test_hackernews_algolia_discovery_is_id_only_and_audited(monkeypatch) -> None:
    def fake_request(url: str):
        assert "tags=story" in url
        assert "num_comments" in url
        return {
            "hits": [
                {"objectID": "101", "created_at_i": 30},
                {"objectID": "102", "created_at_i": 20},
                {"objectID": "103", "created_at_i": 10},
            ]
        }

    monkeypatch.setattr(hackernews, "_request_json", fake_request)
    ids, audit = hackernews._discover_story_ids_algolia(2, 300)

    assert ids == [101, 102]
    assert audit["provider"] == "HN Search powered by Algolia"
    assert audit["content_source"] == "Official Hacker News Firebase item API"
    assert audit["query_count"] == 1


def test_hackernews_excludes_temporally_reparented_story(tmp_path, monkeypatch) -> None:
    items = {
        1: {"id": 1, "type": "story", "time": 1, "descendants": 1, "kids": [11]},
        2: {"id": 2, "type": "story", "time": 1, "descendants": 2, "kids": [22]},
        3: {"id": 3, "type": "story", "time": 1, "descendants": 1, "kids": [31]},
        11: {"id": 11, "type": "comment", "time": 2, "parent": 1},
        21: {"id": 21, "type": "comment", "time": 2, "parent": 22},
        22: {"id": 22, "type": "comment", "time": 3, "parent": 2, "kids": [21]},
        31: {"id": 31, "type": "comment", "time": 2, "parent": 3},
    }

    def fake_request(url: str):
        if url.endswith("/topstories.json"):
            return [1, 2, 3]
        return items[int(url.rsplit("/", 1)[1].split(".", 1)[0])]

    monkeypatch.setattr(hackernews, "_request_json", fake_request)
    frame, manifest = hackernews.collect_hackernews(
        tmp_path,
        story_count=2,
        max_comments_per_story=5,
        workers=2,
        base_url="https://example.test",
    )

    assert set(frame["content_id"]) == {"1", "3"}
    assert manifest["target_met"] is True
    assert manifest["temporally_reparented_stories_excluded"] == 1


def test_platform_event_mapping_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="Missing PlatformEvent fields"):
        event_from_mapping({"platform": "reddit"})


def test_tbbt_streaming_import_handles_partial_fixture(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    fixture = [
        {
            "id": "one",
            "created_utc": 1577836800,
            "subreddit": "treated",
            "author": "a",
            "score": 2,
            "removal_reason": None,
        },
        {
            "id": "two",
            "created_utc": 1577836860,
            "subreddit": "treated",
            "author": "b",
            "score": 3,
            "removal_reason": "moderator",
        },
    ]
    with zipfile.ZipFile(raw / "ban.zip", "w") as archive:
        archive.writestr(
            "ban/treated/in_before.jsonl",
            "\n".join(json.dumps(item) for item in fixture),
        )
    result = import_tbbt(raw, tmp_path / "out", max_records_per_archive=10)
    panel = pd.read_csv(tmp_path / "out" / "tbbt_daily_panel.csv")
    assert result["status"] == "partial"
    assert result["records_read"] == 2
    assert panel.iloc[0]["messages"] == 2
    assert round(panel.iloc[0]["active_authors_estimate"]) == 2


def test_tbbt_migration_maps_reddit_and_voat_to_one_intervention(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / "migration.zip", "w") as archive:
        archive.writestr(
            "migration/2015-fatpeoplehate/2015-fatpeoplehate-in-before",
            json.dumps(
                {
                    "id": "reddit-one",
                    "created_utc": 1433808000,
                    "subreddit": "fatpeoplehate",
                    "author": "a",
                }
            ),
        )
        archive.writestr(
            "migration/2015-fatpeoplehate/2015-fatpeoplehate-out-after",
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "voat-one",
                            "created_utc": 1433980800,
                            "subverse": "fatpeoplehate",
                            "author": "a",
                        }
                    ),
                    json.dumps(
                        {
                            "id": "voat-two",
                            "created_utc": 1433980801,
                            "subverse": "other",
                            "author": "b",
                        }
                    ),
                ]
            ),
        )
    result = import_tbbt(
        raw,
        tmp_path / "out",
        max_records_per_archive=10,
        categories={"migration"},
    )
    panel = pd.read_csv(tmp_path / "out" / "tbbt_daily_panel.csv")
    assert result["selected_categories"] == ["migration"]
    assert panel["intervention_id"].nunique() == 1
    assert set(panel["platform"]) == {"reddit", "voat"}
    voat = panel.loc[panel["platform"] == "voat"].iloc[0]
    assert voat["community_id"] == "fatpeoplehate"
    assert voat["messages"] == 2
    outcome_manifest = json.loads(
        (tmp_path / "out" / "outcome_panel_manifest.json").read_text(encoding="utf-8")
    )
    assert outcome_manifest["status"] == "descriptive_ready"
    assert outcome_manifest["claim_allowed"] is False
    assert "never-treated" in outcome_manifest["prohibited_use"]
    outcome = pd.read_csv(tmp_path / "out" / "tbbt_outcome_panel.csv")
    assert not outcome.duplicated(["unit_id", "period", "outcome"]).any()


def test_tbbt_duplicate_slug_is_separated_by_intervention_type(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    record = {
        "id": "one",
        "created_utc": 1536624000,
        "subreddit": "greatawakening",
        "author": "a",
    }
    with zipfile.ZipFile(raw / "ban.zip", "w") as archive:
        archive.writestr(
            "ban/2018-greatawakening/2018-greatawakening-in-before",
            json.dumps(record),
        )
    with zipfile.ZipFile(raw / "migration.zip", "w") as archive:
        archive.writestr(
            "migration/2018-greatawakening/2018-greatawakening-in-before",
            json.dumps(record),
        )
    import_tbbt(
        raw,
        tmp_path / "out",
        max_records_per_archive=10,
        categories={"ban", "migration"},
    )
    panel = pd.read_csv(tmp_path / "out" / "tbbt_daily_panel.csv")
    ids = dict(zip(panel["intervention_type"], panel["intervention_id"]))
    assert ids == {"ban": "tbbt-07", "migration": "tbbt-21"}


def test_lemmy_restore_is_excluded() -> None:
    payload = {
        "removed_posts": [
            {
                "mod_remove_post": {
                    "id": 1,
                    "post_id": 9,
                    "removed": True,
                    "when_": "2026-01-01T00:00:00Z",
                    "reason": "rule",
                },
                "moderator": {"id": 3},
                "post": {
                    "id": 9,
                    "community_id": 4,
                    "published": "2025-12-31T00:00:00Z",
                    "ap_id": "https://example/post/9",
                },
                "community": {"id": 4},
            },
            {
                "mod_remove_post": {
                    "id": 2,
                    "post_id": 9,
                    "removed": False,
                    "when_": "2026-01-02T00:00:00Z",
                },
                "post": {"id": 9, "community_id": 4},
                "community": {"id": 4},
            },
        ]
    }
    included, excluded = parse_lemmy_modlog(payload, "lemmy.test")
    assert len(included) == 1
    assert len(excluded) == 1
    assert excluded.iloc[0]["excluded_reason"] == "restore_event"


def test_lemmy_cascade_join_namespaces_ids_and_excludes_broken_parents() -> None:
    posts = pd.DataFrame(
        [
            {
                "content_id": "1",
                "published": "2026-01-01T00:00:00Z",
                "community": "news",
                "title": "one",
                "url": "https://example.test/one",
            },
            {
                "content_id": "2",
                "published": "2026-01-01T00:00:00Z",
                "community": "news",
                "title": "two",
                "url": "https://example.test/two",
            },
        ]
    )
    comments = pd.DataFrame(
        [
            {
                "content_id": "1",
                "event_id": "10",
                "parent_event_id": "1",
                "created_at": "2026-01-01T00:01:00Z",
                "community": "news",
            },
            {
                "content_id": "2",
                "event_id": "20",
                "parent_event_id": "missing",
                "created_at": "2026-01-01T00:01:00Z",
                "community": "news",
            },
        ]
    )

    events, manifest = build_lemmy_cascade_events(posts, comments)

    assert set(events["content_id"]) == {"1"}
    assert set(events["event_id"]) == {"post:1", "comment:10"}
    assert events.loc[events["event_type"] == "comment", "parent_event_id"].item() == "post:1"
    assert manifest["n_complete_cascades"] == 1
    assert manifest["excluded_broken_parent_chain"] == 1


def test_url_canonicalization_and_match_types_are_separate(tmp_path) -> None:
    assert canonicalize_url("HTTP://www.Example.com/a/?utm_source=x&b=2") == "https://example.com/a?b=2"
    assert canonicalize_url("https://example.com:invalid/path") == ""
    events = pd.DataFrame(
        [
            {
                "platform": "reddit",
                "community": "news",
                "content_id": "r1",
                "event_id": "r1",
                "parent_event_id": "",
                "created_at": "2026-01-01T00:00:00Z",
                "event_type": "post",
                "title": "Central bank changes interest rates",
                "content_url": "https://example.com/story?utm_source=r",
            },
            {
                "platform": "hackernews",
                "community": "news",
                "content_id": "h1",
                "event_id": "h1",
                "parent_event_id": "",
                "created_at": "2026-01-01T01:00:00Z",
                "event_type": "story",
                "title": "Central bank changes interest rates",
                "content_url": "http://www.example.com/story",
            },
            {
                "platform": "lemmy",
                "community": "news",
                "content_id": "l1",
                "event_id": "l1",
                "parent_event_id": "",
                "created_at": "2026-01-01T02:00:00Z",
                "event_type": "post",
                "title": "Central bank changes interest rates today",
                "content_url": "https://different.test/item",
            },
        ]
    )
    matches, manifest = build_story_matches(events, tmp_path, semantic_threshold=0.5)
    assert set(matches["match_type"]) == {"exact_url", "semantic_event"}
    assert manifest["n_exact_url"] == 1
    assert manifest["n_semantic_event"] >= 1


def test_story_url_resolution_caches_redirect_and_html_canonical(tmp_path, monkeypatch) -> None:
    class FakeResponse:
        url = "https://publisher.test/redirected"
        headers = {"Content-Type": "text/html; charset=utf-8"}
        encoding = "utf-8"
        history = [object()]
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int):
            del chunk_size
            yield b'<html><head><link rel="canonical" href="/final-story"></head></html>'

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "bdmtf.revision.external_sources.requests.get",
        lambda *args, **kwargs: FakeResponse(),
    )
    roots = pd.DataFrame(
        {
            "source_url": ["http://publisher.test/tracking?utm_source=x"],
            "canonical_url": ["https://publisher.test/tracking"],
        }
    )
    cache_path = tmp_path / "url_resolution_cache.json"

    resolved, audit = resolve_story_urls(roots, cache_path, execute=True, workers=1)
    replayed, replay_audit = resolve_story_urls(roots, cache_path, execute=False)

    assert resolved.iloc[0]["canonical_url"] == "https://publisher.test/final-story"
    assert replayed.iloc[0]["canonical_url"] == "https://publisher.test/final-story"
    assert audit["status"] == "complete"
    assert audit["attempted"] == 1
    assert replay_audit["status"] == "cache_replay"


def test_prospective_split_has_nonoverlapping_window(tmp_path) -> None:
    events = pd.DataFrame(
        {
            "platform": ["reddit"] * 3,
            "content_id": ["a", "b", "c"],
            "event_id": ["a", "b", "c"],
            "created_at": [
                "2025-12-31T00:00:00Z",
                "2026-01-02T00:00:00Z",
                "2026-03-10T00:00:00Z",
            ],
        }
    )
    assigned, manifest = make_prospective_split(events, tmp_path, "2026-01-01T00:00:00Z", weeks=8)
    assert assigned["prospective_split"].tolist() == [
        "development",
        "prospective_test",
        "post_window",
    ]
    assert manifest["claim_allowed"] is True


def test_prospective_workflow_requires_predeclared_freeze(tmp_path) -> None:
    frame, manifest = make_prospective_split_workflow(
        tmp_path,
        {"prospective": {"freeze_at": None, "weeks": 8}},
    )
    assert frame.empty
    assert manifest["status"] == "not_frozen"
    assert manifest["claim_allowed"] is False
    assert (tmp_path / "artifacts/prospective/prospective_manifest.json").is_file()


def test_lemmy_confirmatory_matching_is_outcome_blind() -> None:
    interventions = pd.DataFrame(
        [
            {
                "intervention_id": "i1",
                "intervention_type": "lock_post",
                "community_id": "c1",
                "content_id": "treated",
                "occurred_at": "2026-01-10T00:00:00Z",
            }
        ]
    )
    posts = pd.DataFrame(
        [
            {
                "content_id": "treated",
                "community_id": "c1",
                "published": "2026-01-09T00:00:00Z",
                "score": 1000,
            },
            {
                "content_id": "age-match",
                "community_id": "c1",
                "published": "2026-01-08T23:00:00Z",
                "score": 0,
            },
            {
                "content_id": "score-match",
                "community_id": "c1",
                "published": "2026-01-01T00:00:00Z",
                "score": 1000,
            },
        ]
    )

    matches = _match_lemmy_risk_sets(
        interventions,
        posts,
        {
            "same_community": True,
            "without_replacement": True,
            "max_log_age_distance": 0.75,
        },
    )

    assert matches["control_content_id"].tolist() == ["age-match"]
    assert matches["distance_definition"].eq(
        "absolute_log1p_age_hours"
    ).all()


def test_lemmy_confirmatory_matching_does_not_reuse_controls() -> None:
    interventions = pd.DataFrame(
        [
            {
                "intervention_id": "i1",
                "intervention_type": "lock_post",
                "community_id": "c1",
                "content_id": "treated-1",
                "occurred_at": "2026-01-10T00:00:00Z",
            },
            {
                "intervention_id": "i2",
                "intervention_type": "remove_post",
                "community_id": "c1",
                "content_id": "treated-2",
                "occurred_at": "2026-01-11T00:00:00Z",
            },
        ]
    )
    posts = pd.DataFrame(
        [
            {
                "content_id": "treated-1",
                "community_id": "c1",
                "published": "2026-01-09T00:00:00Z",
            },
            {
                "content_id": "treated-2",
                "community_id": "c1",
                "published": "2026-01-10T00:00:00Z",
            },
            {
                "content_id": "control-1",
                "community_id": "c1",
                "published": "2026-01-09T01:00:00Z",
            },
            {
                "content_id": "control-2",
                "community_id": "c1",
                "published": "2026-01-09T02:00:00Z",
            },
        ]
    )

    matches = _match_lemmy_risk_sets(
        interventions,
        posts,
        {
            "same_community": True,
            "without_replacement": True,
        },
    )

    assert len(matches) == 2
    assert matches["control_content_id"].nunique() == 2


def test_lemmy_confirmatory_matching_uses_only_pre_event_trajectory() -> None:
    interventions = pd.DataFrame(
        [
            {
                "intervention_id": "i1",
                "intervention_type": "lock_post",
                "community_id": "c1",
                "content_id": "treated",
                "occurred_at": "2026-01-10T00:00:00Z",
            }
        ]
    )
    posts = pd.DataFrame(
        [
            {
                "content_id": content_id,
                "community_id": "c1",
                "published": "2026-01-01T00:00:00Z",
            }
            for content_id in ("treated", "bad-control", "good-control")
        ]
    )
    event_rows = [
        {
            "content_id": "treated",
            "created_at": "2026-01-09T12:00:00Z",
            "author_id": "a",
            "depth": 1,
        },
        {
            "content_id": "good-control",
            "created_at": "2026-01-09T10:00:00Z",
            "author_id": "b",
            "depth": 1,
        },
    ]
    event_rows.extend(
        {
            "content_id": "bad-control",
            "created_at": f"2026-01-09T{hour:02d}:00:00Z",
            "author_id": f"bad-{hour}",
            "depth": hour + 1,
        }
        for hour in range(5)
    )

    matches = _match_lemmy_risk_sets(
        interventions,
        posts,
        {
            "same_community": True,
            "without_replacement": True,
            "candidate_pool_size": 2,
            "pre_event_matching": True,
            "age_distance_weight": 0.25,
            "pretrajectory_distance_weight": 1.0,
        },
        pd.DataFrame(event_rows),
        pre_days=7,
    )

    assert matches["control_content_id"].tolist() == ["good-control"]
    assert matches["pretrajectory_distance"].iloc[0] == pytest.approx(0.0)
