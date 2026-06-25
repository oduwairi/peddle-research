"""Tests for the continuous collection engine."""

import tempfile
from pathlib import Path

from draper.collection.sweep import (
    CollectionCheckpoint,
    FilterConfig,
    Query,
    QueryProgress,
    SweepPlanner,
    SweepStats,
    select_queries_for_budget,
)

# --- FilterConfig ---


def test_filter_config_loads() -> None:
    config = FilterConfig("configs/filters")
    assert "facebook" in config.platforms
    assert "tiktok" in config.platforms
    assert "x" in config.platforms
    assert "pinterest" in config.platforms
    assert "reddit" in config.platforms


def test_filter_config_get_codes() -> None:
    config = FilterConfig("configs/filters")
    codes = config.get_codes("tiktok", "interests")
    assert len(codes) == 63  # TikTok has 63 interest codes


def test_filter_config_get_codes_top_n() -> None:
    config = FilterConfig("configs/filters")
    codes = config.get_codes("x", "owner_category", top_n=30)
    assert len(codes) == 30


def test_filter_config_get_label() -> None:
    config = FilterConfig("configs/filters")
    codes = config.get_codes("tiktok", "interests")
    if codes:
        label = config.get_label("tiktok", "interests", codes[0])
        assert isinstance(label, str)
        assert len(label) > 0


def test_filter_config_missing_platform() -> None:
    config = FilterConfig("configs/filters")
    codes = config.get_codes("nonexistent", "interests")
    assert codes == []


# --- SweepPlanner ---


def test_planner_platforms() -> None:
    planner = SweepPlanner("configs/sweep_plans.yaml")
    platforms = planner.get_platform_names()
    assert "facebook" in platforms
    assert "tiktok" in platforms
    assert len(platforms) == 5


def test_planner_sweep_names() -> None:
    planner = SweepPlanner("configs/sweep_plans.yaml")
    sweeps = planner.get_sweep_names("facebook")
    assert "keywords_intent" in sweeps
    assert "domains" in sweeps


def test_planner_total_budget() -> None:
    planner = SweepPlanner("configs/sweep_plans.yaml")
    # 500000 / 100 = 5000
    assert planner.get_total_budget() == 5000


def test_planner_keyword_sweep_generates_queries() -> None:
    planner = SweepPlanner("configs/sweep_plans.yaml")
    queries = planner.generate_queries("facebook", "keywords_intent")
    # 10 keywords x 2 orderings x 1 geo (EN-only) = 20 queries
    assert len(queries) == 20
    assert all(q.platform == "facebook" for q in queries)
    assert all(q.sweep_name == "keywords_intent" for q in queries)


def test_planner_vertical_sweep_generates_queries() -> None:
    planner = SweepPlanner("configs/sweep_plans.yaml")
    queries = planner.generate_queries("tiktok", "keywords_vertical")
    # 10 keywords x 2 orderings x 1 geo = 20
    assert len(queries) == 20


def test_planner_domain_sweep_generates_queries() -> None:
    planner = SweepPlanner("configs/sweep_plans.yaml")
    queries = planner.generate_queries("facebook", "domains")
    # 15 domains x 2 orderings x 1 geo = 30
    assert len(queries) == 30


# --- Query ---


def test_query_key_uniqueness() -> None:
    q1 = Query("facebook", "broad", {}, {}, "popularity", geo=1000001)
    q2 = Query("facebook", "broad", {}, {}, "days_active", geo=1000001)
    q3 = Query("tiktok", "broad", {}, {}, "popularity", geo=1000001)

    keys = {q1.key, q2.key, q3.key}
    assert len(keys) == 3  # all unique


def test_query_key_with_filters() -> None:
    q1 = Query("facebook", "ecommerce_tech", {"ecommerce": [11100]}, {}, "popularity")
    q2 = Query("facebook", "ecommerce_tech", {"ecommerce": [11190]}, {}, "popularity")
    assert q1.key != q2.key


def test_query_key_with_ranges() -> None:
    q1 = Query("facebook", "perf", {}, {"reaction": [0, 100]}, "popularity")
    q2 = Query("facebook", "perf", {}, {"reaction": [100, 1000]}, "popularity")
    assert q1.key != q2.key


# --- QueryProgress ---


def test_query_progress_defaults() -> None:
    p = QueryProgress()
    assert p.pages_fetched == 0
    assert p.last_hit is None
    assert p.done is False


def test_query_progress_serialization() -> None:
    p = QueryProgress(pages_fetched=3, last_hit=12345, done=False)
    data = p.to_dict()
    restored = QueryProgress.from_dict(data)
    assert restored.pages_fetched == 3
    assert restored.last_hit == 12345
    assert restored.done is False


# --- CollectionCheckpoint ---


def test_checkpoint_fresh_state() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cp = CollectionCheckpoint(Path(tmpdir))
        progress = cp.get_progress("facebook|broad|popularity")
        assert progress.pages_fetched == 0
        assert progress.done is False


def test_checkpoint_persist_and_reload() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)

        # Save some state
        cp = CollectionCheckpoint(path)
        cp.update_progress("fb|broad", QueryProgress(pages_fetched=3, last_hit=999, done=False))
        cp.update_progress("tt|broad", QueryProgress(pages_fetched=5, last_hit=None, done=True))

        # Reload from disk
        cp2 = CollectionCheckpoint(path)
        fb = cp2.get_progress("fb|broad")
        assert fb.pages_fetched == 3
        assert fb.last_hit == 999
        assert fb.done is False

        tt = cp2.get_progress("tt|broad")
        assert tt.pages_fetched == 5
        assert tt.done is True


def test_checkpoint_total_calls() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cp = CollectionCheckpoint(Path(tmpdir))
        cp.update_progress("q1", QueryProgress(pages_fetched=3))
        cp.update_progress("q2", QueryProgress(pages_fetched=7))
        assert cp.total_calls_made() == 10


# --- SweepStats ---


def test_sweep_stats_record() -> None:
    stats = SweepStats()
    q = Query("facebook", "broad", {}, {}, "popularity")
    stats.record(q, raw=18, unique=15)

    assert stats.total_calls == 1
    assert stats.total_credits == 100
    assert stats.total_ads_raw == 18
    assert stats.total_ads_unique == 15
    assert stats.by_platform["facebook"]["calls"] == 1
    assert stats.by_sweep["facebook:broad"]["unique"] == 15


def test_sweep_stats_serialization() -> None:
    stats = SweepStats()
    q = Query("tiktok", "interests", {}, {}, "popularity")
    stats.record(q, raw=18, unique=12)

    data = stats.to_dict()
    restored = SweepStats.from_dict(data)
    assert restored.total_calls == stats.total_calls
    assert restored.total_ads_unique == stats.total_ads_unique
    assert restored.by_platform == stats.by_platform


# --- select_queries_for_budget ---


def test_select_skips_done_queries() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cp = CollectionCheckpoint(Path(tmpdir))
        cp.update_progress("facebook|broad|popularity", QueryProgress(done=True))

        planner = SweepPlanner("configs/sweep_plans.yaml")
        all_queries = []
        for platform in planner.get_platform_names():
            for sweep_name in planner.get_sweep_names(platform):
                all_queries.extend(planner.generate_queries(platform, sweep_name))

        selected = select_queries_for_budget(all_queries, 100, planner, cp, max_pages=10)

        # The done query should not be selected
        selected_keys = {q.key for q, _ in selected}
        assert "facebook|broad|popularity" not in selected_keys
