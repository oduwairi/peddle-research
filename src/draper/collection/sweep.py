"""Continuous collection engine for AdFlex filter-code strategy.

Core classes:
- Query: one search config (platform + filters + ordering + geo), paginates dynamically
- FilterConfig: loads extracted filter codes from configs/filters/
- SweepPlanner: reads sweep_plans.yaml, generates Query objects
- SweepExecutor: runs queries with pagination, dedup, cursor checkpointing
- SweepStats: tracks calls, credits, ads per platform/sweep
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from draper.scraping.adflex import AdFlexClient
from draper.scraping.schemas import RawAd
from draper.utils.io import append_jsonl, read_jsonl

logger = logging.getLogger("draper")


@dataclass
class Query:
    """A single search config: platform + filters + ordering + geo.

    Pagination is handled dynamically by the executor, not baked in.
    """

    platform: str
    sweep_name: str
    filters: dict[str, list[int]]
    ranges: dict[str, list[int]]
    ordering: str
    geo: int | None = None
    search_field: list[dict[str, str]] | None = None

    @property
    def key(self) -> str:
        """Unique string key for checkpoint tracking."""
        parts = [self.platform, self.sweep_name, self.ordering]
        if self.geo is not None:
            parts.append(f"g{self.geo}")
        for k, v in sorted(self.filters.items()):
            parts.append(f"{k}={v}")
        for k, v in sorted(self.ranges.items()):
            parts.append(f"{k}={v[0]}-{v[1]}")
        if self.search_field:
            for sf in self.search_field:
                parts.append(f"sf:{sf.get('type', '')}={sf.get('text', '')}")
        return "|".join(parts)


@dataclass
class QueryProgress:
    """Pagination state for a single query, persisted in checkpoint."""

    pages_fetched: int = 0
    last_hit: int | None = None
    done: bool = False  # True if API returned has_next_page=False

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages_fetched": self.pages_fetched,
            "last_hit": self.last_hit,
            "done": self.done,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueryProgress:
        return cls(
            pages_fetched=data.get("pages_fetched", 0),
            last_hit=data.get("last_hit"),
            done=data.get("done", False),
        )


@dataclass
class SweepStats:
    """Accumulated stats for assessment."""

    total_calls: int = 0
    total_credits: int = 0
    total_ads_raw: int = 0
    total_ads_unique: int = 0
    by_platform: dict[str, dict[str, int]] = field(default_factory=dict)
    by_sweep: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, query: Query, raw: int, unique: int, credits: int = 100) -> None:
        """Record stats from a single API call."""
        self.total_calls += 1
        self.total_credits += credits
        self.total_ads_raw += raw
        self.total_ads_unique += unique

        p = query.platform
        if p not in self.by_platform:
            self.by_platform[p] = {"calls": 0, "unique": 0, "raw": 0}
        self.by_platform[p]["calls"] += 1
        self.by_platform[p]["unique"] += unique
        self.by_platform[p]["raw"] += raw

        sweep_key = f"{p}:{query.sweep_name}"
        if sweep_key not in self.by_sweep:
            self.by_sweep[sweep_key] = {"calls": 0, "unique": 0, "raw": 0}
        self.by_sweep[sweep_key]["calls"] += 1
        self.by_sweep[sweep_key]["unique"] += unique
        self.by_sweep[sweep_key]["raw"] += raw

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_credits": self.total_credits,
            "total_ads_raw": self.total_ads_raw,
            "total_ads_unique": self.total_ads_unique,
            "by_platform": self.by_platform,
            "by_sweep": self.by_sweep,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SweepStats:
        return cls(
            total_calls=data.get("total_calls", 0),
            total_credits=data.get("total_credits", 0),
            total_ads_raw=data.get("total_ads_raw", 0),
            total_ads_unique=data.get("total_ads_unique", 0),
            by_platform=data.get("by_platform", {}),
            by_sweep=data.get("by_sweep", {}),
        )


class FilterConfig:
    """Loads extracted filter codes from configs/filters/ YAML files."""

    def __init__(self, filters_dir: str | Path = "configs/filters") -> None:
        self._dir = Path(filters_dir)
        self._platforms: dict[str, dict[str, Any]] = {}
        for yaml_file in self._dir.glob("*.yaml"):
            if yaml_file.name.startswith("_"):
                continue
            with yaml_file.open() as f:
                data = yaml.safe_load(f)
            if data and "platform" in data:
                self._platforms[data["platform"]] = data

    def get_codes(
        self,
        platform: str,
        filter_type: str,
        top_n: int | None = None,
        sample_n: int | None = None,
    ) -> list[int]:
        """Get filter codes for a platform/filter_type.

        Args:
            top_n: Take the first N codes (most popular).
            sample_n: Randomly sample N codes from the full list.
                      If both are set, sample_n takes precedence.
        """
        pdata = self._platforms.get(platform, {})
        select = pdata.get("select_filters", {}).get(filter_type, {})
        codes = list(select.keys())
        if sample_n is not None:
            import random

            return random.sample(codes, min(sample_n, len(codes)))
        if top_n is not None:
            codes = codes[:top_n]
        return codes

    def get_label(self, platform: str, filter_type: str, code: int) -> str:
        """Look up human label for a filter code."""
        return str(
            self._platforms.get(platform, {})
            .get("select_filters", {})
            .get(filter_type, {})
            .get(code, "")
        )

    @property
    def platforms(self) -> list[str]:
        return list(self._platforms.keys())


class SweepPlanner:
    """Generates Query objects from sweep plan config."""

    def __init__(
        self,
        plan_path: str | Path = "configs/sweep_plans.yaml",
        filter_config: FilterConfig | None = None,
    ) -> None:
        with open(plan_path) as f:
            self._plan: dict[str, Any] = yaml.safe_load(f)
        self._filters = filter_config or FilterConfig()
        self._geo_groups: dict[str, list[int]] = self._plan.get("geo_groups", {})

    def _resolve_geos(self, geos_value: Any) -> list[int | None]:
        """Resolve geo value: string → geo group lookup, list → as-is, None → [None]."""
        if geos_value is None or geos_value == []:
            return [None]
        if isinstance(geos_value, str):
            group = self._geo_groups.get(geos_value)
            if group is not None:
                return list(group)
            return [None]
        if isinstance(geos_value, list):
            return geos_value
        return [None]

    def generate_queries(self, platform: str, sweep_name: str) -> list[Query]:
        """Generate Query objects for a platform sweep type.

        One Query per filter/ordering/geo combo. No page expansion —
        pagination is handled dynamically by the executor.
        """
        platform_config = self._plan["platforms"].get(platform, {})
        sweeps = platform_config.get("sweeps", [])
        sweep = next((s for s in sweeps if s["name"] == sweep_name), None)
        if sweep is None:
            return []

        if sweep.get("dynamic"):
            return []

        orderings = sweep.get("orderings", ["popularity"])
        geos = self._resolve_geos(sweep.get("geos"))

        queries: list[Query] = []

        filter_type = sweep.get("filter_type")
        filter_types = sweep.get("filter_types")
        range_filter = sweep.get("range_filter")
        keywords: list[str] | None = sweep.get("keywords")
        domains: list[str] | None = sweep.get("domains")

        if keywords:
            sf_type = "text"
            for term in keywords:
                for ordering in orderings:
                    for geo in geos:
                        queries.append(
                            Query(
                                platform=platform,
                                sweep_name=sweep_name,
                                filters={},
                                ranges={},
                                ordering=ordering,
                                geo=geo,
                                search_field=[{"type": sf_type, "text": term}],
                            )
                        )

        elif domains:
            sf_type = "url_chain"
            for domain in domains:
                for ordering in orderings:
                    for geo in geos:
                        queries.append(
                            Query(
                                platform=platform,
                                sweep_name=sweep_name,
                                filters={},
                                ranges={},
                                ordering=ordering,
                                geo=geo,
                                search_field=[{"type": sf_type, "text": domain}],
                            )
                        )

        elif filter_type:
            top_n = sweep.get("top_n")
            sample_n = sweep.get("sample_n")
            codes = self._filters.get_codes(platform, filter_type, top_n, sample_n)
            if not codes:
                logger.warning(
                    f"{platform}:{sweep_name} filter_type={filter_type} "
                    f"returned 0 codes. Check configs/filters/{platform}.yaml"
                )
            for code in codes:
                for ordering in orderings:
                    for geo in geos:
                        queries.append(
                            Query(
                                platform=platform,
                                sweep_name=sweep_name,
                                filters={filter_type: [code]},
                                ranges={},
                                ordering=ordering,
                                geo=geo,
                            )
                        )

        elif filter_types:
            top_n = sweep.get("top_n")
            sample_n = sweep.get("sample_n")
            for ft in filter_types:
                codes = self._filters.get_codes(platform, ft, top_n, sample_n)
                for code in codes:
                    for ordering in orderings:
                        for geo in geos:
                            queries.append(
                                Query(
                                    platform=platform,
                                    sweep_name=sweep_name,
                                    filters={ft: [code]},
                                    ranges={},
                                    ordering=ordering,
                                    geo=geo,
                                )
                            )

        elif range_filter:
            tiers = sweep.get("tiers", [])
            for tier in tiers:
                for ordering in orderings:
                    for geo in geos:
                        queries.append(
                            Query(
                                platform=platform,
                                sweep_name=sweep_name,
                                filters={},
                                ranges={range_filter: tier},
                                ordering=ordering,
                                geo=geo,
                            )
                        )

        else:
            for ordering in orderings:
                for geo in geos:
                    queries.append(
                        Query(
                            platform=platform,
                            sweep_name=sweep_name,
                            filters={},
                            ranges={},
                            ordering=ordering,
                            geo=geo,
                        )
                    )

        return queries

    def get_platform_names(self) -> list[str]:
        return list(self._plan.get("platforms", {}).keys())

    def get_platform_budget_pct(self, platform: str) -> float:
        """Get the budget percentage for a specific platform."""
        return float(self._plan.get("platforms", {}).get(platform, {}).get("budget_pct", 0.0))

    def get_sweep_names(self, platform: str) -> list[str]:
        sweeps = self._plan.get("platforms", {}).get(platform, {}).get("sweeps", [])
        return [s["name"] for s in sweeps]

    def get_total_budget(self) -> int:
        """Total API calls available."""
        budget = self._plan.get("budget", {})
        total: int = int(budget.get("total_credits", 500000))
        per_call: int = int(budget.get("credits_per_call", 100))
        return total // per_call

    @property
    def credits_per_call(self) -> int:
        return int(self._plan.get("budget", {}).get("credits_per_call", 100))


class CollectionCheckpoint:
    """Tracks pagination state per query and overall progress.

    Saved as JSON: {query_key: {pages_fetched, last_hit, done}, ...}
    """

    def __init__(self, path: Path) -> None:
        self._path = path / "collection_state.json"
        self._state: dict[str, QueryProgress] = {}
        self._stats: dict[str, Any] = {}
        if self._path.exists():
            with self._path.open() as f:
                data = json.load(f)
            for key, prog in data.get("queries", {}).items():
                self._state[key] = QueryProgress.from_dict(prog)
            self._stats = data.get("stats", {})

    def get_progress(self, query_key: str) -> QueryProgress:
        """Get pagination state for a query. Returns fresh state if not seen."""
        return self._state.get(query_key, QueryProgress())

    def update_progress(self, query_key: str, progress: QueryProgress) -> None:
        """Update and persist pagination state for a query."""
        self._state[query_key] = progress
        self._save()

    def update_stats(self, stats: SweepStats) -> None:
        """Persist cumulative stats."""
        self._stats = stats.to_dict()
        self._save()

    def get_stats(self) -> SweepStats | None:
        if self._stats:
            return SweepStats.from_dict(self._stats)
        return None

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w") as f:
            json.dump(
                {
                    "queries": {k: v.to_dict() for k, v in self._state.items()},
                    "stats": self._stats,
                },
                f,
                indent=2,
            )

    @property
    def query_states(self) -> dict[str, QueryProgress]:
        return dict(self._state)

    def total_calls_made(self) -> int:
        return sum(p.pages_fetched for p in self._state.values())

    def rebuild_stats(self) -> SweepStats:
        """Compute stats from adflex_ads.jsonl + checkpoint query progress.

        Ad counts come from JSONL (source of truth for data).
        Call counts come from checkpoint pages_fetched (source of truth for API usage).
        """
        stats = SweepStats()
        jsonl_path = self._path.parent / "adflex_ads.jsonl"
        if not jsonl_path.exists():
            return stats

        records = read_jsonl(jsonl_path)
        stats.total_ads_unique = len(records)
        # Raw count unknown from JSONL alone; set equal to unique as lower bound
        stats.total_ads_raw = len(records)

        # Derive call counts from checkpoint (pages_fetched = actual API calls)
        stats.total_calls = self.total_calls_made()
        stats.total_credits = stats.total_calls * 100

        for rec in records:
            platform = rec.get("platform", "unknown")
            vertical = rec.get("vertical", "")

            if platform not in stats.by_platform:
                stats.by_platform[platform] = {"calls": 0, "unique": 0, "raw": 0}
            stats.by_platform[platform]["unique"] += 1
            stats.by_platform[platform]["raw"] += 1

            if vertical:
                if vertical not in stats.by_sweep:
                    stats.by_sweep[vertical] = {"calls": 0, "unique": 0, "raw": 0}
                stats.by_sweep[vertical]["unique"] += 1
                stats.by_sweep[vertical]["raw"] += 1

        # Derive per-platform and per-sweep call counts from checkpoint
        for query_key, progress in self._state.items():
            parts = query_key.split("|")
            if len(parts) >= 2:
                platform = parts[0]
                sweep_name = parts[1]
                sweep_key = f"{platform}:{sweep_name}"

                if platform not in stats.by_platform:
                    stats.by_platform[platform] = {"calls": 0, "unique": 0, "raw": 0}
                stats.by_platform[platform]["calls"] += progress.pages_fetched

                if sweep_key not in stats.by_sweep:
                    stats.by_sweep[sweep_key] = {"calls": 0, "unique": 0, "raw": 0}
                stats.by_sweep[sweep_key]["calls"] += progress.pages_fetched

        self._stats = stats.to_dict()
        self._save()
        return stats

    def reset(self) -> None:
        """Reset all query progress (clear cursors). Stats are preserved."""
        self._state.clear()
        self._save()


class SweepExecutor:
    """Executes queries with cursor-based pagination, dedup, and checkpointing."""

    # Map API body keys (used in sweep configs) → search_ads() param names
    _API_TO_PARAM: dict[str, str] = {
        "type_call_to_actions": "call_to_actions",
        "format": "ad_format",
        "os": "ad_os",
    }

    def __init__(
        self,
        client: AdFlexClient,
        output_dir: Path,
        checkpoint: CollectionCheckpoint,
        seen_ids: set[str] | None = None,
    ) -> None:
        self._client = client
        self._output_dir = output_dir
        self._checkpoint = checkpoint
        self._seen_ids = seen_ids or set()
        self._stats = checkpoint.get_stats() or SweepStats()
        self._output_path = output_dir / "adflex_ads.jsonl"

    @property
    def stats(self) -> SweepStats:
        return self._stats

    @property
    def seen_ids(self) -> set[str]:
        return self._seen_ids

    async def execute_query(self, query: Query, max_pages: int = 10) -> int:
        """Paginate a query from where it left off. Returns number of API calls made.

        Resumes from saved cursor. Paginates until:
        - max_pages reached (across all sessions for this query)
        - API says no more pages
        - Caller stops us (budget check is external)
        """
        progress = self._checkpoint.get_progress(query.key)

        if progress.done:
            return 0

        # Safety: cursor is required for pagination beyond page 1.
        # If missing, reset to page 1 to avoid silent data corruption.
        if progress.pages_fetched > 0 and progress.last_hit is None:
            logger.warning(
                f"Query {query.key}: pages_fetched={progress.pages_fetched} "
                f"but no cursor. Resetting to page 1."
            )
            progress.pages_fetched = 0

        calls_made = 0
        page = progress.pages_fetched + 1
        last_hit = progress.last_hit
        stop_at_page = progress.pages_fetched + max_pages

        while page <= stop_at_page and not progress.done:
            kwargs: dict[str, Any] = {
                "platform": query.platform,
                "orderby": query.ordering,
                "page": page,
            }
            if query.geo is not None:
                kwargs["countries"] = [query.geo]
            if last_hit is not None:
                kwargs["last_hit"] = last_hit

            for filter_key, codes in query.filters.items():
                param_name = self._API_TO_PARAM.get(filter_key, filter_key)
                kwargs[param_name] = codes

            if query.ranges:
                kwargs["ranges"] = query.ranges

            if query.search_field:
                kwargs["search_field"] = query.search_field

            response = await self._client.search_ads(**kwargs)
            resp_data = response.get("data", {})
            ads_data = resp_data.get("ads", [])
            has_next = resp_data.get("has_next_page", False)
            last_hit = resp_data.get("last_hit")
            calls_made += 1

            # Parse, dedup, persist
            new_ads: list[RawAd] = []
            for ad_data in ads_data:
                try:
                    raw_ad = AdFlexClient._parse_ad(ad_data, query.platform)
                    if raw_ad.ad_id not in self._seen_ids:
                        self._seen_ids.add(raw_ad.ad_id)
                        raw_ad.vertical = f"{query.platform}:{query.sweep_name}"
                        new_ads.append(raw_ad)
                except Exception as e:
                    logger.warning(f"Parse error: {e}")

            if new_ads:
                append_jsonl(new_ads, self._output_path)

            self._stats.record(query, raw=len(ads_data), unique=len(new_ads))

            # Update cursor checkpoint
            progress.pages_fetched = page
            progress.last_hit = last_hit
            progress.done = not has_next or last_hit is None or not ads_data
            self._checkpoint.update_progress(query.key, progress)

            if progress.done:
                break

            page += 1

        return calls_made


def load_seen_ids(raw_dir: Path) -> set[str]:
    """Load all previously collected ad IDs for deduplication."""
    seen: set[str] = set()
    for jsonl_file in raw_dir.glob("adflex_*.jsonl"):
        try:
            records = read_jsonl(jsonl_file)
            for rec in records:
                ad_id = rec.get("ad_id", "")
                if ad_id:
                    seen.add(ad_id)
        except Exception as e:
            logger.warning(f"Error reading {jsonl_file}: {e}")
    return seen


def select_queries_for_budget(
    all_queries: list[Query],
    call_budget: int,
    planner: SweepPlanner,
    checkpoint: CollectionCheckpoint,
    max_pages: int = 10,
) -> list[tuple[Query, int]]:
    """Select queries distributed by platform AND sweep type.

    Budget flows: total → platforms (by budget_pct) → sweeps (equal share)
    → queries (round-robin within each sweep).
    Each query gets limited pages so every sweep gets representation.

    Returns list of (query, pages_this_run) tuples.
    """
    # Filter out done queries
    active = [q for q in all_queries if not checkpoint.get_progress(q.key).done]

    # Group by platform → sweep
    by_platform_sweep: dict[str, dict[str, list[Query]]] = {}
    for q in active:
        by_platform_sweep.setdefault(q.platform, {}).setdefault(q.sweep_name, []).append(q)

    selected: list[tuple[Query, int]] = []

    for platform in planner.get_platform_names():
        pct = planner.get_platform_budget_pct(platform)
        platform_budget = int(call_budget * pct)
        sweeps = by_platform_sweep.get(platform, {})

        if not sweeps:
            continue

        # Divide platform budget equally across its sweep types
        sweep_budget = max(1, platform_budget // len(sweeps))

        for _sweep_name, queries in sweeps.items():
            query_count = len(queries)
            if query_count == 0:
                continue

            # Prioritize untouched queries, then in-progress ones
            untouched = [q for q in queries if checkpoint.get_progress(q.key).pages_fetched == 0]
            in_progress = [q for q in queries if checkpoint.get_progress(q.key).pages_fetched > 0]
            prioritized = untouched + in_progress

            # Each query gets an equal slice of the sweep budget
            pages_per_query = max(1, sweep_budget // query_count)
            pages_per_query = min(pages_per_query, max_pages)

            added = 0
            for q in prioritized:
                if added >= sweep_budget:
                    break
                progress = checkpoint.get_progress(q.key)
                remaining = min(pages_per_query, max_pages - progress.pages_fetched)
                if remaining > 0:
                    selected.append((q, remaining))
                    added += remaining

    return selected


def save_run_stats(raw_dir: Path, stats: SweepStats) -> Path:
    """Persist run stats as JSON."""
    stats_path = raw_dir / "run_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats.to_dict(), f, indent=2)
    return stats_path


# --- Legacy helpers kept for enrichment ---


def load_enriched_ids(raw_dir: Path) -> set[str]:
    """Load IDs of ads that have already been enriched via detail calls."""
    path = raw_dir / "enriched_ids.json"
    if not path.exists():
        return set()
    with path.open() as f:
        return set(json.load(f))


def save_enriched_ids(raw_dir: Path, ids: set[str]) -> None:
    """Persist the set of enriched ad IDs."""
    path = raw_dir / "enriched_ids.json"
    with path.open("w") as f:
        json.dump(sorted(ids), f)


def save_enrich_stats(raw_dir: Path, stats: dict[str, Any]) -> Path:
    """Persist enrichment run stats as JSON."""
    stats_path = raw_dir / "enrich_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    return stats_path
