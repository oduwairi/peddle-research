"""AdFlex API audit — test unexplored endpoints, reuse existing data.

Loads existing data from data/raw/adflex_exploration_raw.json and only
makes NEW API calls for endpoints not previously tested:
  - Filters for 6 new platforms (native, display, pinterest, reddit, x, youtube)
  - Search for 3 new platforms (native, display, youtube)
  - Ad detail endpoint (all platforms with search results)
  - Auxiliary filter-item endpoints (interests, owner_categories, publishers)
  - OpenAPI spec fetch (no API key, no credits)

Run with: python scripts/adflex_explore.py
Output:   data/raw/adflex_api_audit.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.adflex.io/api/v1"
API_KEY = os.environ.get("ADFLEX_API_KEY", "")
assert API_KEY, "Set ADFLEX_API_KEY in .env"

client = httpx.Client(timeout=30.0)
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

ALL_PLATFORMS = [
    "facebook",
    "meta",
    "native",
    "display",
    "pinterest",
    "reddit",
    "tiktok",
    "x",
    "youtube",
]

# Platforms already covered in adflex_exploration_raw.json
EXISTING_FILTER_PLATFORMS = {"facebook", "meta", "tiktok"}
EXISTING_SEARCH_PLATFORMS = {"facebook", "tiktok", "x", "pinterest", "reddit", "meta"}

AUXILIARY_ENDPOINTS = [
    "/facebook/filter/interests/items",
    "/facebook/filter/owner_categories/items",
    "/native/filter/publishers/items",
    "/display/filter/publishers/items",
    "/youtube/filter/publishers/items",
]


def safe_call(label: str, method: str, url: str, **kwargs) -> dict:
    """Make an API call, returning status/body/error without raising."""
    start = time.time()
    try:
        if method == "GET":
            r = client.get(url, params={"api_key": API_KEY, **(kwargs.get("params", {}))})
        else:
            r = client.post(url, params={"api_key": API_KEY}, json=kwargs.get("json", {}))
        elapsed = (time.time() - start) * 1000
        body = r.json() if r.status_code == 200 else None
        result = {"status": r.status_code, "body": body, "elapsed_ms": round(elapsed, 1)}
        if r.status_code != 200:
            result["error"] = r.text[:500]
        return result
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return {"status": "error", "body": None, "elapsed_ms": round(elapsed, 1), "error": str(e)}


# ── Load existing data ───────────────────────────────────────────
existing_path = output_dir / "adflex_exploration_raw.json"
if existing_path.exists():
    with open(existing_path) as f:
        existing = json.load(f)
    print(f"Loaded existing data from {existing_path}")
else:
    existing = {}
    print("No existing data found — all calls will be fresh")

audit = {}


# ── 1. OpenAPI spec (free, no API key) ───────────────────────────
print("\n=== Fetching OpenAPI spec ===")
try:
    spec_resp = client.get("https://doc.adflex.io/_bundle/openapi.yaml", timeout=15.0)
    spec_resp.raise_for_status()
    spec = yaml.safe_load(spec_resp.text)
    openapi_paths = {}
    for path, methods in spec.get("paths", {}).items():
        openapi_paths[path] = list(methods.keys())
    audit["openapi_paths"] = openapi_paths
    audit["openapi_info"] = spec.get("info", {})
    print(f"  Found {len(openapi_paths)} endpoints")
    for path, methods in sorted(openapi_paths.items()):
        print(f"    {', '.join(m.upper() for m in methods):6s} {path}")
except Exception as e:
    audit["openapi_paths"] = {"error": str(e)}
    print(f"  ERROR: {e}")


# ── 2. Filters — only fetch platforms we don't have ──────────────
print("\n=== Filters (skipping already-fetched platforms) ===")
all_filters = {}

# Carry over existing filter data
for p in EXISTING_FILTER_PLATFORMS:
    existing_filter = existing.get("filters", {}).get(p)
    if existing_filter and "error" not in existing_filter:
        all_filters[p] = {"status": 200, "body": existing_filter, "source": "cached"}
        fcount = len(existing_filter.get("data", {}).get("filters", []))
        print(f"  {p}: cached ({fcount} filters)")

# Fetch new platforms
new_filter_platforms = [p for p in ALL_PLATFORMS if p not in EXISTING_FILTER_PLATFORMS]
for platform in new_filter_platforms:
    print(f"  {platform}...", end=" ")
    result = safe_call(
        f"filters-{platform}",
        "GET",
        f"{BASE_URL}/filters/{platform}/search",
    )
    all_filters[platform] = result
    if result["status"] == 200 and result["body"]:
        flist = result["body"].get("data", {}).get("filters", [])
        print(f"OK — {len(flist)} filters")
    else:
        print(f"status={result['status']}")
    time.sleep(0.3)
audit["filters"] = all_filters


# ── 3. Auxiliary filter-item endpoints (likely free) ─────────────
print("\n=== Auxiliary filter-item endpoints ===")
auxiliary = {}
for endpoint in AUXILIARY_ENDPOINTS:
    print(f"  {endpoint}...", end=" ")
    result = safe_call(f"aux-{endpoint}", "GET", f"{BASE_URL}{endpoint}")
    auxiliary[endpoint] = result
    if result["status"] == 200 and result["body"]:
        data = result["body"].get("data", result["body"])
        if isinstance(data, list):
            print(f"OK — {len(data)} items")
        elif isinstance(data, dict) and "items" in data:
            print(f"OK — {len(data['items'])} items")
        else:
            keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
            print(f"OK — keys: {keys}")
    else:
        print(f"status={result['status']}")
    time.sleep(0.3)
audit["auxiliary_filters"] = auxiliary


# ── 4. Search — only fetch platforms we don't have ───────────────
print("\n=== Search (skipping already-fetched platforms) ===")
all_search = {}

# Carry over existing search data
for p in EXISTING_SEARCH_PLATFORMS:
    existing_search = existing.get("search", {}).get(p)
    if existing_search and "error" not in existing_search:
        all_search[p] = {"status": 200, "body": existing_search, "source": "cached"}
        ad_count = len(existing_search.get("data", {}).get("ads", []))
        print(f"  {p}: cached ({ad_count} ads)")

# Fetch new platforms
new_search_platforms = [p for p in ALL_PLATFORMS if p not in EXISTING_SEARCH_PLATFORMS]
for platform in new_search_platforms:
    print(f"  {platform}...", end=" ")
    body = {
        "page": 1,
        "orderby": "popularity",
        "search_field": [{"type": "text", "text": "shop"}],
    }
    result = safe_call(
        f"search-{platform}",
        "POST",
        f"{BASE_URL}/{platform}/ads/search",
        json=body,
    )
    all_search[platform] = result
    if result["status"] == 200 and result["body"]:
        ads = result["body"].get("data", {}).get("ads", [])
        has_next = result["body"].get("data", {}).get("has_next_page", False)
        print(f"OK — {len(ads)} ads, has_next={has_next}")
    else:
        print(f"status={result['status']}")
    time.sleep(0.5)
audit["search"] = all_search


# ── 5. Ad detail endpoint (never tested before) ─────────────────
print("\n=== Ad detail endpoint ===")
detail_results = {}
search_vs_detail = {}

for platform in ALL_PLATFORMS:
    search_data = all_search.get(platform, {})
    body = search_data.get("body")
    if not body or search_data.get("status") != 200:
        continue
    ads = body.get("data", {}).get("ads", [])
    if not ads:
        print(f"  {platform}: no ads from search, skipping detail")
        continue

    first_ad = ads[0]
    ad_id = first_ad.get("id")
    if not ad_id:
        continue

    print(f"  {platform} (ad_id={ad_id})...", end=" ")
    result = safe_call(
        f"detail-{platform}",
        "GET",
        f"{BASE_URL}/{platform}/ads/{ad_id}",
    )
    detail_results[platform] = result

    if result["status"] == 200 and result["body"]:
        # Compare search ad vs detail ad
        detail_ad = result["body"].get("data", result["body"])
        if isinstance(detail_ad, list) and detail_ad:
            detail_ad = detail_ad[0]

        search_keys = set(first_ad.keys())
        detail_keys = set(detail_ad.keys()) if isinstance(detail_ad, dict) else set()

        diff = {
            "search_only": sorted(search_keys - detail_keys),
            "detail_only": sorted(detail_keys - search_keys),
            "common": sorted(search_keys & detail_keys),
            "value_diffs": {},
        }
        for key in diff["common"]:
            sv = first_ad.get(key)
            dv = detail_ad.get(key) if isinstance(detail_ad, dict) else None
            if sv != dv:

                def _summarize(val: object) -> object:
                    if isinstance(val, (dict, list)):
                        n = len(val) if hasattr(val, "__len__") else "?"
                        return f"<{type(val).__name__} len={n}>"
                    return val

                diff["value_diffs"][key] = {
                    "search": _summarize(sv),
                    "detail": _summarize(dv),
                }

        search_vs_detail[platform] = diff
        print(f"OK — detail_only={diff['detail_only']}, value_diffs={len(diff['value_diffs'])}")
    else:
        print(f"status={result['status']}")
    time.sleep(0.5)

audit["detail"] = detail_results
audit["search_vs_detail"] = search_vs_detail


# ── Save audit results ───────────────────────────────────────────
audit_path = output_dir / "adflex_api_audit.json"
with open(audit_path, "w") as f:
    json.dump(audit, f, indent=2, default=str)

print(f"\n{'=' * 60}")
print(f"Audit saved to {audit_path} ({audit_path.stat().st_size / 1024:.1f} KB)")
print(f"{'=' * 60}")

# ── Summary ──────────────────────────────────────────────────────
print("\n=== Summary ===")
print("\nPlatform support:")
for p in ALL_PLATFORMS:
    f = all_filters.get(p, {})
    s = all_search.get(p, {})
    d = detail_results.get(p, {})
    f_ok = "✓" if f.get("status") == 200 else "✗"
    s_ok = "✓" if s.get("status") == 200 else "✗"
    s_count = len(s.get("body", {}).get("data", {}).get("ads", [])) if s.get("body") else 0
    d_ok = "✓" if d.get("status") == 200 else "–"
    src = " (cached)" if s.get("source") == "cached" else ""
    print(f"  {p:12s}  filters={f_ok}  search={s_ok} ({s_count:2d} ads){src}  detail={d_ok}")

if search_vs_detail:
    print("\nSearch vs Detail differences:")
    for p, diff in search_vs_detail.items():
        detail_only = diff.get("detail_only", [])
        vdiffs = diff.get("value_diffs", {})
        print(f"  {p}: detail_only_fields={detail_only}, value_diffs={list(vdiffs.keys())}")

# Count new API calls made
new_calls = (
    len(new_filter_platforms)
    + len(AUXILIARY_ENDPOINTS)
    + len(new_search_platforms)
    + len(detail_results)
)
print(f"\nNew API calls made: {new_calls}")
cached = len(EXISTING_FILTER_PLATFORMS) + len(EXISTING_SEARCH_PLATFORMS)
print(f"Reused from cache: {cached} results")
