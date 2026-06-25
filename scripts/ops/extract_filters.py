"""Extract AdFlex filter codes from raw exploration JSON into structured YAML configs.

Reads data/raw/adflex_exploration_raw.json and writes one YAML file per platform
to configs/filters/{platform}.yaml. Run once, commit the output.

Usage:
    python scripts/extract_filters.py
    python scripts/extract_filters.py --input data/raw/adflex_exploration_raw.json
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Extract AdFlex filter codes into YAML configs")
console = Console()

# Filter component types we care about
_SELECT_COMPONENTS = {"StringMulti"}
_RANGE_COMPONENTS = {"NumberMultiRange"}
# Skip these — not useful for collection queries
_SKIP_KEYS = {"search_field", "exact_country", "publish_date", "update_date"}


def _extract_platform(
    platform: str,
    filters_list: list[dict],
) -> dict:
    """Extract filter codes for a single platform."""
    select_filters: dict[str, dict[int, str]] = {}
    range_filters: dict[str, dict[str, int | bool]] = {}
    orderings: dict[str, str] = {}

    for f in filters_list:
        key = f.get("key", "")
        component = f.get("component", "")
        component_data = f.get("component_data", {}) or {}

        if key in _SKIP_KEYS:
            continue

        # Ordering is special — extract as its own section
        if key == "orderby":
            for item in component_data.get("items", []):
                orderings[item["key"]] = item.get("label", "")
            continue

        if component in _SELECT_COMPONENTS or component == "StringSingle":
            items = component_data.get("items", [])
            if items:
                select_filters[key] = {item["key"]: item.get("label", "") for item in items}

        elif component in _RANGE_COMPONENTS:
            range_filters[key] = {
                "min": component_data.get("min", 0),
                "max": component_data.get("max", 0),
                "infinite": component_data.get("infinite_sign", False),
            }

    return {
        "platform": platform,
        "select_filters": select_filters,
        "range_filters": range_filters,
        "orderings": orderings,
    }


@app.command()
def extract(
    input_path: str = typer.Option(
        "data/raw/adflex_exploration_raw.json",
        help="Path to raw exploration JSON",
    ),
    output_dir: str = typer.Option(
        "configs/filters",
        help="Output directory for per-platform YAML files",
    ),
) -> None:
    """Extract filter codes from raw exploration JSON."""
    in_path = Path(input_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with in_path.open() as f:
        raw_data = json.load(f)

    filters_data = raw_data.get("filters", {})
    if not filters_data:
        console.print("[red]No 'filters' key found in input JSON[/red]")
        raise typer.Exit(1)

    # Summary table
    table = Table(title="Extracted Filter Codes")
    table.add_column("Platform")
    table.add_column("Select Filters", justify="right")
    table.add_column("Total Codes", justify="right")
    table.add_column("Range Filters", justify="right")
    table.add_column("Orderings", justify="right")

    index_entries: list[dict[str, str | int]] = []

    for platform, platform_data in filters_data.items():
        filters_list = platform_data.get("data", {}).get("filters", [])
        if not filters_list:
            console.print(f"[yellow]Skipping {platform}: no filter data[/yellow]")
            continue

        extracted = _extract_platform(platform, filters_list)

        # Write YAML
        yaml_path = out_path / f"{platform}.yaml"
        with yaml_path.open("w") as yf:
            yf.write(f"# AdFlex filter codes for {platform}\n")
            yf.write(f"# Auto-generated from {in_path.name}\n\n")
            yaml.dump(extracted, yf, default_flow_style=False, sort_keys=False)

        # Stats
        total_codes = sum(len(v) for v in extracted["select_filters"].values())
        select_count = len(extracted["select_filters"])
        range_count = len(extracted["range_filters"])
        ordering_count = len(extracted["orderings"])

        table.add_row(
            platform,
            str(select_count),
            str(total_codes),
            str(range_count),
            str(ordering_count),
        )

        index_entries.append(
            {
                "platform": platform,
                "file": f"{platform}.yaml",
                "select_filters": select_count,
                "total_codes": total_codes,
                "range_filters": range_count,
            }
        )

        # Print key filter types for this platform
        for filter_name, codes in extracted["select_filters"].items():
            if len(codes) > 0:
                console.print(f"  {platform}/{filter_name}: {len(codes)} codes")

    # Write index
    index_path = out_path / "_index.yaml"
    with index_path.open("w") as f:
        f.write("# AdFlex filter index — auto-generated\n\n")
        yaml.dump({"platforms": index_entries}, f, default_flow_style=False)

    console.print(table)
    console.print(f"\n[green]Wrote {len(index_entries)} platform configs to {out_path}/[/green]")


if __name__ == "__main__":
    app()
