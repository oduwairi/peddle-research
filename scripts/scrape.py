"""Draper.ai scraping CLI — orchestrates data collection from all sources."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from draper.scraping.bigspy import BigSpyClient
from draper.scraping.config import ScrapingConfig
from draper.scraping.rate_limiter import RateLimiter
from draper.scraping.schemas import RawAd
from draper.utils.io import Checkpoint, append_jsonl, write_jsonl
from draper.utils.logging import setup_logging

load_dotenv()

app = typer.Typer(help="Draper.ai data collection CLI")
console = Console()
log = setup_logging()


@app.command()
def bigspy(
    vertical: str = typer.Option("ecommerce", help="Industry vertical to scrape"),
    count: int = typer.Option(500, help="Number of ads to collect"),
    platform: str = typer.Option("facebook", help="Ad platform"),
    country: str = typer.Option(None, help="Country filter"),
    output_dir: str = typer.Option("data/raw", help="Output directory"),
    resume: bool = typer.Option(True, help="Resume from checkpoint if available"),
) -> None:
    """Scrape ads from BigSpy API."""
    api_key = os.environ.get("BIGSPY_API_KEY")
    if not api_key:
        console.print("[red]BIGSPY_API_KEY not set in .env[/red]")
        raise typer.Exit(1)

    output_path = Path(output_dir) / f"bigspy_{vertical}_{platform}.jsonl"
    checkpoint = Checkpoint(output_path) if resume else None

    async def _run() -> list[RawAd]:
        _rl = ScrapingConfig.from_yaml().rate_limits.bigspy
        limiter = RateLimiter(
            requests_per_minute=_rl.requests_per_minute, burst_size=_rl.burst_size
        )
        async with BigSpyClient(api_key=api_key, rate_limiter=limiter) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(f"Scraping {vertical} from BigSpy...", total=None)
                return await client.search_ads_paginated(
                    industry=vertical,
                    platform=platform,
                    country=country,
                    max_results=count,
                    checkpoint=checkpoint,
                )

    ads = asyncio.run(_run())

    if ads:
        written = append_jsonl(ads, output_path)
        console.print(f"[green]Wrote {written} ads to {output_path}[/green]")
    else:
        console.print("[yellow]No ads collected[/yellow]")


@app.command()
def adflex(
    platform: str = typer.Option(
        "facebook", help="Platform: facebook, tiktok, x, pinterest, reddit"
    ),
    keyword: str = typer.Option(None, help="Search keyword"),
    orderby: str = typer.Option("popularity", help="Sort: popularity, days_active, seen_counts"),
    count: int = typer.Option(500, help="Number of ads to collect"),
    output_dir: str = typer.Option("data/raw", help="Output directory"),
    resume: bool = typer.Option(True, help="Resume from checkpoint if available"),
) -> None:
    """Scrape ads from AdFlex API."""
    from draper.scraping.adflex import AdFlexClient

    api_key = os.environ.get("ADFLEX_API_KEY")
    if not api_key:
        console.print("[red]ADFLEX_API_KEY not set in .env[/red]")
        raise typer.Exit(1)

    output_path = Path(output_dir) / f"adflex_{platform}.jsonl"
    checkpoint = Checkpoint(output_path) if resume else None

    async def _run() -> list[RawAd]:
        _rl = ScrapingConfig.from_yaml().rate_limits.adflex
        limiter = RateLimiter(
            requests_per_minute=_rl.requests_per_minute, burst_size=_rl.burst_size
        )
        async with AdFlexClient(api_key=api_key, rate_limiter=limiter) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(f"Scraping {platform} from AdFlex...", total=None)
                return await client.search_ads_paginated(
                    platform=platform,
                    keyword=keyword,
                    orderby=orderby,
                    max_results=count,
                    checkpoint=checkpoint,
                )

    ads = asyncio.run(_run())

    if ads:
        written = write_jsonl(ads, output_path)
        console.print(f"[green]Wrote {written} ads to {output_path}[/green]")
        with_copy = sum(1 for a in ads if a.ad_copy.headline)
        with_engagement = sum(1 for a in ads if a.total_engagement > 0)
        console.print(
            f"  With copy: {with_copy}/{len(ads)}, With engagement: {with_engagement}/{len(ads)}"
        )
    else:
        console.print("[yellow]No ads collected[/yellow]")


@app.command()
def meta(
    search_term: str = typer.Option(None, help="Search keyword"),
    country: str = typer.Option("US", help="Country code (EU codes like DE, FR for richer data)"),
    count: int = typer.Option(100, help="Max ads to collect"),
    ad_type: str = typer.Option("all", help="Ad type filter (all, political_and_issue_ads)"),
    active_only: bool = typer.Option(True, help="Only scrape active ads"),
    output_dir: str = typer.Option("data/raw", help="Output directory"),
) -> None:
    """Scrape ads from Meta Ad Library via Apify.

    EU countries (DE, FR, NL, etc.) return richer data including
    spend ranges, impression ranges, and demographic distribution.
    """
    from draper.scraping.meta_library import MetaLibraryClient

    apify_token = os.environ.get("APIFY_API_TOKEN")
    if not apify_token:
        console.print("[red]APIFY_API_TOKEN not set in .env[/red]")
        raise typer.Exit(1)

    suffix = f"{search_term or 'all'}_{country.lower()}"
    output_path = Path(output_dir) / f"meta_{suffix}.jsonl"

    async def _run() -> list[RawAd]:
        client = MetaLibraryClient(apify_token=apify_token)
        return await client.search_ads(
            search_term=search_term,
            country=country,
            ad_type=ad_type,
            max_results=count,
            active_only=active_only,
        )

    ads = asyncio.run(_run())

    if ads:
        written = append_jsonl(ads, output_path)
        console.print(f"[green]Wrote {written} ads to {output_path}[/green]")
        # Quick summary of data richness
        with_spend = sum(1 for a in ads if a.spend_lower is not None)
        with_impressions = sum(1 for a in ads if a.impression_lower is not None)
        with_copy = sum(1 for a in ads if a.ad_copy.body)
        with_dates = sum(1 for a in ads if a.first_seen and a.last_seen)
        console.print(f"  Ad copy: {with_copy}/{len(ads)}, Dates: {with_dates}/{len(ads)}")
        console.print(
            f"  Spend: {with_spend}/{len(ads)}, Impressions: {with_impressions}/{len(ads)}"
        )
    else:
        console.print("[yellow]No ads collected[/yellow]")


@app.command()
def google(
    query: str = typer.Option(None, help="Search query"),
    advertiser_id: str = typer.Option(None, help="Google advertiser ID"),
    region: str = typer.Option("US", help="Region filter"),
    count: int = typer.Option(100, help="Max ads to collect"),
    output_dir: str = typer.Option("data/raw", help="Output directory"),
) -> None:
    """Scrape ads from Google Ads Transparency Center via SerpApi."""
    from draper.scraping.google_transparency import GoogleTransparencyClient

    serpapi_key = os.environ.get("SERPAPI_API_KEY")
    if not serpapi_key:
        console.print("[red]SERPAPI_API_KEY not set in .env[/red]")
        raise typer.Exit(1)

    output_path = Path(output_dir) / f"google_{region.lower()}.jsonl"

    async def _run() -> list[RawAd]:
        async with GoogleTransparencyClient(serpapi_key=serpapi_key) as client:
            return await client.search_ads(
                query=query,
                advertiser_id=advertiser_id,
                region=region,
                max_results=count,
            )

    ads = asyncio.run(_run())

    if ads:
        written = append_jsonl(ads, output_path)
        console.print(f"[green]Wrote {written} ads to {output_path}[/green]")
    else:
        console.print("[yellow]No ads collected[/yellow]")


@app.command()
def tiktok(
    search_term: str = typer.Option(None, help="Search keyword"),
    country: str = typer.Option("US", help="Country code"),
    count: int = typer.Option(100, help="Max ads to collect"),
    output_dir: str = typer.Option("data/raw", help="Output directory"),
) -> None:
    """Scrape ads from TikTok Ad Library via Apify."""
    from draper.scraping.tiktok_library import TikTokLibraryClient

    apify_token = os.environ.get("APIFY_API_TOKEN")
    if not apify_token:
        console.print("[red]APIFY_API_TOKEN not set in .env[/red]")
        raise typer.Exit(1)

    output_path = Path(output_dir) / f"tiktok_{country.lower()}.jsonl"

    async def _run() -> list[RawAd]:
        client = TikTokLibraryClient(apify_token=apify_token)
        return await client.search_ads(
            search_term=search_term,
            country=country,
            max_results=count,
        )

    ads = asyncio.run(_run())

    if ads:
        written = append_jsonl(ads, output_path)
        console.print(f"[green]Wrote {written} ads to {output_path}[/green]")
    else:
        console.print("[yellow]No ads collected[/yellow]")


@app.command()
def knowledge(
    urls_file: str = typer.Argument(help="Path to JSON file with URL list"),
    output_dir: str = typer.Option("data/raw", help="Output directory"),
    max_concurrent: int = typer.Option(5, help="Max concurrent extractions"),
) -> None:
    """Extract structured marketing knowledge from a list of URLs."""
    import json

    from draper.scraping.knowledge_corpus import process_url_list

    urls_path = Path(urls_file)
    if not urls_path.exists():
        console.print(f"[red]File not found: {urls_file}[/red]")
        raise typer.Exit(1)

    with urls_path.open() as f:
        urls = json.load(f)

    output_path = Path(output_dir) / "knowledge_corpus.jsonl"

    async def _run() -> None:
        articles = await process_url_list(urls, max_concurrent=max_concurrent)
        if articles:
            written = append_jsonl(articles, output_path)
            console.print(f"[green]Wrote {written} articles to {output_path}[/green]")
        else:
            console.print("[yellow]No articles extracted[/yellow]")

    asyncio.run(_run())


if __name__ == "__main__":
    app()
