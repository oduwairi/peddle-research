"""OpenAI Batch-API vertical + training-quality labeller.

Labels every raw ad in one batch with (a) one of 43 business verticals and
(b) a 1-5 training-quality rating. Uses gpt-4o-mini via OpenAI's Batch API
(50% cheaper than real-time). Writes ``business_vertical``,
``business_vertical_confidence``, and ``training_quality`` back to
``data/raw/adflex_ads.jsonl`` in place; scoring re-runs then propagate the
fields to scored/* automatically.

The training-quality rating drives a pre-clustering filter: low-quality
ads (broken copy, pure clickbait, placeholders) are dropped before they
reach construction so we don't waste teacher calls distilling garbage.

Resume-safe by default: ads already carrying a ``business_vertical`` are
skipped on submit. Pass ``--force`` to re-label the full corpus (useful
after taxonomy or rubric changes).

Workflow:
    python scripts/ops/label_verticals.py submit   # build JSONL, upload, create batch
    python scripts/ops/label_verticals.py status   # poll progress
    python scripts/ops/label_verticals.py collect  # download + merge into raw

Or run all three in sequence:
    python scripts/ops/label_verticals.py run
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import typer
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

load_dotenv()

app = typer.Typer(help="Batch-label raw ads with a business vertical via gpt-4o-mini.")
console = Console()

# ── Paths ───────────────────────────────────────────────────────────────────
RAW_PATH = Path("data/raw/adflex_ads.jsonl")
BATCH_DIR = Path("data/raw/.labeling")
STATE_PATH = BATCH_DIR / "batch_state.json"
INPUT_JSONL = BATCH_DIR / "batch_input.jsonl"
OUTPUT_JSONL = BATCH_DIR / "batch_output.jsonl"

# ── Taxonomy ────────────────────────────────────────────────────────────────
#
# Comprehensive single-axis taxonomy: classify by *what is being sold* (the
# product or service itself), not by industry, audience, or advertiser brand.
# Buckets are mutually exclusive by design — every label has a single primary
# criterion so the classifier doesn't have to resolve overlapping axes.
#
# Coverage target: ~95% of commercial ad spend. Deliberate omissions: adult
# content, tobacco/vaping (platform-restricted anyway), and fine-grained
# sub-verticals (e.g. "mortgage" as its own bucket within financial_services)
# that would fragment too small for stratified sampling downstream.
VERTICALS: list[str] = [
    "apparel_accessories",
    "automotive",
    "baby_kids_products",
    "beauty_personal_care",
    "cannabis_cbd",
    "consumer_electronics",
    "consumer_packaged_goods",
    "crypto_web3",
    "dating_apps",
    "events_ticketing",
    "financial_services",
    "gambling_betting",
    "government_public_sector",
    "healthcare_services",
    "home_goods",
    "home_services",
    "industrial_manufacturing",
    "insurance",
    "jewelry_luxury_goods",
    "legal_services",
    "marketing_adtech",
    "news_publishing",
    "nonprofit_charity",
    "online_education",
    "outdoor_sports_gear",
    "personal_services",
    "pet_products",
    "pharmaceuticals",
    "political_advocacy",
    "professional_services",
    "real_estate",
    "recruiting_hr_platforms",
    "religious_faith",
    "restaurants_food_delivery",
    "saas_business_software",
    "social_community_platforms",
    "streaming_media",
    "supplements_wellness",
    "telecom_internet",
    "toys_hobbies_crafts",
    "travel_hospitality",
    "utilities_energy",
    "video_games",
]

MODEL = "gpt-4o-mini-2024-07-18"

SYSTEM_PROMPT = (
    "You will do two things for each ad: assign a business vertical AND "
    "rate the ad's usefulness as training data for a marketing-reasoning "
    "model.\n\n"
    "VERTICAL: Classify by what IS BEING SOLD — the product or service "
    "itself — not by the advertiser's brand, parent industry, or target "
    "audience. Labels are mutually exclusive; pick the single best fit. "
    "Always pick a vertical; never refuse and never return a label "
    "outside the provided list. If the ad genuinely spans two "
    "categories, pick the one the landing page most directly monetizes "
    "and lower confidence accordingly. Confidence: 1.0 unambiguous, "
    "0.7 likely correct, 0.5 best-fit guess, 0.3 near-random.\n\n"
    "TRAINING_QUALITY (1-5): Rate the ad copy as teacher-distillation "
    "material. You are judging the COPY, not the product or engagement.\n"
    "- 1: broken/empty/placeholder (missing fields, Lorem-ipsum-like, "
    "garbled text, pure emoji, visible template variables).\n"
    "- 2: pure clickbait with no substance ('click here', 'you won't "
    "believe', 'swipe up' with no product context).\n"
    "- 3: coherent but generic ('Shop our new collection', bland "
    "feature dump, no distinctive voice).\n"
    "- 4: clear product + value prop + identifiable voice or hook.\n"
    "- 5: distinctive, specific, well-crafted copy with a memorable "
    "hook, angle, or tension that a model could learn craft from.\n"
    "Be calibrated: most real ads are 3. Don't grade on a curve.\n\n"
    'Return {"vertical":"<label>","confidence":0-1,"training_quality":1-5}.'
)

USER_PROMPT_TMPL = "{advertiser} | {url} | {headline} | {body}"


# ──────────────────────────────────────────────────────────────────────────
# State helpers
# ──────────────────────────────────────────────────────────────────────────


def _load_state() -> dict[str, str]:
    if STATE_PATH.exists():
        return dict(json.loads(STATE_PATH.read_text()))
    return {}


def _save_state(state: dict[str, str]) -> None:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def _unwrap(rec: dict) -> dict:
    """Return the ad dict regardless of bare (raw) or wrapped (scored) format."""
    if isinstance(rec, dict) and "ad" in rec and isinstance(rec["ad"], dict):
        return rec["ad"]
    return rec


def _load_existing_labels(path: Path) -> set[str]:
    """Return ad_ids that already have a non-empty business_vertical."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        ad = _unwrap(rec)
        if ad.get("business_vertical", ""):
            ids.add(ad.get("ad_id", ""))
    return ids


def _client() -> OpenAI:
    if "OPENAI_API_KEY" not in os.environ:
        console.print("[red]OPENAI_API_KEY not set in environment.[/red]")
        raise typer.Exit(1)
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _load_ads(path: Path, already_labelled: set[str]) -> list[dict]:
    """Load ads (raw or scored format) not already labelled.

    No language filter — 4o-mini handles most languages well enough, and
    language-based filtering is the clusterer's job, not the labeler's.
    """
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    ads: list[dict] = []
    for line in raw:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        ad = _unwrap(rec)
        ad_id = ad.get("ad_id", "")
        if not ad_id or ad_id in already_labelled:
            continue
        ads.append(ad)
    return ads


def _build_request(ad: dict) -> dict:
    """Build one batch request object for an ad."""
    user = USER_PROMPT_TMPL.format(
        advertiser=(ad.get("advertiser_name", "") or "")[:80],
        url=(ad.get("landing_page_url", "") or "")[:80],
        headline=(ad.get("ad_copy", {}).get("headline", "") or "").replace("\n", " ")[:200],
        body=(ad.get("ad_copy", {}).get("body", "") or "").replace("\n", " ")[:200],
    )
    return {
        "custom_id": ad["ad_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": 60,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "vertical_classification",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "vertical": {"type": "string", "enum": VERTICALS},
                            "confidence": {"type": "number"},
                            "training_quality": {
                                "type": "integer",
                                "enum": [1, 2, 3, 4, 5],
                            },
                        },
                        "required": ["vertical", "confidence", "training_quality"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    }


# ──────────────────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────────────────


@app.command()
def submit(
    path: str = typer.Option(str(RAW_PATH), "--path", help="JSONL source (raw or scored)."),
    limit: int = typer.Option(0, "--limit", help="Cap number of ads (0 = all)"),
    offset: int = typer.Option(0, "--offset", help="Skip first N ads before applying limit."),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-label ads that already carry a business_vertical (for rubric/taxonomy changes).",
    ),
) -> None:
    """Build batch input JSONL, upload, create batch, save batch_id."""
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    already = set() if force else _load_existing_labels(Path(path))
    if force:
        console.print("[yellow]--force: re-labelling all ads (ignoring existing labels).[/yellow]")
    elif already:
        console.print(f"Already labelled: {len(already):,}")

    ads = _load_ads(Path(path), already)
    if offset > 0:
        ads = ads[offset:]
    if limit > 0:
        ads = ads[:limit]
    console.print(f"Ads to label: {len(ads):,}")
    if not ads:
        console.print("[yellow]Nothing to do.[/yellow]")
        return

    # Build JSONL
    console.print(f"Writing batch input → {INPUT_JSONL}")
    with INPUT_JSONL.open("w", encoding="utf-8") as f:
        for ad in ads:
            f.write(json.dumps(_build_request(ad), ensure_ascii=False) + "\n")

    # Upload
    client = _client()
    console.print("Uploading input file …")
    up = client.files.create(file=INPUT_JSONL.open("rb"), purpose="batch")
    console.print(f"  file_id = {up.id}")

    # Create batch
    console.print("Creating batch job …")
    batch = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "draper.ai vertical labelling"},
    )
    console.print(f"[green]✓ Batch created: {batch.id}[/green]")

    state = {
        "batch_id": batch.id,
        "input_file_id": up.id,
        "submitted_at": str(batch.created_at),
        "ads_in_batch": str(len(ads)),
    }
    _save_state(state)
    console.print(f"State saved → {STATE_PATH}")


@app.command()
def status() -> None:
    """Show current batch status and progress."""
    state = _load_state()
    if "batch_id" not in state:
        console.print("[red]No batch state. Run `submit` first.[/red]")
        raise typer.Exit(1)

    client = _client()
    b = client.batches.retrieve(state["batch_id"])

    t = Table(title=f"Batch {b.id}", show_header=False)
    t.add_row("Status", f"[bold]{b.status}[/bold]")
    t.add_row("Total requests", f"{b.request_counts.total:,}")
    t.add_row("Completed", f"{b.request_counts.completed:,}")
    t.add_row("Failed", f"{b.request_counts.failed:,}")
    t.add_row("Output file", b.output_file_id or "—")
    t.add_row("Error file", b.error_file_id or "—")
    t.add_row("Expires at", str(b.expires_at))
    console.print(t)


@app.command()
def collect(
    path: str = typer.Option(str(RAW_PATH), "--path", help="JSONL target to merge labels into."),
) -> None:
    """Download batch output and merge labels back into the source JSONL in place."""
    state = _load_state()
    if "batch_id" not in state:
        console.print("[red]No batch state. Run `submit` first.[/red]")
        raise typer.Exit(1)

    client = _client()
    b = client.batches.retrieve(state["batch_id"])
    if b.status not in {"completed", "cancelled", "expired"}:
        console.print(f"[yellow]Batch not in a harvest-able state: {b.status}[/yellow]")
        raise typer.Exit(1)
    if not b.output_file_id:
        console.print(f"[red]Batch is {b.status} but has no output_file_id.[/red]")
        raise typer.Exit(1)
    if b.status != "completed":
        console.print(
            f"[yellow]Harvesting partial output from {b.status} batch "
            f"({b.request_counts.completed}/{b.request_counts.total}).[/yellow]"
        )

    console.print(f"Downloading output file {b.output_file_id} …")
    blob = client.files.content(b.output_file_id)
    OUTPUT_JSONL.write_bytes(blob.read())

    # Parse batch output → map ad_id → (vertical, confidence, training_quality)
    labels: dict[str, dict] = {}
    parse_errors = 0
    for line in OUTPUT_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            ad_id = rec["custom_id"]
            body = rec["response"]["body"]
            content = body["choices"][0]["message"]["content"]
            obj = json.loads(content)
            labels[ad_id] = {
                "vertical": obj["vertical"],
                "confidence": float(obj.get("confidence", 0.0)),
                "training_quality": int(obj.get("training_quality", 0)),
            }
        except Exception:
            parse_errors += 1

    if parse_errors:
        console.print(f"[yellow]Parse errors: {parse_errors}[/yellow]")
    console.print(f"Parsed {len(labels):,} labels from batch output.")

    # Merge into source JSONL in place (handles both bare and wrapped records)
    source_path = Path(path)
    raw_lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
    updated = 0
    new_records: list[str] = []
    for line in raw_lines:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            new_records.append(line)  # preserve malformed lines as-is
            continue
        ad = _unwrap(rec)
        ad_id = ad.get("ad_id", "")
        if ad_id in labels:
            ad["business_vertical"] = labels[ad_id]["vertical"]
            ad["business_vertical_confidence"] = labels[ad_id]["confidence"]
            ad["training_quality"] = labels[ad_id]["training_quality"]
            updated += 1
        new_records.append(json.dumps(rec, ensure_ascii=False))

    source_path.write_text("\n".join(new_records) + "\n", encoding="utf-8")
    console.print(
        f"[green]✓ Merged {updated:,} labels into {source_path} "
        f"({len(raw_lines):,} total records)[/green]"
    )

    # Error file (if any)
    if b.error_file_id:
        console.print(f"[yellow]Error file present: {b.error_file_id}[/yellow]")
        err = client.files.content(b.error_file_id)
        (BATCH_DIR / "batch_errors.jsonl").write_bytes(err.read())

    # Distribution — vertical
    dist = Counter(v["vertical"] for v in labels.values())
    total = sum(dist.values())
    t = Table(title="Vertical distribution (this batch)")
    t.add_column("Vertical")
    t.add_column("Count", justify="right")
    t.add_column("%", justify="right")
    for v, c in dist.most_common():
        t.add_row(v, f"{c:,}", f"{c/total:.1%}")
    console.print(t)

    # Distribution — training quality
    q_dist = Counter(v["training_quality"] for v in labels.values())
    q_total = sum(q_dist.values())
    tq = Table(title="Training-quality distribution (this batch)")
    tq.add_column("Quality")
    tq.add_column("Count", justify="right")
    tq.add_column("%", justify="right")
    for q in sorted(q_dist.keys()):
        c = q_dist[q]
        tq.add_row(str(q), f"{c:,}", f"{c/q_total:.1%}" if q_total else "—")
    console.print(tq)


@app.command()
def run(
    path: str = typer.Option(str(RAW_PATH), "--path", help="JSONL source (raw or scored)."),
    limit: int = typer.Option(0, "--limit", help="Cap ads (0 = all)"),
    poll_seconds: int = typer.Option(30, "--poll", help="Poll interval while waiting"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-label ads that already carry a business_vertical.",
    ),
) -> None:
    """Submit + wait for completion + collect, in one call."""
    submit(path=path, limit=limit, force=force)

    client = _client()
    state = _load_state()
    batch_id = state["batch_id"]

    # Poll loop
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Waiting for batch …", total=int(state["ads_in_batch"]))
        while True:
            b = client.batches.retrieve(batch_id)
            progress.update(task, completed=b.request_counts.completed)
            progress.update(task, description=f"Batch status: [bold]{b.status}[/bold]")
            if b.status in {"completed", "failed", "expired", "cancelled"}:
                break
            time.sleep(poll_seconds)

    if b.status != "completed":
        console.print(f"[red]Batch ended in status: {b.status}[/red]")
        raise typer.Exit(1)

    collect(path=path)


if __name__ == "__main__":
    app()
