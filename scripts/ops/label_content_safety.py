"""OpenAI Batch-API content-safety labeller.

Labels every raw ad with a content-safety category using gpt-4o-mini via
OpenAI's Batch API (50% cheaper than real-time). Writes
``content_safety_label`` + ``content_safety_confidence`` back to
``data/raw/adflex_ads.jsonl`` in place. Scoring re-runs then propagate the
fields to scored/* automatically.

Why LLM rather than a wordlist library: library-based scanning on this
corpus had ~85% false-positive rate on explicit hits (``god``, ``fat``,
``kill``, ``maxi``, ``virgin``, ``sexual abuse survivors``), missed all
implicit sexual / hateful phrasing, and skipped 21% non-English ads
outright. 4o-mini handles nuance + multilingual in one shot.

Resume-safe: ads with ``content_safety_label != ""`` are skipped on submit.

Workflow:
    python scripts/ops/label_content_safety.py submit   # build JSONL, upload, create batch
    python scripts/ops/label_content_safety.py status   # poll progress
    python scripts/ops/label_content_safety.py collect  # download + merge into raw

Or run all three in sequence:
    python scripts/ops/label_content_safety.py run
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

app = typer.Typer(help="Batch-label raw ads with a content-safety category via gpt-4o-mini.")
console = Console()

# ── Paths ───────────────────────────────────────────────────────────────────
RAW_PATH = Path("data/raw/adflex_ads.jsonl")
BATCH_DIR = Path("data/raw/.labeling_safety")
STATE_PATH = BATCH_DIR / "batch_state.json"
INPUT_JSONL = BATCH_DIR / "batch_input.jsonl"
OUTPUT_JSONL = BATCH_DIR / "batch_output.jsonl"

# ── Taxonomy ────────────────────────────────────────────────────────────────
# Flat list of safety labels. `safe` is the overwhelming majority bucket;
# the other six describe the reason an ad is unsafe so downstream filters
# can gate on specific risks (e.g. keep profanity, drop adult_sexual).
SAFETY_LABELS: list[str] = [
    "safe",
    "profanity",            # swears / crass language, no targeted group, no explicit sex
    "adult_sexual",         # sexual innuendo, suggestive, nudity, porn-adjacent
    "hate_discrimination",  # slurs, derogatory targeting of race/gender/religion/orientation
    "violence_graphic",     # gore, graphic harm, self-harm encouragement
    "substance_drugs",      # illegal drugs, excessive alcohol push, nicotine marketing to minors
    "shock_misleading",     # scam-feel, fear-preying medical/weight-loss, deceptive urgency
]

MODEL = "gpt-4o-mini-2024-07-18"

SYSTEM_PROMPT = (
    "Classify the ad's CONTENT SAFETY — not the product category. Most ads are 'safe'. "
    "Only assign an unsafe label when the copy itself is problematic for a training corpus. "
    "Labels:\n"
    "- safe: normal ad copy, even if the product is alcohol, betting, pharma, or dating.\n"
    "- profanity: uses f-word/s-word/crude swears as-is (not censored, not wordplay). "
    "Mild 'hell'/'damn' is safe.\n"
    "- adult_sexual: sexual innuendo, suggestive/explicit sex, nudity, porn-adjacent offers. "
    "Lingerie ads selling garments are safe; 'hot singles near you'-style is adult_sexual.\n"
    "- hate_discrimination: slurs or content derogating protected groups.\n"
    "- violence_graphic: gore, graphic harm, self-harm or violence encouragement. "
    "Horror-game trailers with stylised violence are safe.\n"
    "- substance_drugs: promoting illegal drugs, pushing excessive consumption, or "
    "targeting nicotine/alcohol at minors. A regulated THC/CBD product sold legally is safe.\n"
    "- shock_misleading: scam-feeling copy, predatory medical/weight-loss claims, "
    "deceptive urgency (fake countdowns, fabricated endorsements).\n"
    "Always pick one label — never refuse. "
    "Use confidence for uncertainty: 1.0 certain, 0.5 likely, 0.2 wild guess. "
    'Return {"label":"<label>","confidence":0-1}.'
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
    if isinstance(rec, dict) and isinstance(rec.get("ad"), dict):
        return rec["ad"]
    return rec


def _load_existing_labels(path: Path) -> set[str]:
    """Return ad_ids that already have a non-empty content_safety_label."""
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
        if ad.get("content_safety_label", ""):
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
    multilingual coverage was the main reason we aren't using a wordlist.
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
    copy = ad.get("ad_copy") or {}
    user = USER_PROMPT_TMPL.format(
        advertiser=(ad.get("advertiser_name", "") or "")[:80],
        url=(ad.get("landing_page_url", "") or "")[:80],
        headline=(copy.get("headline", "") or "").replace("\n", " ")[:240],
        body=(copy.get("body", "") or "").replace("\n", " ")[:240],
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
            "max_tokens": 40,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "content_safety_classification",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "enum": SAFETY_LABELS},
                            "confidence": {"type": "number"},
                        },
                        "required": ["label", "confidence"],
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
) -> None:
    """Build batch input JSONL, upload, create batch, save batch_id."""
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    already = _load_existing_labels(Path(path))
    if already:
        console.print(f"Already labelled: {len(already):,}")

    ads = _load_ads(Path(path), already)
    if limit > 0:
        ads = ads[:limit]
    console.print(f"Ads to label: {len(ads):,}")
    if not ads:
        console.print("[yellow]Nothing to do.[/yellow]")
        return

    console.print(f"Writing batch input → {INPUT_JSONL}")
    with INPUT_JSONL.open("w", encoding="utf-8") as f:
        for ad in ads:
            f.write(json.dumps(_build_request(ad), ensure_ascii=False) + "\n")

    client = _client()
    console.print("Uploading input file …")
    up = client.files.create(file=INPUT_JSONL.open("rb"), purpose="batch")
    console.print(f"  file_id = {up.id}")

    console.print("Creating batch job …")
    batch = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "draper.ai content-safety labelling"},
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
                "label": obj["label"],
                "confidence": float(obj.get("confidence", 0.0)),
            }
        except Exception:
            parse_errors += 1

    if parse_errors:
        console.print(f"[yellow]Parse errors: {parse_errors}[/yellow]")
    console.print(f"Parsed {len(labels):,} labels from batch output.")

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
            new_records.append(line)
            continue
        ad = _unwrap(rec)
        ad_id = ad.get("ad_id", "")
        if ad_id in labels:
            ad["content_safety_label"] = labels[ad_id]["label"]
            ad["content_safety_confidence"] = labels[ad_id]["confidence"]
            updated += 1
        new_records.append(json.dumps(rec, ensure_ascii=False))

    source_path.write_text("\n".join(new_records) + "\n", encoding="utf-8")
    console.print(
        f"[green]✓ Merged {updated:,} labels into {source_path} "
        f"({len(raw_lines):,} total records)[/green]"
    )

    if b.error_file_id:
        console.print(f"[yellow]Error file present: {b.error_file_id}[/yellow]")
        err = client.files.content(b.error_file_id)
        (BATCH_DIR / "batch_errors.jsonl").write_bytes(err.read())

    # Distribution
    dist = Counter(v["label"] for v in labels.values())
    total = sum(dist.values())
    unsafe = total - dist.get("safe", 0)
    t = Table(title="Content-safety distribution (this batch)")
    t.add_column("Label")
    t.add_column("Count", justify="right")
    t.add_column("%", justify="right")
    for lbl, c in dist.most_common():
        style = "" if lbl == "safe" else "red"
        t.add_row(lbl, f"{c:,}", f"{c/total:.1%}", style=style)
    console.print(t)
    console.print(f"[bold]Unsafe total:[/bold] {unsafe:,} / {total:,} ({unsafe/total:.2%})")


@app.command()
def run(
    path: str = typer.Option(str(RAW_PATH), "--path", help="JSONL source (raw or scored)."),
    limit: int = typer.Option(0, "--limit", help="Cap ads (0 = all)"),
    poll_seconds: int = typer.Option(30, "--poll", help="Poll interval while waiting"),
) -> None:
    """Submit + wait for completion + collect, in one call."""
    submit(path=path, limit=limit)

    client = _client()
    state = _load_state()
    batch_id = state["batch_id"]

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
