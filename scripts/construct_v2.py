"""Construction v2 CLI — thin dispatch over ``draper.construction_v2.pipeline``.

Subcommands:

    select          Pick the source-ad batch; write selection.parquet + lineage hash.
    submit          Submit a single-pass teacher batch for one provider.
    collect         Poll + ingest a batch's results; remove on success.
    list            Show in-flight batches.
    cancel          Cancel an in-flight batch.
    ingest          Parse responses + leak guard + fidelity + grounding.
    filter          Run the v2 quality filter.
    build           Assemble the HF DatasetDict at data/final_v2/.

Single-pass is the only production teacher. ``--run-id`` scopes batch
registries and audit overlays under ``data/constructed_v2/runs/<run_id>/``
for smoke and exploration runs.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

# Ensure ``src/`` is importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from draper.construction.batch import make_batch_client  # noqa: E402
from draper.construction.batch.registry import (  # noqa: E402
    BatchRegistry,
    PendingBatch,
    PendingBatchSidecar,
    utc_now_iso,
)
from draper.construction_v2 import pipeline  # noqa: E402
from draper.construction_v2.captions import (  # noqa: E402
    CAPTION_PROMPT_VARIANTS,
    CAPTION_TASK_FORMAT,
    build_caption_request,
    captions_path,
    estimate_caption_cost_usd,
    iter_captionable_ads,
    parse_caption_response,
    write_caption_rows,
)
from draper.construction_v2.captions.builder import (  # noqa: E402
    CaptionableAd,
    CaptionRow,
    load_captions_lookup,
)
from draper.construction_v2.config import ConstructionV2Config  # noqa: E402
from draper.construction_v2.dataset.builder import build_dataset  # noqa: E402
from draper.construction_v2.dataset.quality_filter import QualityFilter  # noqa: E402
from draper.construction_v2.dataset.source_selector import (  # noqa: E402
    SourceAd,
    load_source_ads_by_id,
    select_source_ads,
)
from draper.construction_v2.ingest.response_parser import (  # noqa: E402
    ParsedResponse,
    parse_response,
)
from draper.construction_v2.ingest.skills import get_bundle  # noqa: E402
from draper.construction_v2.schemas.records import ExampleRecord  # noqa: E402
from draper.utils.io import read_jsonl, write_jsonl  # noqa: E402
from draper.utils.logging import setup_logging  # noqa: E402

app = typer.Typer(help="Draper.ai construction v2 pipeline.")
console = Console()


def _load(config_path: str) -> ConstructionV2Config:
    return ConstructionV2Config.from_yaml(config_path)


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


@app.command()
def select(
    config_path: str = typer.Option("configs/construction_v2.yaml", "--config", "-c"),
    target: int | None = typer.Option(None, "--target", help="Override target_count."),
    min_composite: float | None = typer.Option(
        None, "--min-composite", help="Override min_composite."
    ),
    allow_unbalanced: bool = typer.Option(
        False,
        "--allow-unbalanced",
        help=(
            "Override config.selection.allow_unbalanced "
            "(take top-N by composite, then enforce max_platform_share)."
        ),
    ),
    force_unbalanced: bool = typer.Option(
        False,
        "--force-unbalanced",
        help="Waive max_platform_share assertion (dangerous on skewed corpora).",
    ),
    exclude_from_run: list[str] = typer.Option(  # noqa: B008
        [],
        "--exclude-from-run",
        help=(
            "Exclude ad_ids already consumed by a prior run_id "
            "(reads runs/<run_id>/copywriting/briefs.jsonl). Repeatable."
        ),
    ),
) -> None:
    """Pick the source-ad batch and write the selection audit."""
    setup_logging()
    config = _load(config_path)
    if target is not None:
        config.selection.target_count = target
    if min_composite is not None:
        config.selection.min_composite = min_composite
    if allow_unbalanced:
        config.selection.allow_unbalanced = True
    excluded: set[str] = set()
    for run in exclude_from_run:
        run_ids: set[str] = set()
        # Skill-correct briefs path (runs/<run>/<skill>/briefs.jsonl). Hardcoding
        # "copywriting" here silently missed every collected brief on the
        # image_brief skill, so exclusion only worked while batches were still
        # in-flight (found via the registry below).
        prior = pipeline.briefs_path(config, run_id=run)
        if prior.exists():
            for row in read_jsonl(prior):
                aid = row.get("ad_id") if isinstance(row, dict) else None
                if isinstance(aid, str):
                    run_ids.add(aid)
        # Also pull ad_ids from any in-flight batches still in the
        # registry — these have been reserved but not yet written to
        # briefs.jsonl by collect.
        try:
            registry_batches = pipeline.registry_for(config, run_id=run).all()
        except Exception as exc:  # noqa: BLE001 — registry load failures
            console.print(f"[yellow]Could not load registry for run `{run}`: {exc}[/yellow]")
            registry_batches = []

        in_flight = 0
        for batch in registry_batches:
            for sc in batch.sidecars:
                for aid in sc.source_ad_ids:
                    if aid not in run_ids:
                        in_flight += 1
                    run_ids.add(aid)
        if not run_ids:
            console.print(f"[red]run `{run}` has no briefs.jsonl or in-flight batches[/red]")
            raise typer.Exit(code=2)
        excluded |= run_ids
        console.print(
            f"excluding {len(run_ids)} ad_ids consumed by run `{run}` "
            f"({len(run_ids) - in_flight} collected + {in_flight} in-flight)"
        )
    if excluded:
        console.print(f"total excluded (deduped across runs): {len(excluded)}")
    chosen = select_source_ads(
        config,
        force_unbalanced=force_unbalanced,
        exclude_ad_ids=excluded or None,
    )
    console.print(f"[green]selected[/green] {len(chosen)} ads")


# ---------------------------------------------------------------------------
# submit / collect / list / cancel
# ---------------------------------------------------------------------------


@app.command()
def submit(
    provider: str = typer.Option(..., "--provider", "-p", help="anthropic | openai | gemini"),
    config_path: str = typer.Option("configs/construction_v2.yaml", "--config", "-c"),
    slice_spec: str = typer.Option(
        "0/1",
        "--slice",
        help="Disjoint ad-id partition i/N from selection.parquet (e.g. 0/3).",
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Cap requests submitted within the slice."
    ),
    model_override: str | None = typer.Option(
        None, "--model", help="Override `providers[<provider>].model`."
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Scope registry+audit under data/constructed_v2/runs/<run_id>/.",
    ),
    allow_lineage_drift: bool = typer.Option(
        False,
        "--allow-lineage-drift",
        help="Proceed even if selection.parquet hash no longer matches config.",
    ),
) -> None:
    """Submit a single-pass teacher batch for one provider's ad slice."""
    setup_logging()
    config = _load(config_path)
    result = asyncio.run(
        pipeline.submit_single_pass(
            config,
            provider=provider,
            slice_spec=slice_spec,
            limit=limit,
            model_override=model_override,
            run_id=run_id,
            allow_lineage_drift=allow_lineage_drift,
        )
    )
    if result.skipped:
        console.print(f"[yellow]nothing to submit[/yellow] ({result.skipped_reason})")
        return
    console.print(
        f"[green]submitted single-pass batch[/green] {result.batch_id} "
        f"({result.request_count} requests, provider={result.provider}, "
        f"model={result.model})"
    )


_TERMINAL_BATCH_STATUSES = {
    "completed",
    "failed",
    "cancelled",
    "cancelled_stuck",
    "expired",
    "ended",
}


async def _refresh_registry(registry: BatchRegistry, batches: list[PendingBatch]) -> None:
    """Live-poll each non-terminal batch and write the result back to
    the registry so the rendered table reflects current provider state.

    Errors per batch are logged but do not abort the refresh — stale
    rows just keep their cached status.
    """
    import logging

    log = logging.getLogger("draper")
    for b in batches:
        if b.status in _TERMINAL_BATCH_STATUSES:
            continue
        try:
            client = make_batch_client(b.model)
            info = await client.poll(b.batch_id)
        except Exception as exc:  # noqa: BLE001 — provider-specific failures
            log.warning("Refresh failed for %s: %s", b.batch_id, exc)
            continue
        registry.update_status(
            b.batch_id,
            status=info.status.value,
            completed_count=info.completed_count,
            failed_count=info.failed_count,
            request_count=info.request_count,
        )


@app.command()
def list_(  # `list` shadows a builtin but Typer prefers the explicit name
    config_path: str = typer.Option("configs/construction_v2.yaml", "--config", "-c"),
    run_id: str | None = typer.Option(None, "--run-id"),
    registry_path: str | None = typer.Option(
        None,
        "--registry-path",
        help="Inspect an arbitrary registry JSON (escape hatch for legacy runs).",
    ),
    refresh: bool = typer.Option(
        True,
        "--refresh/--no-refresh",
        help="Poll providers for live status of non-terminal batches before rendering.",
    ),
) -> None:
    """Show batches tracked in the (run-scoped) v2 registry.

    By default polls each non-terminal batch live so the table reflects
    current provider state; pass ``--no-refresh`` to read cached status
    only (faster, but possibly stale).
    """
    setup_logging()
    config = _load(config_path)
    registry: BatchRegistry
    if registry_path:
        registry = BatchRegistry(Path(registry_path).parent.parent, pipeline.TASK_FORMAT)
    else:
        registry = pipeline.registry_for(config, run_id=run_id)
    batches = registry.all()
    if not batches:
        console.print("[yellow]no batches tracked[/yellow]")
        return
    if refresh:
        non_terminal = sum(1 for b in batches if b.status not in _TERMINAL_BATCH_STATUSES)
        if non_terminal:
            console.print(f"[dim]polling {non_terminal} non-terminal batch(es)…[/dim]")
            asyncio.run(_refresh_registry(registry, batches))
            batches = registry.all()
    title = f"construction_v2 batches (run_id={run_id or '<production>'})"
    table = Table(title=title)
    table.add_column("batch_id")
    table.add_column("provider")
    table.add_column("model")
    table.add_column("status")
    table.add_column("requests", justify="right")
    table.add_column("done", justify="right")
    table.add_column("failed", justify="right")
    table.add_column("submitted_at")
    for b in batches:
        table.add_row(
            b.batch_id,
            b.provider,
            b.model,
            b.status,
            str(b.request_count),
            str(b.completed_count),
            str(b.failed_count),
            b.submitted_at,
        )
    console.print(table)


# Alias `list_` to `list` from the user's perspective.
app.command(name="list")(list_)


@app.command()
def collect(
    batch_id: str = typer.Argument(..., help="Batch ID to collect."),
    config_path: str = typer.Option("configs/construction_v2.yaml", "--config", "-c"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Poll a batch, persist its single-pass results, remove on success."""
    setup_logging()
    config = _load(config_path)
    try:
        result = asyncio.run(pipeline.collect_batch(config, batch_id, run_id=run_id))
    except pipeline.PartialFailureThreshold as e:
        console.print(f"[red]partial-failure threshold tripped[/red] {e}")
        raise typer.Exit(code=2) from e
    if result.missing:
        console.print(f"[red]batch not found in registry:[/red] {batch_id}")
        return
    if result.stuck_cancelled:
        console.print(
            f"[yellow]stuck-batch force-cancelled[/yellow] {batch_id} "
            f"(exceeded stuck_timeout_minutes)"
        )
        return
    if not result.terminal:
        console.print(f"[yellow]not terminal yet[/yellow] {batch_id} (status={result.status})")
        return
    console.print(
        f"[green]terminal[/green] {batch_id} status={result.status} "
        f"briefs={result.briefs_written} rationales={result.rationales_written} "
        f"parse_failures={result.parse_failures} "
        f"provider_errors={result.provider_errors}"
    )


@app.command()
def cancel(
    batch_id: str = typer.Argument(...),
    config_path: str = typer.Option("configs/construction_v2.yaml", "--config", "-c"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Cancel a tracked batch."""
    setup_logging()
    config = _load(config_path)
    registry = pipeline.registry_for(config, run_id=run_id)
    pending = registry.get(batch_id)
    if pending is None:
        console.print(f"[red]batch not found:[/red] {batch_id}")
        return
    client = make_batch_client(pending.model)
    info = asyncio.run(client.cancel(batch_id))
    registry.update_status(batch_id, status=info.status.value)
    console.print(f"[yellow]cancelled[/yellow] {batch_id} → {info.status.value}")


# ---------------------------------------------------------------------------
# Captioning subcommands (image-brief skill prerequisite)
# ---------------------------------------------------------------------------


# Public sync (non-batch) prices used to print a pre-submit cost estimate.
# These are caller-side estimates — actual cost comes from usage_metadata
# in the collected batch. Keys are the BatchRequest.model strings.
# Verify against ai.google.dev / openai.com / anthropic.com pricing before
# relying on the projection for large submissions.
_CAPTIONER_PRICES_USD_PER_M: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash": (0.30, 2.50),
    "gemini-2.5-flash": (0.30, 2.50),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5.4-mini": (0.40, 1.60),
}


def _slice_captionable(ads: list[CaptionableAd], *, slice_spec: str) -> list[CaptionableAd]:
    """Slice the captionable list per ``pipeline.parse_slice_spec``."""
    start, end, n = pipeline.parse_slice_spec(slice_spec)
    total = len(ads)
    size = (total + n - 1) // n
    lo = start * size
    hi = min((end + 1) * size, total)
    return ads[lo:hi]


def _caption_registry(config: ConstructionV2Config, run_id: str | None) -> BatchRegistry:
    """Registry for captioning batches.

    Distinct ``task_format`` (``vlm_caption_v1``) keeps caption batches
    separate from teacher batches in the same on-disk tree.
    """
    root = pipeline.runs_dir(config, run_id) if run_id else Path(config.output_dir)
    return BatchRegistry(root, CAPTION_TASK_FORMAT)


@app.command(name="caption-submit")
def caption_submit(
    provider: str = typer.Option(..., "--provider", "-p", help="anthropic | openai | gemini"),
    config_path: str = typer.Option("configs/construction_v2.yaml", "--config", "-c"),
    slice_spec: str = typer.Option(
        "0/1",
        "--slice",
        help="Disjoint ad-id partition i/N (or i-j/N) over the captionable corpus.",
    ),
    prompt_variant: str = typer.Option(
        "literal",
        "--prompt",
        help=f"Which captioning prompt to use ({', '.join(CAPTION_PROMPT_VARIANTS)}).",
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Cap requests submitted within the slice (smoke runs)."
    ),
    model_override: str | None = typer.Option(
        None,
        "--model",
        help="Override the captioner model (defaults to providers[<provider>].model).",
    ),
    skip_already_captioned: bool = typer.Option(
        True,
        "--skip-existing/--recaption",
        help=(
            "Skip ads already present in data/captions/v1/captions.parquet "
            "(default; --recaption overwrites)."
        ),
    ),
    bypass_selection: bool = typer.Option(
        False,
        "--bypass-selection",
        help=(
            "Caption the entire image-capable corpus instead of the current "
            "selection.parquet. Smoke / exploration only; production captions "
            "should always run post-select."
        ),
    ),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Scope the registry under runs/<run_id>/."
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the cost-estimate confirmation prompt.",
    ),
) -> None:
    """Submit a VLM captioning batch over the current selection's image ads.

    Captioning is a construction step, not a sourcing step: by default
    this command reads the active config's ``selection.parquet`` and
    only captions ads chosen by ``select``. That keeps the captioning
    corpus size matched to the construction corpus (no wasted VLM calls
    on ads the teacher will never see).

    Pass ``--bypass-selection`` to caption the full image-capable corpus
    — only useful for Phase-0-style smoke runs that need captions
    before any selection has happened.

    Captions land in ``data/captions/v1/captions.parquet`` after a
    follow-up ``caption-collect`` call. Re-running a slice with the
    same provider+model+prompt is idempotent under ``--skip-existing``
    (the default); use ``--recaption`` to force re-captioning.
    """
    setup_logging()
    config = _load(config_path)

    if prompt_variant not in CAPTION_PROMPT_VARIANTS:
        console.print(
            f"[red]unknown prompt variant[/red] {prompt_variant!r}; "
            f"choose from {sorted(CAPTION_PROMPT_VARIANTS)}"
        )
        raise typer.Exit(code=2)

    provider_cfg = config.provider_config(provider)
    model = model_override or provider_cfg.model

    scored_path = Path(config.selection.scored_ads_path)
    existing_ad_ids: set[str] = set()
    if skip_already_captioned and captions_path().exists():
        import polars as pl  # local import — polars is otherwise unused here

        # Only skip ads with a SUCCESSFUL caption. Rows where the provider
        # returned an error (e.g. Gemini "Cannot fetch content" misfires)
        # remain in the parquet for audit, but should be eligible for retry.
        # Match empty string AND null (both indicate success).
        df = pl.read_parquet(captions_path())
        existing_ad_ids = set(
            df.filter((pl.col("provider_error") == "") | pl.col("provider_error").is_null())[
                "ad_id"
            ].to_list()
        )

    # Production path: intersect with the current selection.parquet so
    # we never caption an ad the teacher won't see. Bypass only for
    # smoke/exploration runs that don't run select first.
    include_ad_ids: set[str] | None = None
    if not bypass_selection:
        try:
            selection_ids = pipeline.load_selection_ad_ids(config)
        except FileNotFoundError as exc:
            console.print(
                f"[red]selection.parquet not found:[/red] {exc}\n"
                "Run `construct_v2 select` first, or pass --bypass-selection "
                "to caption the full image-capable corpus."
            )
            raise typer.Exit(code=2) from exc
        include_ad_ids = set(selection_ids)
        console.print(
            f"captioning will be intersected with [bold]{len(include_ad_ids):,}[/bold] "
            f"selected ad_ids (skill={config.skill})"
        )

    all_ads = list(
        iter_captionable_ads(
            scored_path,
            exclude_ad_ids=existing_ad_ids,
            include_ad_ids=include_ad_ids,
        )
    )
    sliced = _slice_captionable(all_ads, slice_spec=slice_spec)
    if limit is not None:
        sliced = sliced[:limit]
    if not sliced:
        console.print(
            f"[yellow]nothing to caption[/yellow] (slice empty after "
            f"existing-skip; total captionable={len(all_ads)})"
        )
        return

    # Pre-submit cost preview.
    prices = _CAPTIONER_PRICES_USD_PER_M.get(model)
    if prices is not None:
        in_p, out_p = prices
        est_usd = estimate_caption_cost_usd(
            n_ads=len(sliced),
            input_price_per_m=in_p,
            output_price_per_m=out_p,
        )
        console.print(
            f"about to caption [bold]{len(sliced)}[/bold] ads with {model} "
            f"({provider} batch). estimated cost (batch tier): "
            f"[bold]${est_usd:.2f}[/bold]"
        )
    else:
        console.print(
            f"about to caption [bold]{len(sliced)}[/bold] ads with {model} "
            f"({provider} batch). [yellow]no public price entry for {model};[/yellow] "
            f"add one to _CAPTIONER_PRICES_USD_PER_M to enable estimates."
        )
    if not yes:
        confirm = typer.confirm("submit?", default=False)
        if not confirm:
            console.print("[yellow]aborted[/yellow]")
            return

    requests = [
        build_caption_request(
            ad,
            model=model,
            prompt_variant=prompt_variant,
            max_tokens=1024,
            temperature=0.3,
        )
        for ad in sliced
    ]
    sidecars = [
        PendingBatchSidecar(
            custom_id=req.custom_id,
            prompt_index=i,
            source_ad_ids=[ad.ad_id],
            prompt_style=prompt_variant,
            provider_label=provider,
        )
        for i, (ad, req) in enumerate(zip(sliced, requests, strict=True))
    ]

    client = make_batch_client(model)
    info = asyncio.run(client.submit(requests))

    registry = _caption_registry(config, run_id)
    registry.add(
        PendingBatch(
            batch_id=info.batch_id,
            provider=info.provider,
            model=model,
            task_format=CAPTION_TASK_FORMAT,
            submitted_at=utc_now_iso(),
            status=info.status.value,
            request_count=len(requests),
            sidecars=sidecars,
        )
    )
    console.print(
        f"[green]submitted captioning batch[/green] {info.batch_id} "
        f"({len(requests)} requests, provider={info.provider}, model={model}, "
        f"prompt={prompt_variant})"
    )


@app.command(name="caption-collect")
def caption_collect(
    batch_id: str = typer.Argument(..., help="Captioning batch ID to collect."),
    config_path: str = typer.Option("configs/construction_v2.yaml", "--config", "-c"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Collect a finished captioning batch, append rows to captions.parquet."""
    setup_logging()
    config = _load(config_path)
    registry = _caption_registry(config, run_id)
    pending = registry.get(batch_id)
    if pending is None:
        console.print(f"[red]captioning batch not found in registry:[/red] {batch_id}")
        raise typer.Exit(code=2)

    client = make_batch_client(pending.model)
    info = asyncio.run(client.poll(batch_id))
    if info.status.value != "completed":
        console.print(
            f"[yellow]batch not completed[/yellow] {batch_id} (status={info.status.value})"
        )
        registry.update_status(
            batch_id,
            status=info.status.value,
            completed_count=info.completed_count,
            failed_count=info.failed_count,
            request_count=info.request_count,
        )
        return

    responses = asyncio.run(client.fetch_results(batch_id))

    # Resolve ad_id → CaptionableAd for the slice this batch covered.
    ad_ids_in_batch: set[str] = set()
    for s in pending.sidecars:
        ad_ids_in_batch.update(s.source_ad_ids)
    scored_path = Path(config.selection.scored_ads_path)
    ad_lookup: dict[str, CaptionableAd] = {}
    for ad in iter_captionable_ads(scored_path):
        if ad.ad_id in ad_ids_in_batch:
            ad_lookup[ad.ad_id] = ad
            if len(ad_lookup) == len(ad_ids_in_batch):
                break

    rows: list[CaptionRow] = []
    matched = 0
    missing_ad = 0
    provider_errors = 0
    for resp in responses:
        # custom_id format: caption-<ad_id>
        prefix = "caption-"
        if not resp.custom_id.startswith(prefix):
            continue
        ad_id = resp.custom_id[len(prefix) :]
        ad = ad_lookup.get(ad_id)
        if ad is None:
            missing_ad += 1
            continue
        # Use the sidecar's prompt_style as the prompt_version — set at submit time.
        sidecar = pending.sidecar_by_custom_id(resp.custom_id)
        prompt_variant = sidecar.prompt_style if sidecar else "unknown"
        row = parse_caption_response(
            resp,
            ad=ad,
            captioner_model=pending.model,
            prompt_variant=prompt_variant,
        )
        rows.append(row)
        matched += 1
        if row.provider_error:
            provider_errors += 1

    out = write_caption_rows(rows)
    registry.remove(batch_id)
    console.print(
        f"[green]collected[/green] {batch_id} → {out} "
        f"({matched} rows, provider_errors={provider_errors}, missing_ad_lookup={missing_ad})"
    )


# ---------------------------------------------------------------------------
# Legacy aliases (hidden) — kept for one migration phase.
# ---------------------------------------------------------------------------


@app.command(name="batch-list", hidden=True)
def _alias_batch_list(
    config_path: str = typer.Option("configs/construction_v2.yaml", "--config", "-c"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Deprecated: use `list`."""
    list_(config_path=config_path, run_id=run_id, registry_path=None)


@app.command(name="batch-cancel", hidden=True)
def _alias_batch_cancel(
    batch_id: str = typer.Argument(...),
    config_path: str = typer.Option("configs/construction_v2.yaml", "--config", "-c"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Deprecated: use `cancel`."""
    cancel(batch_id=batch_id, config_path=config_path, run_id=run_id)


# ---------------------------------------------------------------------------
# ingest / filter / build
# ---------------------------------------------------------------------------


@app.command()
def ingest(
    config_path: str = typer.Option("configs/construction_v2.yaml", "--config", "-c"),
    input_path: str | None = typer.Option(
        None, "--input", help="Override responses_raw.jsonl path."
    ),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Parse responses + leak guard + fidelity + grounding."""
    setup_logging()
    config = _load(config_path)
    try:
        result = pipeline.ingest_responses(
            config,
            input_path=Path(input_path) if input_path else None,
            run_id=run_id,
        )
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return
    s = result.stats
    console.print(
        f"[green]ingest[/green] in={s.total_input} ok={s.passed} "
        f"parse_fail={s.parse_failed} teacher_fail={s.teacher_failed} "
        f"fidelity_fail={s.fidelity_failed} grounding_fail={s.grounding_failed} "
        f"labels_fail={s.labels_failed} leak={s.leak_failed} "
        f"content_bridge_fail={s.content_bridge_failed} "
        f"missing_brief={s.missing_brief} missing_ad={s.missing_ad}"
    )
    console.print(f"wrote: {pipeline.examples_path(config, run_id=run_id)}")


@app.command(name="filter")  # `filter` shadows a builtin but Typer prefers it
def filter_cmd(
    config_path: str = typer.Option("configs/construction_v2.yaml", "--config", "-c"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Run the v2 quality filter over the ingested examples."""
    setup_logging()
    config = _load(config_path)
    ex_path = pipeline.examples_path(config, run_id=run_id)
    if not ex_path.exists():
        console.print(f"[red]no examples to filter at[/red] {ex_path}")
        return
    examples = [ExampleRecord.model_validate(r) for r in read_jsonl(ex_path)]
    ads_by_id = load_source_ads_by_id(config, [e.ad_id for e in examples])
    result = QualityFilter(config.filter, ads_by_id=ads_by_id).filter_all(examples)
    out_path = pipeline.filtered_path(config, run_id=run_id)
    write_jsonl([e.model_dump(mode="json") for e in result.passed], out_path)
    pipeline.append_jsonl(
        [r.model_dump(mode="json") for r in result.rejected],
        pipeline.audit_path(config, "filter_rejections.jsonl", run_id=run_id),
    )
    console.print(
        f"[green]filter[/green] in={result.stats.total_input} "
        f"passed={result.stats.passed} "
        f"length={result.stats.rejected_length} "
        f"dedup={result.stats.rejected_duplicate}"
    )
    console.print(f"wrote: {out_path}")


@app.command()
def build(
    config_path: str = typer.Option("configs/construction_v2.yaml", "--config", "-c"),
    output: str | None = typer.Option(None, "--output", help="Override final_dir."),
    input_path: str | None = typer.Option(None, "--input", help="Override filtered.jsonl path."),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Assemble the final HF DatasetDict."""
    setup_logging()
    config = _load(config_path)
    in_path = Path(input_path) if input_path else pipeline.filtered_path(config, run_id=run_id)
    if not in_path.exists():
        console.print(f"[red]no filtered examples at[/red] {in_path}")
        return
    examples = [ExampleRecord.model_validate(r) for r in read_jsonl(in_path)]
    out_dir = Path(output) if output else Path(config.final_dir)
    build_dataset(
        examples,
        out_dir,
        dataset_config=config.dataset,
        audit_dir=config.audit_dir,
    )
    console.print(f"[green]built[/green] DatasetDict at {out_dir}")


# ---------------------------------------------------------------------------
# render — Markdown audit of collected single-pass results
# ---------------------------------------------------------------------------


def _md_blockquote(text: str) -> str:
    if not text:
        return "> _(empty)_"
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def _md_fence(content: str, lang: str = "") -> str:
    return f"```{lang}\n{content}\n```"


def _md_source_block(ad: SourceAd) -> str:
    advertiser = ad.raw.get("advertiser_name") if isinstance(ad.raw, dict) else None
    landing = ad.raw.get("landing_page_url") if isinstance(ad.raw, dict) else None
    creative_url = ad.raw.get("creative_url") if isinstance(ad.raw, dict) else None
    lines = [
        f"- **ad_id**: `{ad.ad_id}`",
        f"- **platform**: `{ad.platform}`",
    ]
    if isinstance(advertiser, str) and advertiser:
        lines.append(f"- **advertiser**: {advertiser}")
    if isinstance(landing, str) and landing:
        lines.append(f"- **landing_page**: {landing}")
    if isinstance(creative_url, str) and creative_url:
        # Plain URL line (smoke-comparison parity) plus an inline embed so the
        # real creative is viewable beside the generated image brief.
        lines.append(f"- **creative_url**: {creative_url}")
        lines.extend(["", f"![ad {ad.ad_id} creative]({creative_url})"])
    lines.extend(["", "**Ad copy** (verbatim):", ""])
    for fld in ("headline", "body", "description"):
        val = getattr(ad, fld) or ""
        if val:
            lines.extend([f"- *{fld}*:", "", _md_fence(val), ""])
        else:
            lines.append(f"- *{fld}*: _(empty)_")
    return "\n".join(lines)


@app.command()
def render(
    config_path: str = typer.Option("configs/construction_v2.yaml", "--config", "-c"),
    run_id: str | None = typer.Option(None, "--run-id"),
    output: str | None = typer.Option(
        None, "--output", help="Override output .md path (default: timestamped)."
    ),
) -> None:
    """Render collected single-pass briefs+responses as a reviewable Markdown.

    Reads ``briefs.jsonl`` + ``responses_raw.jsonl`` from the run's
    copywriting directory and writes a per-ad section listing the
    source ad, the teacher's parsed ``<brief>`` JSON, the ``<think>``
    block as a blockquote, and the deliverable in a fenced block.
    """
    from datetime import datetime

    setup_logging()
    config = _load(config_path)

    briefs_p = pipeline.briefs_path(config, run_id=run_id)
    responses_p = pipeline.responses_path(config, run_id=run_id)
    if not briefs_p.exists() or not responses_p.exists():
        console.print(f"[red]missing inputs[/red] briefs={briefs_p} responses={responses_p}")
        raise typer.Exit(code=1)

    briefs: dict[str, dict[str, object]] = {}
    for row in read_jsonl(briefs_p):
        if isinstance(row, dict) and "ad_id" in row:
            brief_val = row.get("brief", {})
            briefs[row["ad_id"]] = brief_val if isinstance(brief_val, dict) else {}

    responses: dict[str, dict[str, object]] = {}
    for row in read_jsonl(responses_p):
        if isinstance(row, dict) and "ad_id" in row:
            responses[row["ad_id"]] = row

    ad_ids = sorted(responses.keys())
    ads_by_id = load_source_ads_by_id(config, ad_ids)
    bundle = get_bundle(config.skill)
    # Captions are the image-brief supervision target; load them so each ad
    # section can show the VLM caption beside the generated brief (smoke parity).
    captions = load_captions_lookup() if config.skill == "image_brief" else {}

    by_model: dict[str, int] = {}
    for r in responses.values():
        m = str(r.get("model") or "unknown")
        by_model[m] = by_model.get(m, 0) + 1

    lines: list[str] = [
        f"# Single-pass review · run_id `{run_id or '<production>'}`",
        "",
        f"- ads with collected responses: {len(responses)}",
        f"- briefs cached: {len(briefs)}",
    ]
    if by_model:
        lines.append("- by model:")
        for m, n in sorted(by_model.items()):
            lines.append(f"  - `{m}`: {n}")
    lines.extend(
        [
            "",
            "<details><summary><b>Teacher SYSTEM prompt</b> "
            "(identical for every ad — click to expand)</summary>",
            "",
            _md_fence(bundle.system_prompt),
            "",
            "</details>",
            "",
            "---",
            "",
        ]
    )

    for aid in ad_ids:
        ad = ads_by_id.get(aid)
        if ad is None:
            continue
        r = responses[aid]
        model = str(r.get("model") or "?")
        batch_id = str(r.get("batch_id") or "")
        content = str(r.get("content") or "")
        brief = briefs.get(aid)
        parsed = parse_response(content)

        lines.append(f"## ad `{aid}` · `{model}`")
        lines.append("")
        lines.append(f"_batch `{batch_id}` · custom_id `teacher-{aid}`_")
        lines.append("")
        lines.append("### Source")
        lines.append("")
        lines.append(_md_source_block(ad))
        lines.append("")

        caption = captions.get(aid)
        if caption:
            lines.append("**VLM caption** (supervision target for `<image_brief>`):")
            lines.append("")
            lines.append(_md_fence(caption))
            lines.append("")

        lines.append("### Brief")
        lines.append("")
        if brief:
            import json

            lines.append(_md_fence(json.dumps(brief, indent=2, ensure_ascii=False), lang="json"))
        else:
            lines.append("> _(brief missing)_")
        lines.append("")

        lines.append("### Think")
        lines.append("")
        if isinstance(parsed, ParsedResponse):
            lines.append(_md_blockquote(parsed.think))
        else:
            lines.append(f"> _(unparseable: {parsed.value})_")
        lines.append("")

        lines.append("### Deliverable")
        lines.append("")
        if isinstance(parsed, ParsedResponse):
            lines.append(_md_fence(parsed.deliverable))
        else:
            lines.append("> _(deliverable missing — see Think section)_")
        lines.append("")

        lines.append("---")
        lines.append("")

    if output:
        out_path = Path(output)
    else:
        scope = pipeline.runs_dir(config, run_id) if run_id else Path(config.output_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = scope / f"single_pass_review_{ts}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]wrote[/green] {out_path}  ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    app()
