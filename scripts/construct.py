"""Training data construction CLI.

Usage:
    python scripts/construct.py status                         # Progress overview
    python scripts/construct.py cluster                        # Pre-compute clusters
    python scripts/construct.py prepare positioning 10         # Prep next batch (chat mode)
    python scripts/construct.py ingest positioning             # Ingest a chat response
    python scripts/construct.py batch-submit positioning 50 \
        --model gpt-4o-mini                                    # Submit to API batch
    python scripts/construct.py batch-list                     # Show pending batches
    python scripts/construct.py batch-collect positioning      # Poll + ingest results
    python scripts/construct.py batch-cancel <batch_id>        # Cancel an in-flight job
    python scripts/construct.py validate positioning           # Validate latest examples
    python scripts/construct.py filter                         # Run quality filter
    python scripts/construct.py build                          # Assemble final HF Dataset
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from draper.construction.base_constructor import BaseConstructor

# ---------------------------------------------------------------------------
# Ensure project root is importable when run as a script
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from draper.construction.batch import (  # noqa: E402
    BatchRegistry,
    BatchRequest,
    BatchStatus,
    PendingBatch,
    PendingBatchSidecar,
    make_batch_client,
    provider_for_model,
)
from draper.construction.batch.registry import utc_now_iso  # noqa: E402
from draper.construction.bundle import build_bundle  # noqa: E402
from draper.construction.cluster_report import build_report, load_ads  # noqa: E402
from draper.construction.clusterer import AdClusterer  # noqa: E402
from draper.construction.dataset_builder import DatasetBuilder  # noqa: E402
from draper.construction.dice import prepare_bundles  # noqa: E402
from draper.construction.ingestion import (  # noqa: E402
    ingest_response,
    load_ads_by_id,
)
from draper.construction.personas import PersonaLibrary  # noqa: E402
from draper.construction.provider_rotation import (  # noqa: E402
    ProviderCapacity,
    classify_provider,
    format_provider_capacity,
    suggest_next_provider,
    tally_provider_counts,
)
from draper.construction.quality_filter import QualityFilter  # noqa: E402
from draper.construction.schemas import (  # noqa: E402
    ConstructionConfig,
    PromptStyle,
    TaskFormat,
    TrainingExample,
)
from draper.construction.source_selector import SourceSelector  # noqa: E402
from draper.utils.io import Checkpoint, read_jsonl  # noqa: E402
from draper.utils.llm_client import complete_with_usage  # noqa: E402
from draper.utils.logging import setup_logging  # noqa: E402

app = typer.Typer(help="Training data construction pipeline.")
console = Console()


def _load_config(config_path: str = "configs/construction.yaml") -> ConstructionConfig:
    return ConstructionConfig.from_yaml(config_path)


def _resolve_styles(
    style_arg: str,
    count: int,
    constructor: BaseConstructor,
) -> list[PromptStyle]:
    """Convert the --style CLI arg into a list of PromptStyle values.

    Valid values: ``natural``, ``data_grounded``, ``context_distilled``,
    or ``auto`` (use the configured 3-way ratio).
    """
    try:
        forced = PromptStyle(style_arg)
    except ValueError:
        # "auto" or any non-enum value → configured ratio
        return constructor.assign_styles(count)
    return [forced] * count


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@app.command()
def status(config_path: str = "configs/construction.yaml") -> None:
    """Show per-format progress and overall statistics."""
    cfg = _load_config(config_path)

    table = Table(title="Construction Progress")
    table.add_column("Format", style="cyan")
    table.add_column("Generated", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Remaining", justify="right")
    table.add_column("Progress", justify="right")

    total_gen = 0
    total_target = 0
    for fmt in TaskFormat:
        output_path = Path(cfg.output_dir) / fmt.value / "examples.jsonl"
        checkpoint = Checkpoint(output_path)
        generated = int(checkpoint.get("generated_count", 0))
        target = cfg.target_for(fmt)
        remaining = max(0, target - generated)
        pct = f"{generated / target * 100:.0f}%" if target > 0 else "n/a"
        table.add_row(fmt.value, str(generated), str(target), str(remaining), pct)
        total_gen += generated
        total_target += target

    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{total_gen}[/bold]",
        f"[bold]{total_target}[/bold]",
        f"[bold]{max(0, total_target - total_gen)}[/bold]",
        f"[bold]{total_gen / total_target * 100:.0f}%[/bold]" if total_target else "n/a",
    )
    console.print(table)

    # Cluster artifacts
    clusters_dir = Path(cfg.clusters_dir)
    if clusters_dir.exists():
        console.print("\n[bold]Cluster artifacts:[/bold]")
        for p in sorted(clusters_dir.glob("*.jsonl")):
            records = read_jsonl(p)
            console.print(f"  {p.name}: {len(records)} records")

    # Style distribution
    style_counts: dict[str, int] = {s.value: 0 for s in PromptStyle}
    provider_counts: dict[str, int] = {
        "claude": 0,
        "gpt": 0,
        "gemini": 0,
        "unknown": 0,
    }
    for fmt in TaskFormat:
        path = Path(cfg.output_dir) / fmt.value / "examples.jsonl"
        if not path.exists():
            continue
        for rec in read_jsonl(path):
            meta = rec.get("metadata", {})
            ps = meta.get("prompt_style", PromptStyle.DATA_GROUNDED.value)
            if ps in style_counts:
                style_counts[ps] += 1
            provider = classify_provider(meta.get("construction_model", ""))
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
    total_styles = sum(style_counts.values())
    if total_styles > 0:
        parts = [
            f"{v} {k} ({v / total_styles * 100:.0f}%)" for k, v in style_counts.items() if v > 0
        ]
        console.print("\n[bold]Prompt styles:[/bold] " + ", ".join(parts))

    total_providers = sum(provider_counts.values())
    if total_providers > 0:
        prov_targets = cfg.provider_rotation.targets()
        provider_parts: list[str] = []
        for prov in ("claude", "gpt", "gemini"):
            count = provider_counts.get(prov, 0)
            share = count / total_providers if total_providers else 0.0
            target_share: float = prov_targets.get(prov, 0.0)
            delta = share - target_share
            sign = "+" if delta >= 0 else ""
            provider_parts.append(
                f"{prov} {count} ({share * 100:.0f}% vs target "
                f"{target_share * 100:.0f}%, {sign}{delta * 100:.0f}pp)"
            )
        if provider_counts.get("unknown", 0):
            provider_parts.append(f"unknown {provider_counts['unknown']}")
        console.print("[bold]Provider mix:[/bold] " + ", ".join(provider_parts))

    # Cost tracking
    costs_path = Path(cfg.output_dir) / "construction_costs.json"
    if costs_path.exists():
        with costs_path.open() as f:
            records = json.load(f)
        console.print(f"\n[bold]API costs:[/bold] {len(records)} calls recorded")


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------


@app.command()
def cluster(config_path: str = "configs/construction.yaml") -> None:
    """Pre-compute ad clusters, pairs, and selections."""
    setup_logging()
    cfg = _load_config(config_path)

    console.print(f"Loading scored ads from {cfg.scored_ads_path}...")
    clusterer = AdClusterer.from_config(cfg)

    console.print(f"Computing clusters → {cfg.clusters_dir}")
    summary = clusterer.compute_and_save(cfg.clusters_dir)

    table = Table(title="Cluster Summary")
    table.add_column("Artifact", style="cyan")
    table.add_column("Count", justify="right")
    for name, count in summary.items():
        table.add_row(name, str(count))
    console.print(table)


# ---------------------------------------------------------------------------
# cluster-report (read-only capacity study)
# ---------------------------------------------------------------------------


@app.command(name="cluster-report")
def cluster_report_cmd(config_path: str = "configs/construction.yaml") -> None:
    """Simulate strict-pass capacity per format. Writes nothing to disk.

    Run before ``cluster`` to validate thresholds produce enough
    fingerprint-unique bundles for each format's raw target.
    """
    setup_logging()
    cfg = _load_config(config_path)

    console.print(f"Loading scored ads from {cfg.scored_ads_path}...")
    ads = load_ads(cfg.scored_ads_path)
    console.print(f"Loaded {len(ads)} scored ads.")

    raw_targets = {
        fmt_name: int(round(fmt.target * cfg.overgeneration_buffer))
        for fmt_name, fmt in cfg.formats.items()
    }
    report = build_report(ads, cfg, raw_targets=raw_targets or None)

    table = Table(title="Strict-Pass Capacity")
    table.add_column("Format", style="cyan")
    table.add_column("Ad pool", justify="right")
    table.add_column("Bundles", justify="right")
    table.add_column("Raw target", justify="right")
    table.add_column("% of target", justify="right")
    table.add_column("Notes", style="dim")
    for fc in report.formats:
        pct = f"{fc.pct_of_target:.0f}%"
        table.add_row(
            fc.format_name,
            str(fc.unique_ad_pool),
            str(fc.bundles_available),
            str(fc.raw_target),
            pct,
            ", ".join(fc.notes),
        )
    console.print(table)

    if report.cross_format_overlap:
        console.print("\n[bold]Cross-format ad overlap[/bold]")
        overlap_table = Table()
        overlap_table.add_column("Format A", style="cyan")
        overlap_table.add_column("Format B", style="cyan")
        overlap_table.add_column("Shared ads", justify="right")
        for (a, b), count in sorted(
            report.cross_format_overlap.items(), key=lambda kv: -kv[1]
        ):
            if count == 0:
                continue
            overlap_table.add_row(a, b, str(count))
        console.print(overlap_table)

    for fc in report.formats:
        if not fc.platforms:
            continue
        total = sum(fc.platforms.values())
        parts = [
            f"{plat} {cnt} ({cnt / total * 100:.0f}%)"
            for plat, cnt in fc.platforms.most_common()
        ]
        console.print(f"[bold]{fc.format_name}[/bold] platforms: " + ", ".join(parts))


# ---------------------------------------------------------------------------
# prepare (chat-client mode)
# ---------------------------------------------------------------------------


@app.command()
def prepare(
    format_name: str,
    batch_size: int = typer.Argument(10, help="Number of bundles to prepare"),
    style: str = typer.Option(
        "auto",
        help=(
            "Prompt style: natural, data_grounded, context_distilled, or auto "
            "(configured 3-way ratio, default 40/40/20)"
        ),
    ),
    provider: str = typer.Option(
        "auto",
        help=(
            "Declared teacher provider for this batch: claude, gpt, gemini, "
            "or auto (suggest based on current provider distribution)"
        ),
    ),
    personas_path: str = "configs/personas.yaml",
    config_path: str = "configs/construction.yaml",
) -> None:
    """Prepare the next batch of training-example bundles for chat-client mode.

    Rolls dice (style, persona, seed, evol-instruct), selects unused source
    ads via ``SourceSelector``, and assembles a self-contained bundle per
    example. Each bundle is printed ready to paste into one chat session;
    the chat agent returns structured ``<user_prompt>`` and
    ``<assistant_response>`` tags.

    Writes a ``_last_prepared.json`` sidecar with the full rolled-dice
    metadata so ``ingest`` can save each example with correct provenance.
    """
    setup_logging()
    cfg = _load_config(config_path)
    task_format = TaskFormat(format_name)

    constructor = _get_constructor(task_format, cfg)
    consumed = constructor.consumed_ad_ids() | constructor.rejected_ad_ids()

    selector = SourceSelector(config=cfg)
    batches = selector.select_batches(task_format, consumed, batch_size)

    if not batches:
        console.print("[red]No source ads available for this format.[/red]")
        return

    # Resolve provider: "auto" → pick the underrepresented provider.
    if provider == "auto":
        counts = tally_provider_counts(cfg)
        provider = suggest_next_provider(counts, cfg.provider_rotation)
        console.print(f"[dim]Provider auto-selected: {provider} (current counts: {counts})[/dim]")
    elif provider not in ("claude", "gpt", "gemini"):
        console.print(
            f"[red]Invalid --provider '{provider}'. Use claude, gpt, gemini, or auto.[/red]"
        )
        return

    styles = _resolve_styles(style, len(batches), constructor)
    personas = PersonaLibrary.from_yaml(personas_path)

    # Header
    gen = constructor.generated_count
    tgt = constructor.target
    console.print(
        f"\n[bold]=== Format: {format_name} "
        f"({gen}/{tgt} done, {max(0, tgt - gen)} remaining) "
        f"— provider: {provider} ===[/bold]"
    )
    console.print(
        f"\nAfter pasting responses into {provider}, run:  "
        f"[cyan]python scripts/construct.py ingest {format_name} "
        f"--prompt-index N[/cyan]"
    )

    prepared = prepare_bundles(
        cfg=cfg,
        constructor=constructor,
        personas=personas,
        batches=batches,
        styles=styles,
        provider_label=provider,
    )

    for pb in prepared:
        ctx = pb.context
        bundle = build_bundle(ctx)
        console.print(
            f"\n[bold]=== Bundle {pb.prompt_index + 1} of {len(prepared)} "
            f"[{ctx.style.value.upper()}, persona={ctx.persona.id}, "
            f"seed={ctx.seed_idx}, evol={ctx.evol_op or 'none'}, "
            f"difficulty={ctx.difficulty}, turns={ctx.turn_structure}"
            f"{'/' + ctx.followup_type if ctx.followup_type else ''}] ==="
            f"[/bold]"
        )
        console.print(bundle)

    sidecar_data = [pb.sidecar for pb in prepared]
    sidecar_path = Path(cfg.output_dir) / format_name / "_last_prepared.json"
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with sidecar_path.open("w") as f:
        json.dump(sidecar_data, f, indent=2)
    console.print(f"\n[dim]Saved {len(prepared)} prompt source mappings to {sidecar_path}[/dim]")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@app.command()
def validate(
    format_name: str,
    config_path: str = "configs/construction.yaml",
) -> None:
    """Validate examples for a format against the TrainingExample schema."""
    cfg = _load_config(config_path)
    task_format = TaskFormat(format_name)
    path = Path(cfg.output_dir) / task_format.value / "examples.jsonl"

    if not path.exists():
        console.print(f"[red]No examples found at {path}[/red]")
        return

    records = read_jsonl(path)
    valid = 0
    invalid = 0
    for i, rec in enumerate(records):
        try:
            TrainingExample(**rec)
            valid += 1
        except Exception as e:
            invalid += 1
            console.print(f"[red]Example {i}: {e}[/red]")

    total = len(records)
    console.print(f"\n[bold]{valid}[/bold] valid, [bold]{invalid}[/bold] invalid out of {total}")


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@app.command()
def ingest(
    format_name: str,
    prompt_index: int = typer.Option(0, help="Which bundle from the last prepare batch"),
    file: str = typer.Option("", help="Read response from file instead of stdin"),
    config_path: str = "configs/construction.yaml",
) -> None:
    """Ingest a chat-agent response to a prepared bundle.

    Parses the required tags (``<user_prompt>``, ``<assistant_response>``),
    looks up the dice-roll metadata from the ``_last_prepared.json``
    sidecar, and saves a ``TrainingExample`` with full provenance
    (persona, seed, evol operator, style, difficulty, declared provider).
    """
    setup_logging()
    cfg = _load_config(config_path)
    task_format = TaskFormat(format_name)

    # Load sidecar
    sidecar_path = Path(cfg.output_dir) / format_name / "_last_prepared.json"
    if not sidecar_path.exists():
        console.print(
            f"[red]No _last_prepared.json found at {sidecar_path}. "
            f"Run 'prepare {format_name}' first.[/red]"
        )
        return

    with sidecar_path.open() as f:
        sidecar_data: list[dict[str, object]] = json.load(f)

    if prompt_index >= len(sidecar_data):
        console.print(
            f"[red]prompt_index {prompt_index} out of range "
            f"(last prepare had {len(sidecar_data)} bundles, 0-indexed).[/red]"
        )
        return

    sidecar = sidecar_data[prompt_index]

    # Read response
    if file:
        response_text = Path(file).read_text(encoding="utf-8")
    else:
        console.print(
            "[bold]Paste the chat response below, then press "
            "Ctrl-D (Unix) or Ctrl-Z+Enter (Windows) to finish:[/bold]"
        )
        response_text = sys.stdin.read()

    if not response_text.strip():
        console.print("[red]Empty response, nothing to ingest.[/red]")
        return

    constructor = _get_constructor(task_format, cfg)
    ads_by_id = load_ads_by_id(cfg.scored_ads_path)
    result = ingest_response(
        response_text=response_text,
        sidecar=sidecar,
        constructor=constructor,
        ads_by_id=ads_by_id,
    )

    if result.error:
        console.print(f"[red]{result.error}[/red]")
        return

    total = constructor.generated_count
    target = constructor.target
    console.print(
        f"[green]Saved {result.saved} example(s) "
        f"[{sidecar.get('prompt_style', '')}, "
        f"persona={sidecar.get('persona_id', '')}, "
        f"seed={sidecar.get('seed_idx', -1)}, "
        f"evol={sidecar.get('evol_op') or 'none'}] "
        f"(total: {total}/{target})[/green]"
    )


# ---------------------------------------------------------------------------
# pilot (real-time API, small batch — for tight iteration)
# ---------------------------------------------------------------------------


# Default teacher models per provider label, shared between `pilot`
# (real-time API) and `batch-submit` (Batch API) so both modes target
# the same generation. Tuned for cost/quality on the copywriting
# backtranslation task — Pro/Sonnet/5.5-pro are overkill here. Override
# with the per-provider flags if a newer ID ships.
_DEFAULT_TEACHER_MODELS: dict[str, str] = {
    "claude": "claude-haiku-4-5-20251001",
    "gpt": "gpt-5.5",
    "gemini": "gemini-3-flash-preview",  # native google-genai SDK
}
_PILOT_DEFAULT_MODELS = _DEFAULT_TEACHER_MODELS

_PROVIDER_LABEL_FOR_MODEL: dict[str, str] = {
    "openai": "gpt",
    "anthropic": "claude",
    "gemini": "gemini",
}


async def _pilot_call_one(
    bundle_text: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, str, str]:
    """Call one teacher API and return (response, model_id, error)."""
    try:
        result = await complete_with_usage(
            messages=[{"role": "user", "content": bundle_text}],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=None,
        )
    except Exception as exc:  # noqa: BLE001 — surface all provider errors
        return "", model, str(exc)
    return result.content, result.model or model, ""


@app.command()
def pilot(
    format_name: str,
    per_provider: int = typer.Option(
        5,
        help="Bundles per provider. 5 × 3 providers = 15 examples total.",
    ),
    providers: str = typer.Option(
        "claude,gpt,gemini",
        help="Comma-separated provider labels (claude / gpt / gemini).",
    ),
    claude_model: str = typer.Option(
        _PILOT_DEFAULT_MODELS["claude"],
        help="Override the Claude model used for the pilot.",
    ),
    gpt_model: str = typer.Option(
        _PILOT_DEFAULT_MODELS["gpt"],
        help="Override the OpenAI model used for the pilot.",
    ),
    gemini_model: str = typer.Option(
        _PILOT_DEFAULT_MODELS["gemini"],
        help="Override the Gemini model used for the pilot (OpenRouter path).",
    ),
    style: str = typer.Option(
        "auto",
        help="Prompt style (auto = configured ratio).",
    ),
    max_tokens: int = typer.Option(4096, help="Per-call completion token budget."),
    temperature: float = typer.Option(0.7, help="Sampling temperature."),
    personas_path: str = "configs/personas.yaml",
    config_path: str = "configs/construction.yaml",
) -> None:
    """Run a small real-time (non-batch) API pilot.

    Rolls ``per_provider`` bundles for each provider in ``providers``,
    fires each bundle at the provider's sync API in parallel via
    ``complete_with_usage``, and ingests the responses immediately. This
    is the tight-iteration counterpart to ``batch-submit`` — no 24h wait,
    no registry bookkeeping. Use it to dial in prompt changes on a handful
    of examples before committing to a full batch.
    """
    setup_logging()
    cfg = _load_config(config_path)
    task_format = TaskFormat(format_name)

    provider_list = [p.strip().lower() for p in providers.split(",") if p.strip()]
    for prov in provider_list:
        if prov not in _PILOT_DEFAULT_MODELS:
            console.print(f"[red]Unknown provider '{prov}'. Use claude/gpt/gemini.[/red]")
            raise typer.Exit(code=1)

    model_for: dict[str, str] = {
        "claude": claude_model,
        "gpt": gpt_model,
        "gemini": gemini_model,
    }

    constructor = _get_constructor(task_format, cfg)
    personas = PersonaLibrary.from_yaml(personas_path)

    # One shared consumed-IDs pool across providers so bundles don't
    # duplicate ads across the pilot run. The selector honours this.
    consumed = constructor.consumed_ad_ids() | constructor.rejected_ad_ids()
    consumed_fingerprints = frozenset(constructor.consumed_bundle_fingerprints())
    selector = SourceSelector(config=cfg)

    total_needed = per_provider * len(provider_list)
    all_batches = selector.select_batches(
        task_format, consumed, total_needed, consumed_fingerprints=consumed_fingerprints
    )
    if len(all_batches) < total_needed:
        console.print(
            f"[yellow]Warning: pool yielded {len(all_batches)}/{total_needed} "
            f"bundles — providers will get fewer examples each.[/yellow]"
        )

    # Partition bundles into per-provider slices.
    per_slices: dict[str, list[list]] = {}
    cursor = 0
    for prov in provider_list:
        slice_size = min(per_provider, max(0, len(all_batches) - cursor))
        per_slices[prov] = all_batches[cursor : cursor + slice_size]
        cursor += slice_size

    ads_by_id = load_ads_by_id(cfg.scored_ads_path)

    grand_total_saved = 0
    grand_total_failed = 0

    for prov in provider_list:
        batches = per_slices[prov]
        if not batches:
            continue
        model = model_for[prov]
        styles = _resolve_styles(style, len(batches), constructor)

        prepared = prepare_bundles(
            cfg=cfg,
            constructor=constructor,
            personas=personas,
            batches=batches,
            styles=styles,
            provider_label=prov,
        )

        bundle_texts = [build_bundle(pb.context) for pb in prepared]
        console.print(
            f"\n[bold]→ {prov} ({model}): calling {len(bundle_texts)} bundles...[/bold]"
        )

        async def _run_all(texts: list[str], model_id: str) -> list[tuple[str, str, str]]:
            return await asyncio.gather(
                *[_pilot_call_one(t, model_id, temperature, max_tokens) for t in texts]
            )

        results = asyncio.run(_run_all(bundle_texts, model))

        saved = 0
        failed = 0
        for pb, bundle_text, (content, reported_model, err) in zip(
            prepared, bundle_texts, results, strict=False
        ):
            if err or not content:
                console.print(
                    f"[yellow]  ⚠ bundle {pb.prompt_index} failed: "
                    f"{err or 'empty response'}[/yellow]"
                )
                failed += 1
                continue
            result = ingest_response(
                response_text=content,
                sidecar=pb.sidecar,
                constructor=constructor,
                ads_by_id=ads_by_id,
                construction_model_override=reported_model or model,
                teacher_bundle=bundle_text,
            )
            if result.verbatim_failed:
                console.print(
                    f"[yellow]  ⚠ bundle {pb.prompt_index} fidelity-rejected: "
                    f"{result.error}[/yellow]"
                )
                failed += 1
                continue
            if result.error:
                console.print(
                    f"[yellow]  ⚠ bundle {pb.prompt_index} ingest error: {result.error}[/yellow]"
                )
                failed += 1
                continue
            saved += result.saved

        console.print(
            f"[green]  {prov}: {saved} saved, {failed} failed[/green]"
        )
        grand_total_saved += saved
        grand_total_failed += failed

    console.print(
        f"\n[bold]Pilot complete: {grand_total_saved} saved, "
        f"{grand_total_failed} failed across {len(provider_list)} providers.[/bold]"
    )
    console.print(
        "\nReview:  [cyan]python scripts/assesment/review_examples.py "
        f"{format_name}[/cyan]"
    )


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# batch-submit / batch-list / batch-collect / batch-cancel (API mode)
# ---------------------------------------------------------------------------


def _custom_id(task_format: str, prompt_index: int) -> str:
    """Stable per-request identifier used by both provider and sidecar."""
    return f"{task_format}-{prompt_index:05d}"


def _sidecar_from_prepared(
    prepared_sidecar: dict[str, object],
    custom_id: str,
    teacher_bundle: str = "",
) -> PendingBatchSidecar:
    """Translate a dice-roll sidecar dict into the registry dataclass."""
    raw_ids = prepared_sidecar.get("source_ad_ids", []) or []
    source_ad_ids = [str(x) for x in raw_ids]  # type: ignore[union-attr]
    return PendingBatchSidecar(
        custom_id=custom_id,
        prompt_index=int(prepared_sidecar.get("prompt_index", 0) or 0),  # type: ignore[arg-type]
        source_ad_ids=source_ad_ids,
        prompt_style=str(prepared_sidecar.get("prompt_style", "data_grounded")),
        persona_id=str(prepared_sidecar.get("persona_id", "")),
        seed_idx=int(prepared_sidecar.get("seed_idx", -1) or -1),  # type: ignore[arg-type]
        evol_op=str(prepared_sidecar.get("evol_op", "")),
        difficulty=str(prepared_sidecar.get("difficulty", "standard")),
        turn_structure=str(prepared_sidecar.get("turn_structure", "single")),
        followup_type=str(prepared_sidecar.get("followup_type", "")),
        provider_label=str(prepared_sidecar.get("provider", "")),
        source_ad_shape=str(prepared_sidecar.get("source_ad_shape", "")),
        conversation_register=str(
            prepared_sidecar.get("conversation_register")
            or prepared_sidecar.get("brief_register", "")
        ),
        teacher_bundle=teacher_bundle,
    )


def _print_capacity_table(
    task_format: TaskFormat,
    capacity: dict[str, ProviderCapacity],
    requesting: tuple[str, int] | None = None,
) -> None:
    """Render per-provider remaining-bundle capacity for a format."""
    req_provider, req_size = requesting or ("", 0)
    table = Table(title=f"Provider capacity — {task_format.value}")
    table.add_column("Provider", style="cyan")
    table.add_column("Used", justify="right")
    table.add_column("Reserved", justify="right")
    table.add_column("Cap", justify="right")
    table.add_column("Remaining", justify="right")
    table.add_column("Target share", justify="right")
    if requesting:
        table.add_column("After request", justify="right")
    for cap in capacity.values():
        row = [
            cap.provider,
            str(cap.used),
            str(cap.reserved),
            str(cap.cap),
            str(cap.remaining),
            f"{cap.share_pct:.0f}%",
        ]
        if requesting:
            if cap.provider == req_provider:
                projected = cap.projected + req_size
                style = "[red]" if projected > cap.cap else "[green]"
                row.append(f"{style}{projected}/{cap.cap}[/]")
            else:
                row.append("—")
        table.add_row(*row)
    console.print(table)


@app.command(name="provider-capacity")
def provider_capacity(
    format_name: str = typer.Argument(
        "",
        help="Show one format. Default: every format.",
    ),
    config_path: str = "configs/construction.yaml",
) -> None:
    """Show how many bundles each provider has left in each format.

    Cap = ``raw_target * provider_ratio`` from
    ``configs/construction.yaml`` (raw target accounts for the quality
    filter's expected dropout). ``batch-submit`` refuses requests that
    would push a provider past its cap.
    """
    cfg = _load_config(config_path)
    formats = [TaskFormat(format_name)] if format_name else list(TaskFormat)
    for fmt in formats:
        cap = format_provider_capacity(cfg, fmt)
        _print_capacity_table(fmt, cap)
        console.print()


@app.command(name="batch-submit")
def batch_submit(
    format_name: str,
    batch_size: int = typer.Argument(50, help="Number of examples in this batch"),
    model: str = typer.Option(
        _DEFAULT_TEACHER_MODELS["claude"],
        help=(
            "Teacher model for this batch. Provider is auto-detected from "
            "the model name: gpt-*/o1-*/o3-* → OpenAI Batch API; "
            "claude-* → Anthropic Message Batches; "
            "gemini-* → Gemini Batch API (inline mode)."
        ),
    ),
    style: str = typer.Option(
        "auto",
        help=(
            "Prompt style: natural / data_grounded / context_distilled / auto "
            "(uses configured 3-way ratio)."
        ),
    ),
    max_tokens: int = typer.Option(4096, help="Per-request completion token budget."),
    temperature: float = typer.Option(0.7, help="Sampling temperature."),
    personas_path: str = "configs/personas.yaml",
    config_path: str = "configs/construction.yaml",
    allow_overflow: bool = typer.Option(
        False,
        "--allow-overflow",
        help=(
            "Allow this submission to push the chosen provider past its "
            "configured share for the format. Default: refuse."
        ),
    ),
) -> None:
    """Roll dice, build bundles, and submit to the provider's Batch API.

    This is the API-mode twin of ``prepare`` — the heavy lifting happens
    asynchronously at the provider (24h SLA, ~50% cheaper than sync). The
    ``_pending_batches.json`` registry remembers the dice rolls so
    ``batch-collect`` can reconstruct full provenance once results land.
    """
    setup_logging()
    cfg = _load_config(config_path)
    task_format = TaskFormat(format_name)

    try:
        provider = provider_for_model(model)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    # Provider quota check: refuse if this batch would push the chosen
    # provider past its share of the format's raw target.
    provider_label = {"openai": "gpt", "anthropic": "claude", "gemini": "gemini"}[provider]
    capacity = format_provider_capacity(cfg, task_format)
    cap_for_provider = capacity[provider_label]
    if cap_for_provider.remaining < batch_size and not allow_overflow:
        _print_capacity_table(task_format, capacity, requesting=(provider_label, batch_size))
        suggestions = [
            f"{p.provider} ({p.remaining} left)"
            for p in capacity.values()
            if p.remaining >= batch_size
        ]
        suggest_msg = (
            f"Try one of: {', '.join(suggestions)}."
            if suggestions
            else "All providers are at or near their cap for this format."
        )
        console.print(
            f"\n[red]Refusing submission: {provider_label} has only "
            f"{cap_for_provider.remaining} bundle(s) left for "
            f"{task_format.value} (cap {cap_for_provider.cap} = "
            f"{cap_for_provider.share_pct:.0f}% of raw target "
            f"{cfg.raw_target_for(task_format)}), but you asked for "
            f"{batch_size}.[/red]\n{suggest_msg}\n"
            "Pass [cyan]--allow-overflow[/cyan] to override."
        )
        raise typer.Exit(code=1)

    _print_capacity_table(task_format, capacity, requesting=(provider_label, batch_size))

    constructor = _get_constructor(task_format, cfg)

    # Pull reserved ad IDs and pending request count from the registry so
    # this submission doesn't overlap with any in-flight batches.
    registry = BatchRegistry(cfg.output_dir, task_format.value)
    reserved_ids = registry.reserved_ad_ids()
    pending_count = registry.pending_request_count()
    if reserved_ids:
        console.print(
            f"[dim]Excluding {len(reserved_ids)} ad IDs reserved in "
            f"{len(registry._active())} active batch(es).[/dim]"
        )

    # Also exclude ad IDs in _last_prepared.json (manual chat-mode bundles
    # that have been prepared but not yet ingested — invisible to both the
    # JSONL and the batch registry).
    prepared_ids: set[str] = set()
    prepared_fingerprints: set[frozenset[str]] = set()
    last_prepared_path = Path(cfg.output_dir) / task_format.value / "_last_prepared.json"
    if last_prepared_path.exists():
        with last_prepared_path.open() as _f:
            _prepared = json.load(_f)
        for _bundle in _prepared:
            ids = [str(x) for x in _bundle.get("source_ad_ids", [])]
            prepared_ids.update(ids)
            if ids:
                prepared_fingerprints.add(frozenset(ids))
        if prepared_ids:
            console.print(
                f"[dim]Excluding {len(prepared_ids)} ad IDs from "
                f"_last_prepared.json (pending manual ingest).[/dim]"
            )

    rejected_ids = constructor.rejected_ad_ids()
    if rejected_ids:
        console.print(
            f"[dim]Excluding {len(rejected_ids)} ad IDs from "
            f"_rejected_ads.jsonl (prior ingestion-check failures).[/dim]"
        )
    consumed = (
        constructor.consumed_ad_ids() | reserved_ids | prepared_ids | rejected_ids
    )
    # Bundle-level fingerprints from in-flight batches + existing examples +
    # pending chat-mode bundles. The selector uses these to guarantee no new
    # bundle duplicates an ad-set already reserved or written to disk.
    consumed_fingerprints = frozenset(
        registry.reserved_bundle_fingerprints()
        | constructor.consumed_bundle_fingerprints()
        | prepared_fingerprints
    )
    selector = SourceSelector(config=cfg)
    batches = selector.select_batches(
        task_format,
        consumed,
        batch_size,
        consumed_fingerprints=consumed_fingerprints,
    )
    if not batches:
        console.print("[red]No source ads available for this format.[/red]")
        return
    if len(batches) < batch_size and not allow_overflow:
        console.print(
            f"\n[red]Refusing partial submission: source pool yielded only "
            f"{len(batches)}/{batch_size} bundles for {task_format.value}.[/red]\n"
            f"[dim]Pool exhausted after excluding {len(consumed)} consumed/reserved ad IDs. "
            "Either re-cluster with looser criteria, reduce batch size, or "
            "pass [cyan]--allow-overflow[/cyan] to submit the partial batch.[/dim]"
        )
        raise typer.Exit(code=1)

    styles = _resolve_styles(style, len(batches), constructor)
    personas = PersonaLibrary.from_yaml(personas_path)

    prepared = prepare_bundles(
        cfg=cfg,
        constructor=constructor,
        personas=personas,
        batches=batches,
        styles=styles,
        provider_label=provider_label,
        rng_offset=pending_count,
    )

    # Render one BatchRequest per bundle.
    requests: list[BatchRequest] = []
    sidecars: list[PendingBatchSidecar] = []
    for pb in prepared:
        bundle_text = build_bundle(pb.context)
        custom_id = _custom_id(task_format.value, pb.prompt_index)
        requests.append(
            BatchRequest(
                custom_id=custom_id,
                system=None,  # bundle is fully self-contained
                messages=[{"role": "user", "content": bundle_text}],
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
        sidecars.append(_sidecar_from_prepared(pb.sidecar, custom_id, bundle_text))

    client = make_batch_client(model)
    console.print(f"[bold]Submitting {len(requests)} bundles to {provider} ({model})...[/bold]")

    info = asyncio.run(client.submit(requests))

    # Persist the batch in the registry so batch-collect can find it.
    registry.add(
        PendingBatch(
            batch_id=info.batch_id,
            provider=provider,
            model=model,
            task_format=task_format.value,
            submitted_at=utc_now_iso(),
            status=info.status.value,
            request_count=info.request_count or len(requests),
            sidecars=sidecars,
        )
    )

    console.print(
        f"[green]Submitted batch [bold]{info.batch_id}[/bold] "
        f"({info.request_count} requests, status={info.status.value})[/green]"
    )
    console.print(
        "Poll + ingest once complete:  "
        f"[cyan]python scripts/construct.py batch-collect {format_name}[/cyan]"
    )


@app.command(name="batch-list")
def batch_list(config_path: str = "configs/construction.yaml") -> None:
    """Show all pending/recent batches across every task format."""
    cfg = _load_config(config_path)

    table = Table(title="Pending Batches")
    table.add_column("Format", style="cyan")
    table.add_column("Batch ID")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Reqs", justify="right")
    table.add_column("Done", justify="right")
    table.add_column("Failed", justify="right")
    table.add_column("Submitted")

    any_rows = False
    for fmt in TaskFormat:
        registry = BatchRegistry(cfg.output_dir, fmt.value)
        # Live-poll every active batch and collect raw provider status for display.
        live_raw: dict[str, str] = {}  # batch_id → raw provider status string
        for b in registry._active():
            try:
                client = make_batch_client(b.model)
                info = asyncio.run(client.poll(b.batch_id))
                registry.update_status(
                    b.batch_id,
                    status=info.status.value,
                    completed_count=info.completed_count,
                    # Only update request_count when the provider reports a
                    # non-zero value — Gemini returns 0 while running because
                    # completion_stats is null mid-job.
                    failed_count=info.failed_count,
                    request_count=info.request_count if info.request_count > 0 else None,
                )
                # Capture raw provider status (e.g. "finalizing" vs "in_progress")
                raw_status = str(
                    info.raw.get("openai_status")
                    or info.raw.get("gemini_state")
                    or info.raw.get("anthropic_state")
                    or ""
                )
                # Strip verbose Gemini prefix (JOB_STATE_RUNNING → running)
                if raw_status.startswith("JOB_STATE_"):
                    raw_status = raw_status[len("JOB_STATE_"):].lower()
                live_raw[b.batch_id] = raw_status
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning: could not poll {b.batch_id}: {exc}[/yellow]")
        for b in registry.all():
            any_rows = True
            normalized = b.status
            raw = live_raw.get(b.batch_id, "")
            # Show raw provider status in parens when it adds info
            # (e.g. "in_progress (finalizing)" tells you it's packaging, not computing)
            display_status = (
                f"{normalized} ({raw})" if raw and raw != normalized else normalized
            )
            table.add_row(
                fmt.value,
                b.batch_id,
                b.provider,
                b.model,
                display_status,
                str(b.request_count),
                str(b.completed_count),
                str(b.failed_count),
                b.submitted_at,
            )

    if not any_rows:
        console.print("[dim]No pending batches. Submit one with 'batch-submit'.[/dim]")
        return
    console.print(table)


@app.command(name="batch-collect")
def batch_collect(
    format_name: str,
    batch_id: str = typer.Option(
        "",
        help="Specific batch to collect. Default: every pending batch for this format.",
    ),
    config_path: str = "configs/construction.yaml",
) -> None:
    """Poll pending batches for a format and ingest any completed results.

    Completed batches are cleared from the registry after successful
    ingest; in-progress batches have their status updated in place so
    ``batch-list`` stays current.
    """
    setup_logging()
    cfg = _load_config(config_path)
    task_format = TaskFormat(format_name)

    registry = BatchRegistry(cfg.output_dir, task_format.value)
    # Collect pending/in-progress batches AND any that batch-list already
    # polled to 'completed' before ingest ran (never skip completed).
    _skip = {BatchStatus.FAILED.value, BatchStatus.CANCELLED.value, BatchStatus.EXPIRED.value}
    targets = [b for b in registry.all() if b.status not in _skip]
    if batch_id:
        targets = [b for b in targets if b.batch_id == batch_id]
    if not targets:
        console.print("[dim]No pending batches to collect.[/dim]")
        return

    constructor = _get_constructor(task_format, cfg)
    ads_by_id = load_ads_by_id(cfg.scored_ads_path)

    for pending in targets:
        console.print(
            f"\n[bold]Checking {pending.batch_id} ({pending.provider}/{pending.model})...[/bold]"
        )
        client = make_batch_client(pending.model)
        info = asyncio.run(client.poll(pending.batch_id))

        registry.update_status(
            pending.batch_id,
            status=info.status.value,
            completed_count=info.completed_count,
            failed_count=info.failed_count,
        )

        if not info.is_terminal:
            console.print(
                f"[dim]  Status: {info.status.value} "
                f"({info.completed_count}/{info.request_count} done). "
                f"Try again later.[/dim]"
            )
            continue

        if info.status != BatchStatus.COMPLETED:
            console.print(
                f"[red]  Batch ended in non-success state: {info.status.value} "
                f"({info.error}). Leaving in registry for inspection.[/red]"
            )
            continue

        responses = asyncio.run(client.fetch_results(pending.batch_id))
        saved_total = 0
        parse_failures = 0
        verbatim_failures = 0

        for resp in responses:
            sidecar = pending.sidecar_by_custom_id(resp.custom_id)
            if sidecar is None:
                console.print(
                    f"[yellow]  ⚠ No sidecar for custom_id {resp.custom_id}; skipping.[/yellow]"
                )
                parse_failures += 1
                continue
            if resp.error:
                console.print(
                    f"[yellow]  ⚠ {resp.custom_id} failed upstream: {resp.error}[/yellow]"
                )
                parse_failures += 1
                continue
            result = ingest_response(
                response_text=resp.content,
                sidecar=sidecar.__dict__,
                constructor=constructor,
                ads_by_id=ads_by_id,
                construction_model_override=resp.model or pending.model,
                teacher_bundle=sidecar.teacher_bundle,
                batch_id=pending.batch_id,
            )
            if result.verbatim_failed:
                console.print(
                    f"[yellow]  ⚠ {resp.custom_id} fidelity-rejected: "
                    f"{result.error}[/yellow]"
                )
                verbatim_failures += 1
                continue
            if result.error:
                parse_failures += 1
                continue
            saved_total += result.saved

        console.print(
            f"[green]  Saved {saved_total} examples "
            f"({parse_failures} failed, "
            f"{verbatim_failures} verbatim-rejected).[/green]"
        )
        registry.remove(pending.batch_id)


@app.command(name="batch-cancel")
def batch_cancel(
    batch_id: str,
    format_name: str = typer.Option(
        "",
        help="Task format the batch belongs to (for registry bookkeeping). "
        "Falls back to scanning all formats.",
    ),
    config_path: str = "configs/construction.yaml",
) -> None:
    """Attempt to cancel an in-flight batch and update the registry."""
    setup_logging()
    cfg = _load_config(config_path)

    formats = [TaskFormat(format_name)] if format_name else list(TaskFormat)
    for fmt in formats:
        registry = BatchRegistry(cfg.output_dir, fmt.value)
        pending = registry.get(batch_id)
        if pending is None:
            continue
        client = make_batch_client(pending.model)
        info = asyncio.run(client.cancel(batch_id))
        # Remove from registry immediately — cancelled batches free their
        # reserved ad IDs so the next submission can reuse them.
        registry.remove(batch_id)
        console.print(
            f"[green]Cancelled {batch_id}; removed from registry "
            f"(status={info.status.value}).[/green]"
        )
        return

    console.print(f"[red]No registry entry found for batch {batch_id}.[/red]")


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


@app.command(name="filter")
def filter_cmd(config_path: str = "configs/construction.yaml") -> None:
    """Run quality filter on all constructed examples."""
    setup_logging()
    cfg = _load_config(config_path)

    # Load all examples
    all_examples: list[TrainingExample] = []
    for fmt in TaskFormat:
        path = Path(cfg.output_dir) / fmt.value / "examples.jsonl"
        if not path.exists():
            continue
        records = read_jsonl(path)
        all_examples.extend(TrainingExample(**r) for r in records)

    if not all_examples:
        console.print("[red]No examples found to filter[/red]")
        return

    console.print(f"Filtering {len(all_examples)} examples...")
    qf = QualityFilter(config=cfg.quality_filter)
    result = qf.filter_all(all_examples)

    # Print stats
    s = result.stats
    table = Table(title="Quality Filter Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Total input", str(s.total_input))
    table.add_row("Passed", f"[green]{s.passed}[/green]")
    table.add_row("Artifacts repaired", f"[yellow]{s.artifacts_repaired}[/yellow]")
    table.add_row("Rejected (artifact leak)", str(s.rejected_artifact_leak))
    table.add_row("Rejected (structural)", str(s.rejected_structural))
    table.add_row("Rejected (min length)", str(s.rejected_min_length))
    table.add_row("Rejected (language)", str(s.rejected_language))
    table.add_row("Rejected (rubric)", str(s.rejected_rubric))
    table.add_row("Rejected (format-specific)", str(s.rejected_format_specific))
    table.add_row("Rejected (duplicate)", str(s.rejected_duplicate))
    table.add_row("Rejected (prompt-dup)", str(s.rejected_prompt_duplicate))
    table.add_row("Rejected (src-ad-dup)", str(s.rejected_source_ad_duplicate))
    pct = s.passed / s.total_input * 100 if s.total_input else 0
    table.add_row("Pass rate", f"{pct:.1f}%")
    console.print(table)

    # Save filtered examples back (overwrite per-format files)
    from draper.utils.io import write_jsonl

    by_format: dict[str, list[TrainingExample]] = {}
    for ex in result.passed:
        by_format.setdefault(ex.task_format.value, []).append(ex)

    for fmt_name, examples in by_format.items():
        out_path = Path(cfg.output_dir) / fmt_name / "filtered.jsonl"
        write_jsonl(examples, out_path)
        console.print(f"  {fmt_name}: {len(examples)} → {out_path}")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


@app.command()
def build(
    use_filtered: bool = True,
    config_path: str = "configs/construction.yaml",
) -> None:
    """Assemble final HuggingFace Dataset from constructed examples."""
    setup_logging()
    cfg = _load_config(config_path)

    # Load examples
    all_examples: list[TrainingExample] = []
    for fmt in TaskFormat:
        filename = "filtered.jsonl" if use_filtered else "examples.jsonl"
        path = Path(cfg.output_dir) / fmt.value / filename
        if not path.exists():
            # Fall back to unfiltered if filtered doesn't exist
            path = Path(cfg.output_dir) / fmt.value / "examples.jsonl"
            if not path.exists():
                continue
        records = read_jsonl(path)
        all_examples.extend(TrainingExample(**r) for r in records)

    if not all_examples:
        console.print("[red]No examples found to build dataset[/red]")
        return

    console.print(f"Building dataset from {len(all_examples)} examples...")
    builder = DatasetBuilder(
        constructed_dir=cfg.output_dir,
        output_dir=cfg.final_dir,
        split_config=cfg.dataset,
    )
    ds = builder.build(all_examples)
    builder.save(ds)

    # Print summary
    for split_name, split_ds in ds.items():
        console.print(f"  {split_name}: {len(split_ds)} examples")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_constructor(
    task_format: TaskFormat,
    cfg: ConstructionConfig,
) -> BaseConstructor:
    """Lazy-import and instantiate the copywriting constructor."""
    from draper.construction.formats.copywriting.constructor import (
        CopywritingConstructor,
    )

    if task_format is not TaskFormat.COPYWRITING:  # pragma: no cover — enum has one value
        msg = f"Unsupported task format: {task_format.value}"
        raise ValueError(msg)
    return CopywritingConstructor(config=cfg)


if __name__ == "__main__":
    app()
