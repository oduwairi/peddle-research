"""Render a batch of constructed training examples as a readable Markdown report.

For each example, the report shows:
  - the full source ad (what the teacher was grounded in)
  - the derived per-bundle metadata (provider, style, source-ad shape,
    conversation register)
  - every conversation turn (system / user / assistant, including follow-ups)

Legacy cross-format axes (``evol_op``, ``difficulty``, ``turn_structure``,
``followup_type``) and the removed RNG axes (``persona_id``, ``scenario_id``,
``rationale_depth``, ``self_rating``, ``brief_ask``, ``brief_tension``,
``platform_framing``) may appear on older rows but are hidden from this
view — voice / shape are now inferred from the source ad by the teacher,
with ``conversation_register`` the only strongly-enforced axis.

Intended use: manual inspection after each iterative batch during copywriting
(and other format) tuning. The markdown lands next to the examples file so it
can be diffed across iterations.

Usage:
    python scripts/assessment/review_examples.py copywriting
    python scripts/assessment/review_examples.py copywriting --limit 10
    python scripts/assessment/review_examples.py copywriting \
        --examples data/constructed/copywriting/examples.jsonl \
        --output   data/constructed/copywriting/examples_review.md

    # Restrict to examples ingested from one or more provider batches:
    python scripts/assessment/review_examples.py copywriting \
        --batch msgbatch_01WfPZh8fpcQed7G94M5njZG \
        --batch batch_69e8da835e508190a42df2d4ee926276
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


METADATA_FIELDS = (
    "construction_model",
    "prompt_style",
    "seed_idx",
    "source_ad_shape",
    "conversation_register",
    "source_ad_ids",
    "source_tiers",
    "source_scores",
    "platform",
    "vertical",
    "batch_id",
    "construction_timestamp",
)

# Fields stored on older ExampleMetadata rows (RNG axes that were removed,
# plus cross-format axes that never vary for copywriting). Hidden from the
# review output to cut visual noise; the "additional fields" auto-render
# below still surfaces anything unknown so genuinely new metadata doesn't
# go missing.
_HIDDEN_FOR_COPYWRITING: frozenset[str] = frozenset(
    {
        "evol_op",
        "difficulty",
        "turn_structure",
        "followup_type",
        # Removed RNG / conditioning axes; may appear on older examples.
        "persona_id",
        "scenario_id",
        "rationale_depth",
        "self_rating",
        "brief_ask",
        "brief_tension",
        "platform_framing",
    }
)


def _load_ads_lookup(scored_path: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    with scored_path.open() as f:
        for line in f:
            ad = json.loads(line)
            aid = ad.get("ad", {}).get("ad_id") or ad.get("ad_id")
            if aid:
                lookup[aid] = ad
    return lookup


def _load_bundles(bundles_path: Path) -> dict[str, str]:
    """Load the example_id → teacher_bundle sidecar written at ingest time.

    Missing file or missing entries are expected for pre-feature rows; the
    renderer falls back to a placeholder and the reviewer knows the bundle
    wasn't captured for that example.
    """
    if not bundles_path.exists():
        return {}
    out: dict[str, str] = {}
    with bundles_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            eid = rec.get("example_id")
            bundle = rec.get("teacher_bundle", "")
            if eid and bundle:
                out[eid] = bundle
    return out


def _fmt_ad(ad: dict[str, Any]) -> str:
    inner = ad.get("ad", ad)
    copy = inner.get("ad_copy") or inner.get("copy") or {}
    return "\n".join(
        [
            f"**ad_id**: `{inner.get('ad_id','?')}`",
            f"**advertiser**: {inner.get('advertiser_name','?')}",
            f"**platform**: {inner.get('platform','?')}",
            f"**creative_format**: {inner.get('creative_format','?')}",
            f"**business_vertical**: {inner.get('business_vertical','?')}",
            f"**vertical** (raw): {inner.get('vertical','?')}",
            f"**language**: {inner.get('language','?')}",
            f"**landing_page_url**: {inner.get('landing_page_url') or '-'}",
            f"**composite_score**: {ad.get('composite_score', '?')}",
            f"**tier**: {ad.get('tier', '?')}",
            "",
            "**headline**:",
            f"> {copy.get('headline') or '(none)'}",
            "",
            "**body**:",
            f"> {copy.get('body') or '(none)'}",
            "",
            "**description**:",
            f"> {copy.get('description') or '(none)'}",
            "",
            "**cta**:",
            f"> {copy.get('cta') or '(none)'}",
        ]
    )


def _short_batch(batch_id: str) -> str:
    """Trim provider batch IDs to something legible in a table cell."""
    if not batch_id:
        return "-"
    # Gemini IDs arrive as ``batches/<id>``; drop the prefix, then cap length.
    tail = batch_id.rsplit("/", 1)[-1]
    return tail[:14] + ("…" if len(tail) > 14 else "")


def _summary_row(i: int, r: dict[str, Any]) -> str:
    m = r.get("metadata", {})
    scores = m.get("source_scores") or []
    ad_ids = m.get("source_ad_ids") or []
    score_str = f"{scores[0]:.2f}" if scores else "-"
    ad_str = ad_ids[0][:12] if ad_ids else "-"
    return (
        f"| {i} | {m.get('construction_model','')} | "
        f"{m.get('platform') or '-'} | "
        f"{m.get('source_ad_shape') or '-'} | "
        f"`{ad_str}` | {score_str} | "
        f"`{_short_batch(m.get('batch_id') or '')}` |"
    )


def _dice_header(m: dict[str, Any]) -> str:
    """One-line context summary at the top of each example."""
    parts = [
        f"**platform**: `{m.get('platform','?')}`",
        f"**source_ad_shape**: `{m.get('source_ad_shape','?')}`",
        f"**provider**: `{m.get('construction_model','?')}`",
    ]
    if m.get("batch_id"):
        parts.append(f"**batch_id**: `{m['batch_id']}`")
    return " · ".join(parts)


def _render_example(
    i: int,
    r: dict[str, Any],
    ads: dict[str, dict[str, Any]],
    bundles: dict[str, str],
    include_bundle: bool = False,
    include_system: bool = False,
    include_source_ads: bool = True,
    include_metadata: bool = True,
) -> list[str]:
    m = r.get("metadata", {})
    messages = r.get("messages", [])
    example_id = r.get("example_id", "?")
    header = (
        f"## Example {i} — `{example_id}` "
        f"({r.get('task_format','?')})\n"
    )
    out: list[str] = [
        header,
        _dice_header(m) + "\n",
    ]

    if include_source_ads:
        out.append("### Source ad(s) (teacher's grounding)\n")
        for aid in m.get("source_ad_ids", []):
            ad = ads.get(aid)
            if ad is None:
                out.append(f"> ⚠️ ad_id `{aid}` not found in scored_ads lookup\n")
                continue
            out.append(_fmt_ad(ad))
            out.append("")

    if include_bundle:
        out.append("### Teacher bundle (exact prompt sent to the provider)")
        bundle = bundles.get(example_id, "")
        if bundle:
            out.append("```")
            out.append(bundle)
            out.append("```\n")
        else:
            out.append(
                "> *(not captured — row pre-dates the bundle sidecar, or capture failed)*\n"
            )

    if include_metadata:
        out.append("### Metadata")
        out.append("```yaml")
        seen: set[str] = set()
        for k in METADATA_FIELDS:
            if k in m:
                out.append(f"{k}: {m[k]}")
                seen.add(k)
        extras = [
            k for k in m if k not in seen and k not in _HIDDEN_FOR_COPYWRITING
        ]
        if extras:
            out.append("# --- additional fields ---")
            for k in sorted(extras):
                out.append(f"{k}: {m[k]}")
        out.append("```\n")

    assistant_seen = False
    for j, msg in enumerate(messages):
        role = msg.get("role", "?")
        if role == "system" and not include_system:
            continue
        content = msg.get("content", "")
        if role == "user":
            label = "USER" if j == 0 or not assistant_seen else "USER (follow-up)"
        elif role == "assistant":
            label = "ASSISTANT (turn 2)" if assistant_seen else "ASSISTANT"
            assistant_seen = True
        else:
            label = role.upper()
        out.append(f"### {label}")
        out.append("```")
        out.append(content)
        out.append("```\n")
    out.append("---\n")
    return out


@app.command()
def main(
    format_name: str = typer.Argument(..., help="Format under review (e.g. 'copywriting')."),
    examples: Path = typer.Option(  # noqa: B008
        None,
        help="Path to examples.jsonl. Defaults to data/constructed/<format>/examples.jsonl.",
    ),
    output: Path = typer.Option(  # noqa: B008
        None,
        help="Markdown output path. Defaults to <examples-dir>/examples_review.md.",
    ),
    scored: Path = typer.Option(  # noqa: B008
        Path("data/scored/v3/scored_ads.jsonl"),
        help="Scored-ads JSONL used to look up source-ad content.",
    ),
    limit: int = typer.Option(0, help="Render only the last N examples (0 = all)."),
    batch: list[str] = typer.Option(  # noqa: B008
        None,
        "--batch",
        help="Restrict output to examples whose metadata.batch_id matches. "
        "Repeatable — pass multiple times to include several batches. "
        "Examples with no batch_id (pre-feature rows, chat-mode ingests) "
        "are excluded when this filter is set.",
    ),
    include_bundle: bool = typer.Option(
        False,
        "--include-bundle/--no-bundle",
        help="Include the full teacher bundle (the exact prompt sent to the "
        "provider) in each example. Off by default — it's long and repeats.",
    ),
    include_system: bool = typer.Option(
        False,
        "--include-system/--no-system",
        help="Include the system message in the rendered conversation. Off by "
        "default — inspectors usually care about user/assistant turns.",
    ),
    include_source_ads: bool = typer.Option(
        True,
        "--include-source-ads/--no-source-ads",
        help="Include the source ad(s) the teacher was grounded in. On by default.",
    ),
    include_metadata: bool = typer.Option(
        True,
        "--include-metadata/--no-metadata",
        help="Include the per-example metadata block. On by default.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Shortcut for --include-bundle --include-system. Overrides individual flags when set.",
    ),
) -> None:
    """Render a Markdown review of constructed examples for inspection."""
    if verbose:
        include_bundle = True
        include_system = True
    examples_path = examples or Path(f"data/constructed/{format_name}/examples.jsonl")
    output_path = output or examples_path.parent / "examples_review.md"

    if not examples_path.exists():
        typer.secho(f"examples file not found: {examples_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if not scored.exists():
        typer.secho(f"scored-ads file not found: {scored}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Loading scored ads from {scored}...")
    ads = _load_ads_lookup(scored)
    typer.echo(f"Loaded {len(ads):,} ads.")

    bundles_path = examples_path.parent / "bundles.jsonl"
    bundles = _load_bundles(bundles_path)
    if bundles:
        typer.echo(f"Loaded {len(bundles)} teacher bundle(s) from {bundles_path}.")
    else:
        typer.echo(f"No teacher bundles found at {bundles_path}.")

    rows = [json.loads(line) for line in examples_path.read_text().splitlines() if line.strip()]
    if batch:
        wanted = set(batch)
        before = len(rows)
        rows = [r for r in rows if (r.get("metadata") or {}).get("batch_id") in wanted]
        typer.echo(
            f"Batch filter: kept {len(rows)}/{before} examples matching "
            f"{sorted(wanted)}."
        )
        if not rows:
            typer.secho(
                "No examples matched the batch filter — check batch IDs "
                "(or re-run batch-collect if those batches aren't ingested yet).",
                fg=typer.colors.YELLOW,
            )
    if limit > 0:
        rows = rows[-limit:]
    typer.echo(f"Rendering {len(rows)} example(s) from {examples_path}...")

    lines: list[str] = [
        f"# {format_name.title()} Examples — Review\n",
        f"Total examples: **{len(rows)}**  ",
        f"Source: `{examples_path}`\n",
        "---\n",
        "## Summary\n",
        "| # | Provider | Platform | Ad shape | Source ad | Score | Batch |",
        "|---|---|---|---|---|---|---|",
    ]
    lines.extend(_summary_row(i, r) for i, r in enumerate(rows))
    lines.append("\n---\n")
    for i, r in enumerate(rows):
        lines.extend(
            _render_example(
                i,
                r,
                ads,
                bundles,
                include_bundle=include_bundle,
                include_system=include_system,
                include_source_ads=include_source_ads,
                include_metadata=include_metadata,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    typer.echo(f"Wrote {output_path} ({output_path.stat().st_size:,} bytes).")


if __name__ == "__main__":
    app()
