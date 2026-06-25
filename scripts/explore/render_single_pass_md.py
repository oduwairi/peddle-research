"""Render single-pass teacher I/O as a readable Markdown trace.

For each ad, emit a clean section with:

- Source ad metadata (advertiser, platform, headline/body/description)
- Pretty-printed ``<brief>`` JSON
- The ``<think>`` block as a blockquote
- The verbatim deliverable region as a blockquote (freeform prose for the
  image_brief skill — art-direction copy re-registered from the literal VLM
  caption, with an optional trailing ``Avoid:`` exclusion line)
- Any parse errors

The SYSTEM prompt is emitted once at the top (it's identical across
every ad) inside a collapsible ``<details>`` block to keep the file
scannable. Deliverable content is preserved character-for-character —
no paraphrasing — so the file still serves as an audit artifact.

Usage::

    python scripts/explore/render_single_pass_md.py \\
        --config configs/smoke/construction_v2.openai.yaml \\
        --submissions data/constructed_v2/smoke/single_pass/submissions.json \\
        --selection data/constructed_v2/smoke/_audit_shared/selection.parquet \\
        --out data/constructed_v2/smoke/_audit_shared/single_pass_comparison.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

# Ensure ``src/`` is importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from draper.construction.batch import make_batch_client  # noqa: E402
from draper.construction_v2.config import ConstructionV2Config  # noqa: E402
from draper.construction_v2.dataset.source_selector import (  # noqa: E402
    SourceAd,
    load_source_ads_by_id,
)
from draper.construction_v2.ingest.skills import get_bundle  # noqa: E402


def _blockquote(text: str) -> str:
    """Prefix each line with ``> `` for a markdown blockquote."""
    if not text:
        return "> _(empty)_"
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def _fence(content: str, lang: str = "") -> str:
    return f"```{lang}\n{content}\n```"


def _source_ad_block(ad: SourceAd) -> str:
    """Render a SourceAd as a clean bullet list.

    For image-brief ads enriched with a VLM caption (joined onto
    ``ad.raw["image_caption"]`` by ``bundle.prepare_source_ads``), the
    caption is surfaced as a dedicated block so the reviewer can compare
    it side-by-side with the teacher's ``<image_brief>`` deliverable.
    """
    advertiser = ad.raw.get("advertiser_name") if isinstance(ad.raw, dict) else None
    landing = ad.raw.get("landing_page_url") if isinstance(ad.raw, dict) else None
    creative_url = ad.raw.get("creative_url") if isinstance(ad.raw, dict) else None
    caption = ad.raw.get("image_caption") if isinstance(ad.raw, dict) else None
    lines = [
        f"- **ad_id**: `{ad.ad_id}`",
        f"- **platform**: `{ad.platform}`",
    ]
    if isinstance(advertiser, str) and advertiser:
        lines.append(f"- **advertiser**: {advertiser}")
    if isinstance(landing, str) and landing:
        lines.append(f"- **landing_page**: {landing}")
    if isinstance(creative_url, str) and creative_url:
        lines.append(f"- **creative_url**: {creative_url}")
    lines.append("")
    lines.append("**Ad copy** (verbatim):")
    lines.append("")
    for fld in ("headline", "body", "description"):
        val = getattr(ad, fld) or ""
        if val:
            lines.append(f"- *{fld}*:")
            lines.append("")
            lines.append(_fence(val))
            lines.append("")
        else:
            lines.append(f"- *{fld}*: _(empty)_")
    if isinstance(caption, str) and caption.strip():
        lines.append("")
        lines.append("**VLM caption** (supervision target for `<image_brief>`):")
        lines.append("")
        lines.append(_fence(caption))
    return "\n".join(lines)


def _brief_block(brief: dict | None) -> str:
    if brief is None:
        return "> _(brief missing or unparseable)_"
    return _fence(json.dumps(brief, indent=2, ensure_ascii=False), lang="json")


def _assemble_brief(
    bundle: object, raw_brief: dict | None, ad: SourceAd
) -> tuple[dict | None, str | None]:
    """Return ``(brief_to_render, error_note)`` for the ASSEMBLED brief.

    The renderer must show what the *student* trains on, not the teacher's raw
    emission. The teacher deliberately does NOT author skill-injected fields —
    image_brief's verbatim platform-labeled ``ad_copy`` is injected at ingest by
    the skill's ``build_brief``. So run ``build_brief`` over the teacher's raw
    brief dict to reconstruct the exact ``ImageBriefInput`` / ``Brief`` the
    builder serializes into the user turn.

    Falls back to the raw teacher brief (with an error note) if assembly fails —
    an audit render must never crash on a malformed teacher brief.
    """
    if raw_brief is None:
        return None, None
    try:
        model = bundle.build_brief(raw_brief, ad)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 — audit tool: never crash a render
        return raw_brief, f"{type(exc).__name__}: {exc}"
    return model.model_dump(mode="json"), None


def _think_block(think: str | None) -> str:
    if not think:
        return "> _(think missing)_"
    return _blockquote(think)


def _deliverable_block(deliverable: str | None) -> str:
    if not deliverable:
        return "> _(deliverable missing)_"
    # The image_brief deliverable is now freeform art-direction PROSE (re-registered
    # from the literal VLM caption, with an optional trailing ``Avoid:`` line) rather
    # than a JSON object. Render it as a blockquote so the prose reads naturally in the
    # audit markdown; a code fence would needlessly monospace readable copy. Copywriting
    # deliverables (ad copy + field labels) are equally readable as a blockquote, so this
    # is uniform across skills and preserves the text character-for-character.
    return _blockquote(deliverable)


def _errors_block(errors: list[str]) -> str:
    if not errors:
        return ""
    bullets = "\n".join(f"- {e}" for e in errors)
    return f"**Parse errors:**\n\n{bullets}\n"


async def _fetch_all(
    submissions_path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, str]]:
    """Return ({label: {ad_id: raw_content}}, {label: model}, {label: batch_id})."""
    subs = json.loads(submissions_path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, str]] = {}
    models: dict[str, str] = {}
    batches: dict[str, str] = {}
    for s in subs:
        client = make_batch_client(s["model"])
        results = await client.fetch_results(s["batch_id"])
        per_ad: dict[str, str] = {}
        for r in results:
            ad_id = r.custom_id.removeprefix("teacher-")
            if r.error:
                per_ad[ad_id] = f"<batch-level error: {r.error}>"
            else:
                per_ad[ad_id] = r.content
        out[s["label"]] = per_ad
        models[s["label"]] = s["model"]
        batches[s["label"]] = s["batch_id"]
    return out, models, batches


def _load_sync_results(
    path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, str]]:
    """Same tuple shape as ``_fetch_all`` but read from a sync-run JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, str]] = {}
    models: dict[str, str] = {}
    batches: dict[str, str] = {}
    for entry in data:
        label = entry["label"]
        models[label] = entry["model"]
        batches[label] = "sync"
        per_ad: dict[str, str] = {}
        for r in entry["results"]:
            if r.get("error"):
                per_ad[r["ad_id"]] = f"<sync error: {r['error']}>"
            else:
                per_ad[r["ad_id"]] = r["content"]
        out[label] = per_ad
    return out, models, batches


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--skill",
        default="copywriting",
        choices=["copywriting", "image_brief"],
        help="Skill bundle to dispatch parsing + system-prompt emission through.",
    )
    parser.add_argument("--config", default="configs/smoke/construction_v2.anthropic.yaml")
    parser.add_argument(
        "--submissions",
        default="data/constructed_v2/smoke/single_pass/submissions.json",
    )
    parser.add_argument(
        "--sync-results",
        default=None,
        help="Read raw sync responses from this JSON file instead of polling batches.",
    )
    parser.add_argument(
        "--selection",
        default="data/constructed_v2/smoke/_audit_shared/selection.parquet",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output markdown path. Defaults to "
        "data/constructed_v2/smoke/_audit_shared/single_pass_comparison_<TS>.md "
        "(timestamped so successive renders don't overwrite each other).",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> None:
    cfg = ConstructionV2Config.from_yaml(args.config)
    sel = pl.read_parquet(args.selection)
    ad_ids = sel["ad_id"].to_list()
    ads_by_id = load_source_ads_by_id(cfg, ad_ids)

    bundle = get_bundle(args.skill)

    # Run the bundle's submit-time enrichment over the loaded ads so the
    # renderer can surface skill-specific context (e.g. VLM captions for
    # image_brief). Identity for copywriting.
    ads_list = [ads_by_id[a] for a in ad_ids if a in ads_by_id]
    enriched, _missing = bundle.prepare_source_ads(ads_list, cfg)
    ads_by_id = {ad.ad_id: ad for ad in enriched}

    if args.sync_results:
        fetched, models, batches = _load_sync_results(Path(args.sync_results))
    else:
        fetched, models, batches = await _fetch_all(Path(args.submissions))
    provider_labels = list(fetched.keys())

    lines: list[str] = [
        f"# Single-pass smoke comparison — skill `{args.skill}`",
        "",
        f"- ads: {len([a for a in ad_ids if a in ads_by_id])}",
        f"- providers: {', '.join(f'`{lbl}` ({models[lbl]})' for lbl in provider_labels)}",
        "",
        "<details><summary><b>Teacher SYSTEM prompt</b> "
        "(identical for every ad — click to expand)</summary>",
        "",
        _fence(bundle.system_prompt),
        "",
        "</details>",
        "",
        "---",
        "",
    ]

    for aid in ad_ids:
        ad = ads_by_id.get(aid)
        if ad is None:
            continue
        lines.append(f"## ad `{aid}`")
        lines.append("")
        lines.append("### Source")
        lines.append("")
        lines.append(_source_ad_block(ad))
        lines.append("")

        for label in provider_labels:
            content = fetched.get(label, {}).get(aid, "")
            model_name = models.get(label, "?")
            batch_id = batches.get(label, "")
            parsed = bundle.parse_response(content)

            lines.append(f"### {label} · `{model_name}`")
            lines.append("")
            lines.append(f"_batch `{batch_id}` · custom_id `teacher-{aid}`_")
            lines.append("")
            errors_md = _errors_block(parsed.errors)
            if errors_md:
                lines.append(errors_md)
            assembled, assemble_err = _assemble_brief(bundle, parsed.brief, ad)
            lines.append("**Brief** (assembled — exactly what the student trains on)")
            lines.append("")
            if assemble_err:
                lines.append(
                    f"> _(could not assemble final brief: {assemble_err}; "
                    "showing raw teacher brief)_"
                )
                lines.append("")
            lines.append(_brief_block(assembled))
            lines.append("")
            lines.append("**Think**")
            lines.append("")
            lines.append(_think_block(parsed.think))
            lines.append("")
            lines.append("**Deliverable**")
            lines.append("")
            lines.append(_deliverable_block(parsed.deliverable))
            lines.append("")

        lines.append("---")
        lines.append("")

    if args.out:
        out_path = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(
            f"data/constructed_v2/smoke/_audit_shared/single_pass_comparison_{ts}.md"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}  ({out_path.stat().st_size:,} bytes)")


def main() -> None:
    asyncio.run(_run(_parse_args(sys.argv[1:])))


if __name__ == "__main__":
    main()
