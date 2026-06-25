"""Smoke-test VLM captioning of AdFlex creatives for the image-brief skill.

Phase 0 of the image-brief skill plan. Samples N random IMAGE ads, downloads
each creative, and calls Gemini 2.5 Flash twice per image — once with a
LITERAL prompt (describe what is there) and once with a STRATEGIC prompt
(describe what the visual is doing for the message). Writes a Markdown
side-by-side render for human review.

Cost is sub-dollar at default N=50 (sync rates, not batch). Full-corpus
projection (~26.4k ads) is printed at the end so the captioning prompt and
captioner model can be locked in before Phase 2 commits real money.

Run with:
  uv run python scripts/explore/captioning_smoke.py            # 50 ads
  uv run python scripts/explore/captioning_smoke.py --n 10     # smaller smoke
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import typer
from dotenv import load_dotenv
from google import genai
from google.genai import types as gtypes

load_dotenv()

app = typer.Typer(no_args_is_help=False, add_completion=False)

SCORED_ADS = Path("data/scored/v3/scored_ads.parquet")
DEFAULT_N = 50
MODEL = "gemini-3.5-flash"
DOWNLOAD_CONCURRENCY = 8
CAPTION_CONCURRENCY = 4  # be polite to Gemini
DOWNLOAD_TIMEOUT = 30.0
OUT_DIR = Path("data/explore/captioning_smoke")

# Sync (non-batch) Gemini Flash pricing as of 2026-05. Verify on
# https://ai.google.dev/pricing before relying on the full-corpus projection
# (3.5-flash pricing may differ from these 2.5-flash carry-over numbers).
# Batch rates are 50% of sync.
PRICE_PER_M_INPUT_USD = 0.30
PRICE_PER_M_OUTPUT_USD = 2.50
BATCH_DISCOUNT = 0.5
FULL_CORPUS_ADS = 26_410  # image (22709) + carousel first frame (3701)


LITERAL_PROMPT = """You are describing an advertising creative image so it can be used as supervision for a model that generates image briefs.

Describe what is in this image, factually and concretely. Cover:
- the subject and any people, products, or objects
- the setting and background
- composition and framing (close-up, wide, overhead, etc.)
- lighting and color
- any visible text or logos
- the photographic or design style (photography, illustration, 3D render, flat graphic, etc.)

Be specific. Do not interpret intent or strategy — only describe what is literally visible.
Keep the description to 4-7 sentences.
"""


STRATEGIC_PROMPT = """You are describing an advertising creative image so it can be used as supervision for a model that generates image briefs.

Describe both what is in this image AND why it works for the ad's job. Cover:
- the hero subject and how it is staged
- composition and framing decisions (and what they emphasize)
- mood, lighting, and color palette (and the emotional register they create)
- style choices (photography vs illustration vs 3D vs flat graphic, with references where useful)
- how the visual supports a likely angle or buyer (e.g., "frames the product as an effortless daily ritual" rather than just "person holding bottle")
- anything notable about restraint — what was deliberately left OUT

Be specific and concrete. Avoid generic marketing language ("eye-catching", "engaging"). Keep the description to 4-7 sentences.
"""


@dataclass(frozen=True)
class Pick:
    ad_id: str
    creative_url: str
    headline: str
    body: str
    platform: str
    composite_score: float


def _sample_picks(n: int, seed: int) -> list[Pick]:
    df = pl.read_parquet(SCORED_ADS)
    eligible = df.filter(
        (pl.col("creative_format") == "image")
        & (pl.col("creative_url").is_not_null())
        & (pl.col("creative_url") != "")
    )
    n = min(n, eligible.height)
    sample = eligible.sample(n=n, seed=seed)
    picks: list[Pick] = []
    for row in sample.iter_rows(named=True):
        picks.append(
            Pick(
                ad_id=row["ad_id"],
                creative_url=row["creative_url"],
                headline=row.get("ad_copy_headline") or "",
                body=row.get("ad_copy_body") or "",
                platform=row.get("platform") or "",
                composite_score=float(row.get("composite_score") or 0.0),
            )
        )
    return picks


async def _download_one(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, pick: Pick
) -> tuple[Pick, bytes | None, str | None, str | None]:
    """Return (pick, bytes, mime_type, error)."""
    async with sem:
        try:
            r = await client.get(
                pick.creative_url, follow_redirects=True, timeout=DOWNLOAD_TIMEOUT
            )
            r.raise_for_status()
            mime = r.headers.get("content-type", "image/jpeg").split(";")[0].strip()
            return pick, r.content, mime, None
        except httpx.HTTPError as exc:
            return pick, None, None, f"{type(exc).__name__}: {exc}"


async def _download_all(picks: list[Pick]) -> list[tuple[Pick, bytes | None, str | None, str | None]]:
    sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)
    async with httpx.AsyncClient() as client:
        return await asyncio.gather(*(_download_one(client, sem, p) for p in picks))


async def _caption_one(
    client: genai.Client,
    sem: asyncio.Semaphore,
    pick: Pick,
    image_bytes: bytes,
    mime: str,
    prompt: str,
    label: str,
) -> dict[str, Any]:
    async with sem:
        try:
            resp = await client.aio.models.generate_content(
                model=MODEL,
                contents=[
                    gtypes.Content(
                        role="user",
                        parts=[
                            gtypes.Part.from_bytes(data=image_bytes, mime_type=mime),
                            gtypes.Part.from_text(text=prompt),
                        ],
                    )
                ],
                config=gtypes.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1024,
                    thinking_config=gtypes.ThinkingConfig(thinking_budget=0),
                ),
            )
            text = ""
            cands = resp.candidates or []
            if cands and cands[0].content:
                parts = cands[0].content.parts or []
                text = "".join(p.text for p in parts if p.text is not None)
            usage = resp.usage_metadata
            return {
                "ad_id": pick.ad_id,
                "prompt_variant": label,
                "caption": text.strip(),
                "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
                "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 — surface all VLM errors
            return {
                "ad_id": pick.ad_id,
                "prompt_variant": label,
                "caption": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }


def _render_md(
    picks: list[Pick],
    images_ok: dict[str, bool],
    captions: dict[tuple[str, str], dict[str, Any]],
    out_md: Path,
) -> None:
    lines: list[str] = [
        "# Captioning smoke — literal vs strategic",
        "",
        f"Model: `{MODEL}`  ·  N={len(picks)}  ·  see `report.json` for token usage and cost",
        "",
        "---",
        "",
    ]
    for p in picks:
        lit = captions.get((p.ad_id, "literal"), {})
        strat = captions.get((p.ad_id, "strategic"), {})
        copy_excerpt = (p.headline + " — " + p.body).strip(" —")[:240]
        lines.append(f"## `{p.ad_id}`  ·  {p.platform}  ·  score={p.composite_score:.2f}")
        lines.append("")
        lines.append(f"![creative]({p.creative_url})")
        lines.append("")
        lines.append(f"**Ad copy:** {copy_excerpt}")
        lines.append("")
        if not images_ok.get(p.ad_id, False):
            lines.append("> _image failed to download_")
            lines.append("")
            continue
        lines.append("**Literal caption:**")
        lines.append("")
        lines.append(f"> {lit.get('caption', '_(no caption)_')}")
        lines.append("")
        lines.append("**Strategic caption:**")
        lines.append("")
        lines.append(f"> {strat.get('caption', '_(no caption)_')}")
        lines.append("")
        lines.append("---")
        lines.append("")
    out_md.write_text("\n".join(lines))


@app.command()
def main(
    n: int = typer.Option(DEFAULT_N, help="Sample size"),
    seed: int = typer.Option(42, help="RNG seed"),
    output: Path = typer.Option(OUT_DIR, help="Output directory for report + render"),
) -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        typer.echo("ERROR: GEMINI_API_KEY not set in .env", err=True)
        raise typer.Exit(1)

    # Cost preview before calling anything.
    sync_per_ad_usd = (1500 * PRICE_PER_M_INPUT_USD + 400 * PRICE_PER_M_OUTPUT_USD) / 1_000_000
    smoke_cost = sync_per_ad_usd * n * 2  # 2 prompts per ad
    full_batch_cost = sync_per_ad_usd * FULL_CORPUS_ADS * BATCH_DISCOUNT
    typer.echo(
        f"About to caption {n} ads × 2 prompts with {MODEL} (sync). "
        f"Estimated smoke cost: ${smoke_cost:.3f}"
    )
    typer.echo(
        f"Full-corpus projection ({FULL_CORPUS_ADS} ads, single strategic pass, batch tier): "
        f"≈ ${full_batch_cost:.2f}"
    )
    typer.echo("")

    output.mkdir(parents=True, exist_ok=True)
    picks = _sample_picks(n, seed)
    typer.echo(f"Sampled {len(picks)} IMAGE ads")

    typer.echo("Downloading creatives...")
    dl_results = asyncio.run(_download_all(picks))
    images: dict[str, tuple[bytes, str]] = {}
    images_ok: dict[str, bool] = {}
    download_failures: list[dict[str, str]] = []
    for pick, data, mime, err in dl_results:
        if data is None or mime is None:
            images_ok[pick.ad_id] = False
            download_failures.append({"ad_id": pick.ad_id, "url": pick.creative_url, "error": err or "unknown"})
        else:
            images[pick.ad_id] = (data, mime)
            images_ok[pick.ad_id] = True
    typer.echo(
        f"Downloaded {len(images)}/{len(picks)} creatives "
        f"({len(download_failures)} failures)"
    )

    typer.echo("Captioning (literal + strategic per ad)...")
    client = genai.Client(api_key=api_key)
    sem = asyncio.Semaphore(CAPTION_CONCURRENCY)

    async def _run_all_captions() -> list[dict[str, Any]]:
        coros: list[asyncio.Task[dict[str, Any]]] = []
        for pick in picks:
            if pick.ad_id not in images:
                continue
            data, mime = images[pick.ad_id]
            coros.append(
                asyncio.create_task(
                    _caption_one(client, sem, pick, data, mime, LITERAL_PROMPT, "literal")
                )
            )
            coros.append(
                asyncio.create_task(
                    _caption_one(client, sem, pick, data, mime, STRATEGIC_PROMPT, "strategic")
                )
            )
        return await asyncio.gather(*coros)

    cap_rows = asyncio.run(_run_all_captions())

    captions: dict[tuple[str, str], dict[str, Any]] = {
        (r["ad_id"], r["prompt_variant"]): r for r in cap_rows
    }

    # Actual cost tally (sync rates, since we used sync).
    total_in = sum(r["input_tokens"] for r in cap_rows)
    total_out = sum(r["output_tokens"] for r in cap_rows)
    actual_cost = (total_in * PRICE_PER_M_INPUT_USD + total_out * PRICE_PER_M_OUTPUT_USD) / 1_000_000
    cap_errors = [r for r in cap_rows if r["error"]]

    report = {
        "model": MODEL,
        "n_sampled": len(picks),
        "n_captioned": len(images),
        "n_download_failures": len(download_failures),
        "n_caption_errors": len(cap_errors),
        "download_failures": download_failures,
        "caption_errors": [{"ad_id": r["ad_id"], "variant": r["prompt_variant"], "error": r["error"]} for r in cap_errors],
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "smoke_cost_usd_sync": round(actual_cost, 4),
        "projected_full_corpus_cost_usd_batch_single_pass": round(
            ((total_in / max(len(cap_rows), 1) * PRICE_PER_M_INPUT_USD
              + total_out / max(len(cap_rows), 1) * PRICE_PER_M_OUTPUT_USD) / 1_000_000)
            * FULL_CORPUS_ADS
            * BATCH_DISCOUNT,
            2,
        ),
        "captions": cap_rows,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    _render_md(picks, images_ok, captions, output / "render.md")

    typer.echo("")
    typer.echo("=== Done ===")
    typer.echo(f"  captioned:        {len(images)} / {len(picks)} ads (× 2 prompts)")
    typer.echo(f"  caption errors:   {len(cap_errors)}")
    typer.echo(f"  input tokens:     {total_in:,}")
    typer.echo(f"  output tokens:    {total_out:,}")
    typer.echo(f"  smoke cost (sync): ${actual_cost:.4f}")
    typer.echo(
        f"  full-corpus projection (batch, single pass): "
        f"${report['projected_full_corpus_cost_usd_batch_single_pass']:.2f}"
    )
    typer.echo("")
    typer.echo(f"  report: {output / 'report.json'}")
    typer.echo(f"  render: {output / 'render.md'}")


if __name__ == "__main__":
    app()
