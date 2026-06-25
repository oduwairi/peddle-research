"""Generate live-deployment ad creative for the RQ1 cells.

For each brief in the input file, produces a Meta ad creative per cell:

- ``gpt55``: pure GPT-5.5 single-shot using the writer-side training system
  prompt verbatim, plus a gpt-image-2 image generated from the asset brief.
- ``draper_direct``: single-shot ``draper-r16`` on Modal vLLM with the same
  eval system prompt — no agent loop, no tools, no JSON schema. This is the
  in-distribution path that mirrors ``data/eval/inferences/C/*.json`` (config C
  in the eval pipeline). Use this for head-to-head writer-vs-writer copy
  comparisons.
- ``draper_agent``: full frontend agent loop (orchestrator + Draper writer +
  generate_image), hit via the existing /api/eval/run endpoint. Use this for
  end-to-end pipeline comparisons (RAG, research, image gen all in the loop).

The training-time system prompt is sourced from
``src/draper/construction/formats/copywriting/constructor.py``; the same string
is duplicated in ``frontend/lib/agent/tools/draft-campaign.ts`` and
``frontend/lib/agent/tools/ask-draper.ts``. If you change it in one place,
update all three.

Briefs MUST follow the eval format (terse, founder-paste, period-separated
facts — see ``data/eval/inferences/C/*.json``). Marketing prose, URLs, and
multi-paragraph descriptions are out-of-distribution for ``draper-r16`` and
will trigger the product-catalog failure mode.

Output: ``data/live_deploy/<run_id>/<cell>/<brief_id>/{copy.json, image.png}``.

Usage:
    python scripts/live_deploy/generate_ads.py \\
        --briefs scripts/live_deploy/briefs.example.json \\
        --cells gpt55,draper_direct \\
        --run-id 2026-05-17

Env vars required:
- ``OPENAI_API_KEY``: for GPT-5.5 prose + extraction + gpt-image-2.
- ``VLLM_BASE_URL`` / ``VLLM_API_KEY``: Modal vLLM endpoint (draper_direct).
- ``EVAL_FRONTEND_D_URL`` / ``EVAL_SERVICE_TOKEN``: only needed for draper_agent.
- ``SCORING_PREDICTOR_URL`` / ``SCORING_PREDICTOR_API_KEY``: local predictor.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import openai
import typer
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Verbatim copy of the writer-side system prompt. Same string as
# src/draper/construction/formats/copywriting/constructor.py:79-82 and
# frontend/lib/agent/tools/draft-campaign.ts:45-46.
# ---------------------------------------------------------------------------
TRAINING_SYSTEM_PROMPT = (
    "You are an ad copywriter. When a user describes a product or campaign, "
    "you write ad copy and a short rationale explaining why the execution works."
)

# Meta CTA enum — must match frontend/lib/agent/platforms.ts META_CTAS exactly.
META_CTAS = [
    "Sign up",
    "Learn more",
    "Get offer",
    "Shop now",
    "Subscribe",
    "Get quote",
    "Apply now",
    "Book now",
    "Download",
    "Contact us",
    "Send message",
    "Get started",
]


class MetaAdCopy(BaseModel):
    """Structured Meta ad fields, mirroring frontend MetaCampaignBodySchema."""

    primary_text: str = Field(..., description="Feed body copy, ≤500 chars hard cap.")
    headline: str = Field(..., description="Below-image headline, ≤40 chars hard cap.")
    description: str = Field("", description="Optional sub-headline, ≤30 chars; empty if omitted.")
    cta_label: str = Field(..., description=f"One of {META_CTAS}.")
    asset_brief: str = Field(
        ...,
        description="One-sentence image brief — subject, composition, mood, no text in image.",
    )


@dataclass(frozen=True)
class Brief:
    id: str
    platform: str
    user_prompt: str


@dataclass(frozen=True)
class GenerateConfig:
    run_id: str
    output_root: Path
    gpt55_model: str = "gpt-5.5"
    draper_model: str = "draper-r16"
    extractor_model: str = "gpt-5.4-mini"
    image_model: str = "gpt-image-2"
    predictor_url: str = "http://127.0.0.1:8001"
    predictor_api_key: str | None = None


def _next_iter_dir(parent: Path, brief_id: str) -> Path:
    """Return a fresh dir under ``parent`` so prior iterations are preserved.

    First run writes to ``<brief_id>/``. Subsequent runs write to
    ``<brief_id>__v2/``, ``<brief_id>__v3/``, ... — picked by counting existing
    siblings rather than mtimes so the numbering is stable across re-runs.
    """
    base = parent / brief_id
    if not base.exists():
        return base
    n = 2
    while (parent / f"{brief_id}__v{n}").exists():
        n += 1
    return parent / f"{brief_id}__v{n}"


async def _score_meta_copy(
    base_url: str, api_key: str, fields: MetaAdCopy
) -> dict[str, float] | None:
    """Score a Meta ad via the local scoring predictor. Returns None on miss.

    Mirrors ``frontend/lib/agent/scoring/predictor-client.ts`` — same URL path,
    same X-API-Key header, same item shape.
    """
    url = f"{base_url.rstrip('/')}/score"
    payload = {
        "items": [
            {
                "platform": "meta",
                "vertical": "unknown",
                "headline": fields.headline,
                "body": fields.primary_text,
                "description": fields.description or None,
            }
        ]
    }
    try:
        async with httpx.AsyncClient(timeout=15) as http:
            r = await http.post(
                url,
                json=payload,
                headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            )
            r.raise_for_status()
            data = r.json()
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        typer.echo(f"[score] predictor call failed: {e}", err=True)
        return None
    scores = data.get("scores") or []
    if not scores:
        return None
    head = scores[0]
    return {
        "composite": head.get("composite"),
        "survivability": head.get("survivability"),
        "engagement_volume": head.get("engagement_volume"),
        "engagement_velocity": head.get("engagement_velocity"),
    }


def load_briefs(path: Path) -> list[Brief]:
    raw = json.loads(path.read_text())
    return [Brief(**b) for b in raw["briefs"]]


async def _writer_prose(
    client: openai.AsyncOpenAI,
    model: str,
    brief: Brief,
    *,
    label: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """Single-shot completion using the eval training system prompt verbatim.

    Same call shape as ``src/draper/evaluation/inference/vllm_runner.py`` and
    ``openai_runner.py``: ``[{system}, {user}]``, no tools, no JSON schema. This
    is the in-distribution path for ``draper-r16`` (matches its training format)
    and is reused for ``gpt-5.5`` so the two cells are apples-to-apples.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
            {"role": "user", "content": brief.user_prompt},
        ],
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = await client.chat.completions.create(**kwargs)
    text: str | None = resp.choices[0].message.content
    if not text:
        raise RuntimeError(f"{label} returned empty content for brief {brief.id}")
    return text.strip()


async def _extract_meta_fields(
    client: openai.AsyncOpenAI, model: str, prose: str
) -> MetaAdCopy:
    """Extract structured Meta fields from free-form prose.

    Mirrors the orchestrator's extraction pass in draft-campaign.ts. The model
    is told to slot verbatim from the writer's prose, not to rewrite.
    """
    extraction_prompt = (
        "You are extracting structured Meta ad fields from a copywriter's prose draft. "
        "Slot fields VERBATIM from the prose where possible — do not rewrite the copy. "
        "If a field is implied but not written verbatim, choose the closest verbatim "
        f"phrase. cta_label must be exactly one of: {META_CTAS}."
    )
    resp = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": f"Writer's draft:\n\n{prose}"},
        ],
        response_format=MetaAdCopy,
    )
    parsed = resp.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("Extraction returned no parsed content")
    if parsed.cta_label not in META_CTAS:
        # Fall back to a safe default rather than fail — Meta will accept any
        # of the listed CTAs.
        parsed = parsed.model_copy(update={"cta_label": "Learn more"})
    return parsed


async def _generate_image(
    client: openai.AsyncOpenAI, model: str, asset_brief: str, out_path: Path
) -> None:
    """Generate a Meta-square (1024x1024) image via gpt-image-2."""
    resp = await client.images.generate(
        model=model,
        prompt=asset_brief,
        size="1024x1024",
        quality="medium",
        n=1,
    )
    data = resp.data[0]
    if data.b64_json:
        out_path.write_bytes(base64.b64decode(data.b64_json))
        return
    if data.url:
        async with httpx.AsyncClient(timeout=60) as http:
            r = await http.get(data.url)
            r.raise_for_status()
            out_path.write_bytes(r.content)
        return
    raise RuntimeError("gpt-image-2 returned neither b64 nor url")


async def run_gpt55_cell(
    client: openai.AsyncOpenAI, cfg: GenerateConfig, brief: Brief
) -> dict[str, Any]:
    cell_dir = _next_iter_dir(cfg.output_root / cfg.run_id / "gpt55", brief.id)
    cell_dir.mkdir(parents=True, exist_ok=True)

    prose = await _writer_prose(client, cfg.gpt55_model, brief, label="gpt55")
    (cell_dir / "raw_prose.txt").write_text(prose)

    fields = await _extract_meta_fields(client, cfg.extractor_model, prose)

    score: dict[str, float] | None = None
    if cfg.predictor_api_key:
        score = await _score_meta_copy(cfg.predictor_url, cfg.predictor_api_key, fields)
    else:
        typer.echo("[score] SCORING_PREDICTOR_API_KEY not set — skipping score", err=True)

    copy_dict = fields.model_dump()
    copy_dict["predicted_score"] = score
    copy_path = cell_dir / "copy.json"
    copy_path.write_text(json.dumps(copy_dict, indent=2))

    image_path = cell_dir / "image.png"
    await _generate_image(client, cfg.image_model, fields.asset_brief, image_path)

    return {
        "brief_id": brief.id,
        "cell": "gpt55",
        "iter_dir": cell_dir.name,
        "copy_path": str(copy_path),
        "image_path": str(image_path),
        "predicted_score": score,
        "fields": fields.model_dump(),
    }


async def run_draper_direct_cell(
    vllm_client: openai.AsyncOpenAI,
    openai_client: openai.AsyncOpenAI,
    cfg: GenerateConfig,
    brief: Brief,
) -> dict[str, Any]:
    """Single-shot Draper inference — no agent loop, no JSON schema.

    Mirrors the eval `VLLMRunner` path used for config C in
    `data/eval/inferences/C/`. After Draper produces prose, we extract Meta
    fields with the same gpt-5.4-mini extractor used for the gpt55 cell so
    both cells produce identical-shape outputs for Meta Ads Manager.
    """
    cell_dir = _next_iter_dir(cfg.output_root / cfg.run_id / "draper_direct", brief.id)
    cell_dir.mkdir(parents=True, exist_ok=True)

    # Draper-r16 was trained with greedy decoding; mirror VLLMRunner defaults
    # (temperature=0.0, max_tokens=1024) so this is bit-comparable to eval runs.
    prose = await _writer_prose(
        vllm_client,
        cfg.draper_model,
        brief,
        label="draper_direct",
        max_tokens=1024,
        temperature=0.0,
    )
    (cell_dir / "raw_prose.txt").write_text(prose)

    fields = await _extract_meta_fields(openai_client, cfg.extractor_model, prose)

    score: dict[str, float] | None = None
    if cfg.predictor_api_key:
        score = await _score_meta_copy(cfg.predictor_url, cfg.predictor_api_key, fields)
    else:
        typer.echo("[score] SCORING_PREDICTOR_API_KEY not set — skipping score", err=True)

    copy_dict = fields.model_dump()
    copy_dict["predicted_score"] = score
    copy_path = cell_dir / "copy.json"
    copy_path.write_text(json.dumps(copy_dict, indent=2))

    image_path = cell_dir / "image.png"
    await _generate_image(openai_client, cfg.image_model, fields.asset_brief, image_path)

    return {
        "brief_id": brief.id,
        "cell": "draper_direct",
        "iter_dir": cell_dir.name,
        "copy_path": str(copy_path),
        "image_path": str(image_path),
        "predicted_score": score,
        "fields": fields.model_dump(),
    }


async def run_draper_agent_cell(cfg: GenerateConfig, brief: Brief) -> dict[str, Any]:
    """Hit the existing /api/eval/run endpoint on the production frontend."""
    base_url = os.environ["EVAL_FRONTEND_D_URL"].rstrip("/")
    token = os.environ["EVAL_SERVICE_TOKEN"]

    cell_dir = _next_iter_dir(cfg.output_root / cfg.run_id / "draper_agent", brief.id)
    cell_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "userPrompt": brief.user_prompt,
        "platform": brief.platform,
        "exampleId": brief.id,
    }
    async with httpx.AsyncClient(timeout=300) as http:
        r = await http.post(
            f"{base_url}/api/eval/run",
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        r.raise_for_status()
        result = r.json()

    (cell_dir / "raw.json").write_text(json.dumps(result, indent=2))

    campaign = result.get("campaign")
    if not campaign:
        return {
            "brief_id": brief.id,
            "cell": "draper_agent",
            "error": "no campaign emitted",
            "result": result,
        }

    # Distill the Meta-specific fields the user will paste into Meta Ads
    # Manager. The frontend already produced the image and uploaded it to S3 —
    # we just point at the URL.
    # emit_campaign auto-scores via the predictor and attaches the composite as
    # campaign.predicted_score. Surface all four heads if available (full
    # breakdown is in raw.json); otherwise just the composite scalar.
    composite = campaign.get("predicted_score")
    predicted_score: dict[str, float] | float | None
    if isinstance(composite, (int, float)):
        predicted_score = {"composite": float(composite)}
    else:
        predicted_score = composite  # may be None or already a dict

    copy = {
        "primary_text": campaign.get("primary_text"),
        "headline": campaign.get("headline"),
        "description": campaign.get("description", ""),
        "cta_label": campaign.get("cta_label"),
        "asset_brief": campaign.get("asset_brief"),
        "asset_image_url": campaign.get("asset_image_url"),
        "predicted_score": predicted_score,
    }
    copy_path = cell_dir / "copy.json"
    copy_path.write_text(json.dumps(copy, indent=2))

    return {
        "brief_id": brief.id,
        "cell": "draper_agent",
        "iter_dir": cell_dir.name,
        "copy_path": str(copy_path),
        "asset_image_url": copy["asset_image_url"],
        "predicted_score": predicted_score,
    }


app = typer.Typer(add_completion=False)


@app.command()
def main(
    briefs: Path = typer.Option(..., "--briefs", help="JSON file with the brief list."),  # noqa: B008
    cells: str = typer.Option(  # noqa: B008
        "gpt55,draper_direct",
        "--cells",
        help="Comma-separated list of cells to generate (gpt55, draper_direct, draper_agent).",
    ),
    run_id: str = typer.Option(..., "--run-id", help="Sub-dir name for outputs."),  # noqa: B008
    output_root: Path = typer.Option(  # noqa: B008
        Path("data/live_deploy"),
        "--output-root",
        help="Root output directory.",
    ),
) -> None:
    cfg = GenerateConfig(
        run_id=run_id,
        output_root=output_root,
        predictor_url=os.environ.get("SCORING_PREDICTOR_URL", "http://127.0.0.1:8001"),
        predictor_api_key=os.environ.get("SCORING_PREDICTOR_API_KEY"),
    )
    brief_list = load_briefs(briefs)
    selected_cells = {c.strip() for c in cells.split(",") if c.strip()}

    async def _run() -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        openai_client: openai.AsyncOpenAI | None = None
        vllm_client: openai.AsyncOpenAI | None = None
        if "gpt55" in selected_cells or "draper_direct" in selected_cells:
            openai_client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        if "draper_direct" in selected_cells:
            vllm_client = openai.AsyncOpenAI(
                base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
                api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            )
        for brief in brief_list:
            if "gpt55" in selected_cells:
                assert openai_client is not None
                typer.echo(f"[gpt55] brief={brief.id}")
                results.append(await run_gpt55_cell(openai_client, cfg, brief))
            if "draper_direct" in selected_cells:
                assert openai_client is not None and vllm_client is not None
                typer.echo(f"[draper_direct] brief={brief.id}")
                results.append(
                    await run_draper_direct_cell(vllm_client, openai_client, cfg, brief)
                )
            if "draper_agent" in selected_cells:
                typer.echo(f"[draper_agent] brief={brief.id}")
                results.append(await run_draper_agent_cell(cfg, brief))
        return results

    results = asyncio.run(_run())

    summary_path = cfg.output_root / cfg.run_id / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2))
    typer.echo(f"\nDone. Summary written to {summary_path}")


if __name__ == "__main__":
    app()
