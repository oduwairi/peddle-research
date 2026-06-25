"""Single-pass v2 teacher smoke (N ads × M providers).

Thin orchestrator that dispatches through the skill bundle registry
(:mod:`draper.construction_v2.ingest.skills`). The same harness runs both
the copywriting and image-brief skills — pick the skill with
``--skill copywriting`` (default) or ``--skill image_brief``. Per-skill
behavior (caption enrichment, request builder, response parser, system
prompt) is encapsulated in the bundle.

Usage::

    # copywriting (default)
    python scripts/explore/single_pass_smoke.py \\
        --config configs/smoke/construction_v2.anthropic.yaml \\
        --providers anthropic,openai,gemini \\
        --out-dir data/constructed_v2/smoke/single_pass \\
        --selection data/constructed_v2/smoke/_audit_shared/selection.parquet

    # image-brief
    python scripts/explore/single_pass_smoke.py \\
        --skill image_brief \\
        --config configs/construction_v2_image.yaml \\
        --providers anthropic,openai,gemini \\
        --out-dir data/constructed_v2_image/smoke/single_pass \\
        --selection data/constructed_v2_image/smoke/_audit_shared/selection.parquet

.. note::

    The image_brief teacher now consumes the **literal** VLM caption (the sole
    visual ground truth it re-registers observational -> directive into the prose
    ``<image_brief>`` deliverable). ``literal`` is the new caption-submit default.
    Captions are read by ``bundle.prepare_source_ads`` ->
    ``enrich_source_ads_with_captions`` from ``data/captions/v1/captions.parquet``,
    so RE-CAPTION the selection with the literal prompt (``caption-submit
    --prompt literal --recaption``) **before** the next image_brief smoke submit.
    This harness is skill-agnostic and needs no edit for the prose-deliverable
    cutover — it serializes whatever the teacher emits.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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
from draper.construction_v2.teacher.single_pass import DEFAULT_TEMPERATURE  # noqa: E402
from draper.utils.llm_client import complete_with_usage  # noqa: E402

PROVIDERS: list[tuple[str, str]] = [
    ("anthropic", "claude-sonnet-4-6"),
    ("openai", "gpt-5.4"),
    ("gemini", "gemini-3.1-pro-preview"),
]


async def _submit_one(
    label: str, model: str, ads: list[SourceAd], temperature: float, skill: str
) -> dict[str, str]:
    build_request = get_bundle(skill).build_request
    requests = [build_request(ad, model=model, temperature=temperature) for ad in ads]
    info = await make_batch_client(model).submit(requests)
    return {
        "label": label,
        "model": model,
        "batch_id": info.batch_id,
        "provider": info.provider,
        "status": info.status.value,
        "skill": skill,
    }


async def _sync_one(
    label: str, model: str, ads: list[SourceAd], temperature: float, skill: str
) -> dict[str, Any]:
    """Fire per-ad sync chat completions for one provider.

    Returns ``{"label", "model", "skill", "results": [{"ad_id", "content", "error"}]}``
    so the render script can consume it the same way it consumes batch fetches.
    """
    build_request = get_bundle(skill).build_request

    async def _call(ad: SourceAd) -> dict[str, Any]:
        req = build_request(ad, model=model, temperature=temperature)
        try:
            res = await complete_with_usage(
                messages=req.messages,
                model=req.model,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                system=req.system,
            )
        except Exception as exc:  # noqa: BLE001 — log every provider failure mode
            return {"ad_id": ad.ad_id, "content": "", "error": f"{type(exc).__name__}: {exc}"}
        return {"ad_id": ad.ad_id, "content": res.content, "error": None}

    results = await asyncio.gather(*(_call(ad) for ad in ads))
    ok = sum(1 for r in results if not r["error"])
    print(f"  {label:10s} ({model}) → {ok}/{len(results)} ok")
    return {"label": label, "model": model, "skill": skill, "results": results}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--skill",
        default="copywriting",
        choices=["copywriting", "image_brief"],
        help="Skill bundle to dispatch through (default: copywriting).",
    )
    parser.add_argument(
        "--config",
        default="configs/smoke/construction_v2.anthropic.yaml",
        help="Path to a v2 config yaml (used only to load source ads).",
    )
    parser.add_argument(
        "--providers",
        default=",".join(label for label, _ in PROVIDERS),
        help="Comma-separated provider labels to submit to.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/constructed_v2/smoke/single_pass",
        help="Directory to write submissions.json into.",
    )
    parser.add_argument(
        "--selection",
        default="data/constructed_v2/smoke/_audit_shared/selection.parquet",
        help="Parquet with the ad_id column for the smoke selection.",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Fire per-ad sync chat completions instead of submitting batches; "
        "writes sync_results.json (consumable by render --sync-results).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default {DEFAULT_TEMPERATURE}). "
        "Some models (e.g. gpt-5.5) only accept 1.0.",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> None:
    selected = {p.strip() for p in args.providers.split(",") if p.strip()}
    providers = [(label, model) for label, model in PROVIDERS if label in selected]
    if not providers:
        print(f"no providers selected (filter={selected})")
        return

    cfg = ConstructionV2Config.from_yaml(args.config)
    ad_ids = pl.read_parquet(args.selection)["ad_id"].to_list()
    ads_by_id = load_source_ads_by_id(cfg, ad_ids)
    ads = [ads_by_id[a] for a in ad_ids if a in ads_by_id]
    print(f"loaded {len(ads)} source ads for single-pass smoke (skill={args.skill})")

    # Skill-specific submit-time enrichment (e.g. join VLM captions for image_brief).
    bundle = get_bundle(args.skill)
    ads, missing = bundle.prepare_source_ads(ads, cfg)
    if missing:
        print(
            f"  prepare_source_ads dropped {len(missing)} ads "
            f"(first few: {missing[:3]}); {len(ads)} remain"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sync:
        results = await asyncio.gather(
            *(
                _sync_one(label, model, ads, args.temperature, args.skill)
                for label, model in providers
            )
        )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"sync_results_{ts}.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved sync results to {out_path}")
        return

    submissions = await asyncio.gather(
        *(
            _submit_one(label, model, ads, args.temperature, args.skill)
            for label, model in providers
        )
    )
    (out_dir / "submissions.json").write_text(
        json.dumps(submissions, indent=2), encoding="utf-8"
    )
    for s in submissions:
        print(f"  {s['label']:10s} → {s['batch_id']:40s} ({s['provider']}, {s['status']})")
    print(f"\nSaved submission metadata to {out_dir / 'submissions.json'}")


def main() -> None:
    asyncio.run(_run(_parse_args(sys.argv[1:])))


if __name__ == "__main__":
    main()
