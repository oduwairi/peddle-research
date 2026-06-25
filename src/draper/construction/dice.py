"""Shared per-bundle dice-rolling logic.

Both the chat-first `prepare` command and the API-first `batch-submit`
command need to roll the same set of dice for each example (style,
persona, seed, evol op, difficulty, turn structure, followup type) and
build a `BundleContext`. Extracting that into one helper keeps the two
paths in perfect sync — any change to the dice model happens in one
place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from draper.construction.bundle import BundleContext, make_rng
from draper.construction.formats.registry import get_pipeline
from draper.construction.personas import PersonaLibrary
from draper.construction.schemas import (
    ConstructionConfig,
    PromptStyle,
    TaskFormat,
)
from draper.scoring.schemas import ScoredAd

if TYPE_CHECKING:
    from draper.construction.base_constructor import BaseConstructor


@dataclass
class PreparedBundle:
    """All dice rolls + context for a single bundle, ready to consume.

    Callers use ``context`` to render a bundle string (``build_bundle``)
    and ``sidecar`` to persist dice-roll provenance so results can be
    reconciled later (chat: via ``_last_prepared.json``; batch: via the
    pending-batches registry).
    """

    prompt_index: int
    context: BundleContext
    sidecar: dict[str, object]


def prepare_bundles(
    cfg: ConstructionConfig,
    constructor: BaseConstructor,
    personas: PersonaLibrary,
    batches: list[list[ScoredAd]],
    styles: list[PromptStyle],
    provider_label: str,
    rng_offset: int = 0,
) -> list[PreparedBundle]:
    """Roll dice for a list of source-ad batches and return prepared bundles.

    Parameters mirror what `prepare` / `batch-submit` both have in hand:
    the config (for RNG seed), the task constructor (for ad formatting +
    response structure), the persona library, the per-bundle source ads,
    the per-bundle styles, and a provider label that will be saved in
    example metadata as ``construction_model``.

    ``rng_offset`` is added to ``constructor.generated_count`` when seeding
    the per-bundle RNGs. Pass the number of requests already reserved in
    pending batches so two ``batch-submit`` calls before any ingest produce
    different dice rolls (and therefore different prompts).
    """
    task_format: TaskFormat = constructor.task_format
    prepared: list[PreparedBundle] = []
    effective_count = constructor.generated_count + rng_offset

    pipeline = get_pipeline(task_format)

    for i, (source_ads, prompt_style) in enumerate(zip(batches, styles, strict=False)):
        rng = make_rng(cfg.prompt_style.seed, effective_count, i)
        # Persona is inherited metadata — copywriting doesn't render it,
        # but the sidecar/schema fields expect a valid persona id.
        persona = pipeline.sample_persona(rng, personas, source_ads)
        composition = cfg.composition

        # Per-format axes. Copywriting derives its conditioning from the
        # source ad (source_ad_shape only — no RNG, no platform framing).
        axes = pipeline.roll_bundle_axes(rng, source_ads, composition)
        adjusted_ads = axes.adjusted_ads

        formatted_ads = constructor.format_ads_block(adjusted_ads)

        context = BundleContext(
            task_format=task_format,
            style=prompt_style,
            persona=persona,
            seed_idx=-1,
            seed_text="",
            evol_op=axes.evol_op,
            source_ads=adjusted_ads,
            formatted_ads=formatted_ads,
            response_format="",
            difficulty=axes.difficulty,
            turn_structure="single",
            followup_type="",
            provider=provider_label,
            copywriting_context=axes.copywriting_context,
        )

        sidecar: dict[str, object] = {
            "prompt_index": i,
            "source_ad_ids": [a.ad.ad_id for a in adjusted_ads],
            "prompt_style": prompt_style.value,
            "persona_id": persona.id,
            "seed_idx": -1,
            "evol_op": axes.evol_op,
            "difficulty": axes.difficulty,
            "turn_structure": "single",
            "followup_type": "",
            "provider": provider_label,
        }
        if axes.copywriting_context is not None:
            sidecar.update(axes.copywriting_context.as_sidecar())

        prepared.append(PreparedBundle(prompt_index=i, context=context, sidecar=sidecar))

    return prepared
