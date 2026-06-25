"""Copywriting :class:`FormatPipeline` — wires the submodules together.

Imported by :mod:`draper.construction.formats.copywriting.__init__`,
which registers the singleton with the format registry.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from draper.construction.formats.base import (
    BundleAxes,
    FormatPipeline,
    IngestionCheckResult,
)
from draper.construction.formats.copywriting import (
    ingestion as cw_ingestion,
)
from draper.construction.formats.copywriting import (
    quality_filter as cw_quality,
)
from draper.construction.formats.copywriting import (
    rubric as cw_rubric,
)
from draper.construction.formats.copywriting import (
    selector as cw_selector,
)
from draper.construction.formats.copywriting.dice import (
    derive_copywriting_context,
    render_conversation_register_directive,
)
from draper.construction.schemas import PromptStyle, TaskFormat, TrainingExample
from draper.scoring.schemas import ScoredAd

if TYPE_CHECKING:
    from draper.construction.bundle import BundleContext
    from draper.construction.schemas import CompositionConfig
    from draper.construction.source_selector import SourceSelector

logger = logging.getLogger("draper")


class CopywritingPipeline(FormatPipeline):
    """Backtranslation-mode copywriting pipeline.

    Persona sampling is inherited from :class:`FormatPipeline` (uniform
    random); copywriting bundles don't actually render the persona into
    the teacher prompt — voice comes from the source ad via the
    BACKTRANSLATION style rules — but the shared dice orchestrator still
    requires a persona object on :class:`BundleContext`.
    """

    task_format = TaskFormat.COPYWRITING
    sparse_disallowed = True  # single-ad format — nothing to shrink

    def select_batches(
        self,
        selector: SourceSelector,
        consumed_ids: set[str],
        count: int,
        consumed_fingerprints: frozenset[frozenset[str]],
    ) -> list[list[ScoredAd]]:
        return cw_selector.select_batches(
            selector, consumed_ids, count, consumed_fingerprints
        )

    def roll_bundle_axes(
        self,
        rng: random.Random,
        source_ads: list[ScoredAd],
        composition: CompositionConfig,
    ) -> BundleAxes:
        """Derive per-bundle context from the source ad — no RNG.

        Copywriting leaves voice / shape / depth to the teacher to infer
        from the ad itself via the BACKTRANSLATION style rules. Difficulty
        and evol_op become empty metadata placeholders so downstream
        provenance stays schema-compatible.
        """
        del rng, composition  # copywriting is deterministic from the ad
        context = (
            derive_copywriting_context(source_ads[0]) if source_ads else None
        )
        return BundleAxes(
            difficulty="standard",
            evol_op="",
            adjusted_ads=list(source_ads),
            copywriting_context=context,
        )

    def render_axes_block(self, ctx: BundleContext) -> list[str]:
        """Emit the rolled ``conversation_register`` as a strong directive.

        ``source_ad_shape`` stays implicit (the teacher can read it off
        the ad). ``conversation_register`` is the one strongly-enforced
        axis: the teacher is told the exact register to use on BOTH the
        brief and the response, with no menu of choices, so each provider
        can't collapse to its preferred default opener or prose voice.
        Rationale depth is held constant across registers — only opener
        and prose register vary, never the analytic depth.
        """
        if ctx.copywriting_context is None:
            return []
        directive = render_conversation_register_directive(
            ctx.copywriting_context.conversation_register
        )
        return [directive, ""]

    def ingestion_check(
        self,
        assistant_response: str,
        source_ads: list[ScoredAd],
        prompt_style: PromptStyle,
    ) -> IngestionCheckResult:
        """Enforce backtranslation fidelity on the assistant response."""
        if prompt_style != PromptStyle.BACKTRANSLATION:
            return IngestionCheckResult(passed=True)

        passed, coverage, ad_words = cw_ingestion.check_word_coverage(
            assistant_response, source_ads
        )
        if not passed:
            logger.warning(
                "Backtranslation fidelity failed: only %.0f%% of %d source-"
                "ad content words appear in assistant_response (min=%.0f%%).",
                coverage * 100,
                ad_words,
                cw_ingestion.BACKTRANS_MIN_WORD_COVERAGE * 100,
            )
            return IngestionCheckResult(
                passed=False,
                reason=(
                    f"Backtranslation fidelity failed: word coverage "
                    f"{coverage:.2f} < {cw_ingestion.BACKTRANS_MIN_WORD_COVERAGE:.2f}. "
                    f"Teacher likely fabricated copy instead of reproducing "
                    f"the real ad."
                ),
            )

        verbatim_ok, reason = cw_ingestion.check_verbatim_signature(
            assistant_response, source_ads
        )
        if not verbatim_ok:
            logger.warning("Backtranslation %s", reason)
            return IngestionCheckResult(passed=False, reason=reason)

        return IngestionCheckResult(passed=True)

    def min_length_floor(
        self, example: TrainingExample, default_floor: int
    ) -> int:
        return cw_quality.min_length_floor(example, default_floor)

    def extra_quality_filters(
        self, example: TrainingExample
    ) -> list[tuple[str, str]]:
        return cw_quality.extra_filter_reasons(example)

    def rubric_check(self, assistant_response: str) -> list[str]:
        return cw_rubric.check(assistant_response)
