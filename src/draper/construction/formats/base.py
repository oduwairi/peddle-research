"""Format pipeline base class.

Each task format registers a :class:`FormatPipeline` instance describing
how its bundles are selected, personas sampled, ingestion validated, and
quality-filtered. After the 2026-04 pivot to copywriting-only the
registry has a single active pipeline; the scaffolding is kept because
the dispatch model is clean and a future second format can drop in.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from draper.construction.schemas import TaskFormat

if TYPE_CHECKING:
    from draper.construction.bundle import BundleContext
    from draper.construction.formats.copywriting.dice import CopywritingContext
    from draper.construction.personas import Persona, PersonaLibrary
    from draper.construction.schemas import (
        CompositionConfig,
        PromptStyle,
        TrainingExample,
    )
    from draper.construction.source_selector import SourceSelector
    from draper.scoring.schemas import ScoredAd

    # Alias — per-format pipelines only need the (private-member) surface
    # of ``SourceSelector``; this indirection keeps format packages from
    # importing the concrete class at runtime and causing a cycle.
    SelectorContext = SourceSelector


@dataclass
class IngestionCheckResult:
    """Outcome of a per-format ingestion validation step.

    ``passed=True`` means the response is acceptable. ``reason`` is only
    set on failure. Additional per-format context (coverage scores etc.)
    can live on the dataclass as subclasses need.
    """

    passed: bool = True
    reason: str = ""


@dataclass
class BundleAxes:
    """Dice-roll outputs for a single bundle.

    The four non-copywriting formats roll the default axes (persona-driven
    evol operator + difficulty tier, with difficulty-dependent ad reshape).
    Copywriting derives ``copywriting_context`` from the source ad directly
    and leaves ``difficulty`` / ``evol_op`` as empty metadata placeholders.
    """

    difficulty: str
    evol_op: str
    adjusted_ads: list[ScoredAd]
    copywriting_context: CopywritingContext | None = None


class FormatPipeline:
    """Base class every per-format pipeline extends.

    Subclass methods return sentinel/no-op results by default so a format
    only overrides what it actually customizes. The shared construction
    orchestrator calls these methods; individual formats never call each
    other's pipelines.
    """

    #: Which ``TaskFormat`` this pipeline implements.
    task_format: TaskFormat

    #: When True, ``difficulty.sample_difficulty`` remaps ``sparse`` rolls
    #: to ``standard`` for this format (no ads to shrink in single-ad
    #: formats, or directive mismatch in optimization).
    sparse_disallowed: bool = False

    #: When False, ``difficulty.apply_difficulty`` leaves the batch order
    #: alone under the ``conflicting`` difficulty tier. Optimization sets
    #: this False because its high/low pair already carries the tension.
    shuffle_on_conflicting: bool = True

    # ------------------------------------------------------------------
    # Source selection
    # ------------------------------------------------------------------

    def select_batches(
        self,
        selector: SelectorContext,
        consumed_ids: set[str],
        count: int,
        consumed_fingerprints: frozenset[frozenset[str]],
    ) -> list[list[ScoredAd]]:
        """Return ``count`` source-ad batches for this format.

        The default raises — every format must implement selection
        because the contract (single ad? cluster? high/low pair?) differs
        per format and there is no sensible shared default.
        """
        msg = (
            f"FormatPipeline.select_batches must be implemented by "
            f"{type(self).__name__} (task_format={self.task_format.value})"
        )
        raise NotImplementedError(msg)

    # ------------------------------------------------------------------
    # Dice: persona sampling
    # ------------------------------------------------------------------

    def sample_persona(
        self,
        rng: random.Random,
        personas: PersonaLibrary,
        source_ads: list[ScoredAd],
    ) -> Persona:
        """Pick a persona for one bundle.

        Default: uniform sample from the full pool — what positioning /
        diagnostic / optimization / channel_format_fit use today. Formats
        that want context-aware sampling (e.g., copywriting matching
        persona scale to ad scale) override this.
        """
        del source_ads  # unused by default
        return personas.sample(rng)

    # ------------------------------------------------------------------
    # Dice: per-bundle axes (difficulty + evol op + ad reshape)
    # ------------------------------------------------------------------

    def roll_bundle_axes(
        self,
        rng: random.Random,
        source_ads: list[ScoredAd],
        composition: CompositionConfig,
    ) -> BundleAxes:
        """Roll the per-bundle dice for this format.

        Every active format overrides this; the default raises.
        """
        del rng, source_ads, composition
        msg = (
            f"FormatPipeline.roll_bundle_axes must be implemented by "
            f"{type(self).__name__} (task_format={self.task_format.value})"
        )
        raise NotImplementedError(msg)

    # ------------------------------------------------------------------
    # Bundle rendering: per-format axes block
    # ------------------------------------------------------------------

    def render_axes_block(self, ctx: BundleContext) -> list[str]:
        """Return the lines that sit between style rules and source ads.

        Every active format overrides this; the default raises.
        """
        del ctx
        msg = (
            f"FormatPipeline.render_axes_block must be implemented by "
            f"{type(self).__name__} (task_format={self.task_format.value})"
        )
        raise NotImplementedError(msg)

    # ------------------------------------------------------------------
    # Ingestion validation
    # ------------------------------------------------------------------

    def ingestion_check(
        self,
        assistant_response: str,
        source_ads: list[ScoredAd],
        prompt_style: PromptStyle,
    ) -> IngestionCheckResult:
        """Format-specific response validation at ingest time.

        Default passes everything — the shared structural / length /
        language / rubric checks in ``QualityFilter`` are sufficient for
        most formats. Copywriting overrides this to enforce
        backtranslation fidelity (word coverage + verbatim signature).
        """
        del assistant_response, source_ads, prompt_style
        return IngestionCheckResult(passed=True)

    # ------------------------------------------------------------------
    # Quality filter: format-specific extra stages
    # ------------------------------------------------------------------

    def min_length_floor(self, example: TrainingExample, default_floor: int) -> int:
        """Return the minimum assistant-response length for this example.

        Default returns the global floor. Copywriting lowers the floor
        for backtranslation-style examples (real ads + short rationale
        are naturally tight).
        """
        del example
        return default_floor

    def extra_quality_filters(
        self,
        example: TrainingExample,
    ) -> list[tuple[str, str]]:
        """Return ``(example_id, reason)`` rejections for format-specific stages.

        Called after the shared quality filters. Empty list = pass.
        Copywriting returns reasons for schema-label leak or ad-centrality
        hedge violations in backtranslation responses.
        """
        del example
        return []

    # ------------------------------------------------------------------
    # Rubric
    # ------------------------------------------------------------------

    def rubric_check(self, assistant_response: str) -> list[str]:
        """Return missing required-section names (empty = passed).

        Default: no required sections. Formats register their own rubric
        via override.
        """
        del assistant_response
        return []
