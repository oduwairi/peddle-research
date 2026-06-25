"""Abstract base class for training-data constructors.

Every task constructor subclasses ``BaseConstructor`` and provides:

- ``SYSTEM_PROMPT`` — the system message saved into messages[0] of
  every training example.
- ``format_ads_block()`` — renders source ads into the prose block
  embedded in the teacher bundle.

Teacher-bundle assembly itself lives in ``draper.construction.bundle``;
ingest of the teacher's ``<user_prompt>``/``<assistant_response>``
tagged response lives in ``draper.construction.ingestion``.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from draper.construction.schemas import (
    ConstructionConfig,
    PromptStyle,
    TaskFormat,
    TrainingExample,
)
from draper.scoring.schemas import ScoredAd
from draper.utils.io import Checkpoint, append_jsonl, read_jsonl

logger = logging.getLogger("draper")


def business_category(ad: ScoredAd) -> str:
    """Business-vertical label shown to the teacher and written to metadata.

    Prefers ``RawAd.business_vertical`` (LLM-labeled category like
    ``saas_software``). Falls back to the sweep-bucket tail (``facebook:broad``
    → ``broad``) when the business vertical is unset — and only then, since
    the sweep bucket records *how we found* the ad, not what it's selling.
    """
    raw = ad.ad
    if raw.business_vertical:
        return raw.business_vertical
    sweep = raw.vertical
    return sweep.split(":")[-1] if ":" in sweep else sweep


class BaseConstructor(ABC):
    """Shared base for the active task constructors.

    Parameters
    ----------
    task_format:
        Which task format this constructor builds.
    config:
        Construction config (targets, paths, filter settings).
    """

    SYSTEM_PROMPT: str = ""

    def __init__(
        self,
        task_format: TaskFormat,
        config: ConstructionConfig,
    ) -> None:
        self._task_format = task_format
        self._config = config

        self._output_dir = Path(config.output_dir) / task_format.value
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._output_path = self._output_dir / "examples.jsonl"
        self._checkpoint = Checkpoint(self._output_path)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def task_format(self) -> TaskFormat:
        return self._task_format

    @property
    def target(self) -> int:
        return self._config.target_for(self._task_format)

    @property
    def generated_count(self) -> int:
        """How many examples have been written so far."""
        return int(self._checkpoint.get("generated_count", 0))

    @property
    def remaining(self) -> int:
        return max(0, self.target - self.generated_count)

    @property
    def output_dir(self) -> Path:
        """Root output directory shared across all formats."""
        return self._output_dir.parent

    # ------------------------------------------------------------------
    # Style helpers
    # ------------------------------------------------------------------

    def assign_styles(self, count: int) -> list[PromptStyle]:
        """Assign prompt styles for a batch using this format's ratios.

        Copywriting is backtranslation-only so this collapses to the
        single valid style. Kept style-aware (rather than hard-coding
        BACKTRANSLATION) so a future format could reintroduce a mix.
        """
        valid = self._config.valid_styles_for(self._task_format)
        if len(valid) == 1:
            return [valid[0]] * count
        # Fallback: uniform pick across valid styles. No active format
        # exercises this path today.
        seed = self._config.prompt_style.seed + self.generated_count
        rng = random.Random(seed)
        return [valid[rng.randrange(len(valid))] for _ in range(count)]

    # ------------------------------------------------------------------
    # Abstract methods — subclasses implement
    # ------------------------------------------------------------------

    @abstractmethod
    def format_ads_block(self, source_ads: list[ScoredAd]) -> str:
        """Render the source ads into the per-format prose block that
        gets embedded in the teacher bundle (see ``bundle.build_bundle``)."""

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def save_examples(self, examples: list[TrainingExample]) -> int:
        """Append validated examples to the output JSONL file."""
        valid: list[TrainingExample] = []
        for ex in examples:
            if not self._validate_example(ex):
                logger.warning("Invalid example skipped: %s", ex.example_id)
                continue
            valid.append(ex)

        if not valid:
            return 0

        count = append_jsonl(valid, self._output_path)
        new_total = self.generated_count + count
        self._checkpoint.update(generated_count=new_total)
        logger.info(
            "[%s] Saved %d examples (total: %d / %d)",
            self._task_format.value,
            count,
            new_total,
            self.target,
        )
        return count

    def load_existing_examples(self) -> list[TrainingExample]:
        """Load all previously generated examples for this format."""
        if not self._output_path.exists():
            return []
        records = read_jsonl(self._output_path)
        return [TrainingExample(**r) for r in records]

    def consumed_ad_ids(self) -> set[str]:
        """Return ad IDs already used in generated examples."""
        examples = self.load_existing_examples()
        ids: set[str] = set()
        for ex in examples:
            ids.update(ex.metadata.source_ad_ids)
        return ids

    def rejected_ad_ids(self) -> set[str]:
        """Return ad IDs whose teacher response failed an ingestion check.

        Saved examples are tracked via ``consumed_ad_ids``; failed
        examples need their own ledger so the deterministic selector
        doesn't keep handing the same hard ads to every batch. Reads
        ``_rejected_ads.jsonl`` written by ``ingest_response`` when an
        ingestion check rejects a response.
        """
        path = self._output_path.parent / "_rejected_ads.jsonl"
        if not path.exists():
            return set()
        ids: set[str] = set()
        for rec in read_jsonl(path):
            for ad_id in rec.get("source_ad_ids", []) or []:
                ids.add(str(ad_id))
        return ids

    def consumed_bundle_fingerprints(self) -> frozenset[frozenset[str]]:
        """Bundle-level fingerprints already present in ``examples.jsonl``.

        Each fingerprint is the ``frozenset`` of that example's
        ``source_ad_ids``. Passed to ``SourceSelector.select_batches`` so
        a new submission can never emit a bundle whose ad-set duplicates
        one already in the format file.
        """
        examples = self.load_existing_examples()
        fps: set[frozenset[str]] = set()
        for ex in examples:
            ids = ex.metadata.source_ad_ids
            if ids:
                fps.add(frozenset(ids))
        return frozenset(fps)

    def status(self) -> dict[str, Any]:
        """Current progress summary."""
        return {
            "task_format": self._task_format.value,
            "target": self.target,
            "generated": self.generated_count,
            "remaining": self.remaining,
            "output_path": str(self._output_path),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_example(example: TrainingExample) -> bool:
        """Basic structural validation before writing."""
        if not example.messages:
            return False
        # Must have at least a user and assistant turn
        roles = [m.role for m in example.messages]
        if "user" not in roles or "assistant" not in roles:
            return False
        # Assistant response must be non-trivial
        for msg in example.messages:
            if msg.role == "assistant" and len(msg.content.strip()) < 50:
                return False
        return True
