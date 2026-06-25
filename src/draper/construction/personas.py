"""Persona library for training-data construction.

Personas shape how each training example's user prompt is voiced. The
chat agent receives one persona per example and reshapes the seed
question into that persona's voice before producing a response.

Research basis: Persona Hub (Tencent, arXiv:2406.20094) showed
persona-conditioning measurably improves instruction diversity.
Tulu 3 ablations show +3-5pp on downstream evals vs. unconditional
generation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class Persona(BaseModel):
    """A marketing persona used to condition prompt generation."""

    id: str
    role: str
    tone: str
    sophistication: str = ""
    budget: str = ""
    industry: str = ""
    # Business scale this persona realistically commissions ads at. Used by
    # the copywriting source selector to skip persona×ad mismatches like
    # "local boutique owner brief for a Shangri-La chain ad." Values:
    # ``small`` (local / solo / indie), ``mid`` (DTC / SMB / mid-market),
    # ``large`` (enterprise / global), ``any`` (vertical-agnostic, e.g.,
    # agency strategists who work across scales). Empty = treated as
    # ``any`` for backward compatibility.
    scale: str = ""
    # Task formats this persona is valid for. Empty = valid for all formats
    # (backward compatible). Per-format persona pools let copywriting
    # restrict to personas whose voice makes sense for a commissioning
    # brief, while other formats can keep the broader pool.
    formats: list[str] = Field(default_factory=list)
    # Verticals this persona may be paired with. Allowlist wins if
    # non-empty. Otherwise the denylist applies. Both empty = any vertical.
    # Drives the copywriting persona×vertical compatibility check (review
    # batch Ex 6 paired ``ecommerce_shopify_owner`` with a theatre-play ad).
    allowed_verticals: list[str] = Field(default_factory=list)
    blocked_verticals: list[str] = Field(default_factory=list)
    # Relative sampling weight. Default 1.0 keeps legacy uniform behavior.
    # Use higher weights for common, broadly-applicable personas and lower
    # weights for niche specialists — uniform sampling over a
    # specialist-heavy pool over-represents edge cases in the training mix.
    weight: float = 1.0

    def is_compatible_with(
        self,
        *,
        task_format: str = "",
        ad_scale: str = "",
        ad_vertical: str = "",
    ) -> bool:
        """Return True if this persona may be paired with the given ad context.

        Any empty criterion is skipped. Used by ``PersonaLibrary.sample_for_ad``
        to prune the candidate pool before sampling.
        """
        if task_format and self.formats and task_format not in self.formats:
            return False
        if ad_scale and self.scale and self.scale not in ("any", ad_scale):
            return False
        if ad_vertical:
            if self.allowed_verticals and ad_vertical not in self.allowed_verticals:
                return False
            if ad_vertical in self.blocked_verticals:
                return False
        return True

    def to_bundle_block(self) -> str:
        """Render the persona as text for inclusion in a teacher bundle."""
        lines = [
            f"- Role: {self.role}",
            f"- Tone: {self.tone}",
        ]
        if self.sophistication:
            lines.append(f"- Sophistication: {self.sophistication}")
        if self.budget:
            lines.append(f"- Budget: {self.budget}")
        if self.industry:
            lines.append(f"- Industry: {self.industry}")
        return "\n".join(lines)


class PersonaLibrary(BaseModel):
    """The full pool of personas loaded from ``configs/personas.yaml``."""

    personas: list[Persona] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/personas.yaml") -> PersonaLibrary:
        """Load personas from a YAML file."""
        with Path(path).open() as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        return cls(personas=[Persona(**p) for p in raw.get("personas", [])])

    def sample(self, rng: random.Random) -> Persona:
        """Pick one persona, weighted by ``Persona.weight``."""
        if not self.personas:
            msg = "PersonaLibrary is empty — cannot sample"
            raise ValueError(msg)
        return _weighted_persona_choice(rng, self.personas)

    def sample_for_scale(self, rng: random.Random, ad_scale: str) -> Persona:
        """Pick a persona whose ``scale`` is compatible with ``ad_scale``.

        Personas tagged ``any`` or with empty ``scale`` are always eligible.
        Sampling is weighted by ``Persona.weight`` so common personas get
        picked more often than specialist ones. Falls back to the full pool
        if the compatible set is empty.
        """
        if not self.personas:
            msg = "PersonaLibrary is empty — cannot sample"
            raise ValueError(msg)
        compatible = [
            p
            for p in self.personas
            if not p.scale or p.scale == "any" or p.scale == ad_scale
        ]
        if compatible:
            return _weighted_persona_choice(rng, compatible)
        return _weighted_persona_choice(rng, self.personas)

    def sample_for_ad(
        self,
        rng: random.Random,
        *,
        task_format: str = "",
        ad_scale: str = "",
        ad_vertical: str = "",
    ) -> Persona:
        """Pick a persona compatible with the ad's (format, scale, vertical).

        Stricter than ``sample_for_scale`` — also respects per-persona
        ``formats``, ``allowed_verticals``, and ``blocked_verticals`` fields.
        Sampling is weighted by ``Persona.weight`` within the compatible
        set. Falls back progressively (drop vertical filter, then format
        filter, finally scale-only) so the pool never starves.
        """
        if not self.personas:
            msg = "PersonaLibrary is empty — cannot sample"
            raise ValueError(msg)
        # Strict: all criteria apply.
        compatible = [
            p
            for p in self.personas
            if p.is_compatible_with(
                task_format=task_format,
                ad_scale=ad_scale,
                ad_vertical=ad_vertical,
            )
        ]
        if compatible:
            return _weighted_persona_choice(rng, compatible)
        # Fallback 1: drop vertical filter (pool has nobody for this vertical).
        relaxed = [
            p
            for p in self.personas
            if p.is_compatible_with(task_format=task_format, ad_scale=ad_scale)
        ]
        if relaxed:
            return _weighted_persona_choice(rng, relaxed)
        # Fallback 2: legacy scale-only behavior.
        return self.sample_for_scale(rng, ad_scale or "mid")

    def by_id(self, persona_id: str) -> Persona | None:
        """Look up a persona by ID (for reconstructing from metadata)."""
        for p in self.personas:
            if p.id == persona_id:
                return p
        return None

    def __len__(self) -> int:
        return len(self.personas)


def _weighted_persona_choice(
    rng: random.Random, personas: list[Persona]
) -> Persona:
    """Weighted random choice over personas. Zero/negative weights skipped."""
    positive = [p for p in personas if p.weight > 0]
    if not positive:
        return rng.choice(personas)
    total = sum(p.weight for p in positive)
    r = rng.uniform(0, total)
    cum = 0.0
    for p in positive:
        cum += p.weight
        if r <= cum:
            return p
    return positive[-1]
