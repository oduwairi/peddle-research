"""Teacher-provider rotation helpers.

Research basis ("Synthetic Eggs in Many Baskets", 2025): single-teacher
synthetic data causes distribution collapse — the fine-tune learns the
teacher's stylistic quirks rather than the domain. Batch-level rotation
across 2-3 providers prevents this.

Granularity: per-batch (not per-example). Chat-subscription workflow
pastes one whole batch into one chat session, so per-example rotation is
incoherent. Before each ``prepare`` run, we tally what's been generated
so far and recommend the provider whose current share is farthest below
its target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from draper.construction.batch.registry import BatchRegistry
from draper.construction.schemas import (
    ConstructionConfig,
    ProviderRotationConfig,
    TaskFormat,
)
from draper.utils.io import read_jsonl

# Mirrors the wire-provider names the registry uses.
_REGISTRY_PROVIDER_TO_LABEL = {
    "openai": "gpt",
    "anthropic": "claude",
    "gemini": "gemini",
}


def classify_provider(construction_model: str) -> str:
    """Map a declared or self-reported model name to a provider key.

    Heuristic matching — keywords in the model string determine the
    provider. Unknown strings return ``"unknown"``.
    """
    if not construction_model:
        return "unknown"
    lowered = construction_model.lower()
    if "claude" in lowered or "anthropic" in lowered:
        return "claude"
    if "gpt" in lowered or "openai" in lowered or "o1" in lowered or "o3" in lowered:
        return "gpt"
    if "gemini" in lowered or "google" in lowered:
        return "gemini"
    return "unknown"


def tally_provider_counts(cfg: ConstructionConfig) -> dict[str, int]:
    """Count existing examples per provider across all formats."""
    counts: dict[str, int] = {"claude": 0, "gpt": 0, "gemini": 0, "unknown": 0}
    for fmt in TaskFormat:
        for prov, n in tally_provider_counts_for_format(cfg, fmt).items():
            counts[prov] = counts.get(prov, 0) + n
    return counts


def tally_provider_counts_for_format(
    cfg: ConstructionConfig, task_format: TaskFormat
) -> dict[str, int]:
    """Count examples per provider in one format's ``examples.jsonl``."""
    counts: dict[str, int] = {"claude": 0, "gpt": 0, "gemini": 0, "unknown": 0}
    path = Path(cfg.output_dir) / task_format.value / "examples.jsonl"
    if not path.exists():
        return counts
    for rec in read_jsonl(path):
        meta = rec.get("metadata", {})
        provider = classify_provider(meta.get("construction_model", ""))
        counts[provider] = counts.get(provider, 0) + 1
    return counts


def tally_reserved_provider_counts(registry: BatchRegistry) -> dict[str, int]:
    """Bundles per provider sitting in the registry's in-flight batches.

    Counts batches that are still active (in-flight or completed-but-not-
    yet-ingested) — i.e., bundles whose provider tag will eventually land
    in ``examples.jsonl`` if their results pass the quality filter.
    """
    counts: dict[str, int] = {"claude": 0, "gpt": 0, "gemini": 0, "unknown": 0}
    for b in registry._active():
        label = _REGISTRY_PROVIDER_TO_LABEL.get(b.provider, "unknown")
        counts[label] = counts.get(label, 0) + b.request_count
    return counts


@dataclass
class ProviderCapacity:
    """Per-provider quota accounting for one task format."""

    provider: str
    used: int  # already in examples.jsonl
    reserved: int  # in-flight batches not yet ingested
    cap: int  # raw_target * provider_ratio
    target_ratio: float

    @property
    def projected(self) -> int:
        return self.used + self.reserved

    @property
    def remaining(self) -> int:
        return max(0, self.cap - self.projected)

    @property
    def share_pct(self) -> float:
        return self.target_ratio * 100


def format_provider_capacity(
    cfg: ConstructionConfig,
    task_format: TaskFormat,
) -> dict[str, ProviderCapacity]:
    """Compute per-provider remaining bundle capacity for one format.

    Cap is ``raw_target * provider_ratio`` — raw target accounts for the
    quality filter's expected dropout, so a fully-utilized cap leaves the
    final post-filter share at the configured target ratio.
    """
    used = tally_provider_counts_for_format(cfg, task_format)
    registry = BatchRegistry(cfg.output_dir, task_format.value)
    reserved = tally_reserved_provider_counts(registry)

    raw_target = cfg.raw_target_for(task_format)
    targets = cfg.provider_rotation.targets()

    out: dict[str, ProviderCapacity] = {}
    for prov, ratio in targets.items():
        out[prov] = ProviderCapacity(
            provider=prov,
            used=used.get(prov, 0),
            reserved=reserved.get(prov, 0),
            cap=int(raw_target * ratio),
            target_ratio=ratio,
        )
    return out


def suggest_next_provider(
    current_counts: dict[str, int],
    rotation: ProviderRotationConfig,
) -> str:
    """Pick the provider farthest below its target share.

    Ties broken by the provider's target-share ranking (highest target
    wins). Returns one of ``"claude"``, ``"gpt"``, ``"gemini"``.
    """
    targets = rotation.targets()
    total = sum(current_counts.get(p, 0) for p in targets)  # exclude 'unknown' from denominator
    if total == 0:
        # No history — default to the provider with highest target share.
        return max(targets, key=lambda p: targets[p])

    deficits: dict[str, float] = {}
    for provider, target_ratio in targets.items():
        current_share = current_counts.get(provider, 0) / total
        deficits[provider] = target_ratio - current_share
    # Highest deficit = most underrepresented. Break ties by target share.
    return max(
        targets.keys(),
        key=lambda p: (deficits[p], targets[p]),
    )
