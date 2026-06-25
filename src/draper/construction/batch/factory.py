"""Batch-client factory: model name → `BatchClient` + provider key.

Centralizing the mapping here means the CLI can accept any supported
model name (``gpt-4o-mini``, ``claude-sonnet-4-6``, etc.) and the rest
of the pipeline stays provider-agnostic.

:func:`validate_batch_model` is the single chokepoint where the CLI
asserts a model is batch-eligible BEFORE any network round trip — so
a misconfigured ``--model`` lands as a clean error rather than after
the batch upload silently stalls in ``validating``.
"""

from __future__ import annotations

from draper.construction.batch.anthropic_client import AnthropicBatchClient
from draper.construction.batch.gemini_client import GeminiBatchClient
from draper.construction.batch.openai_client import OpenAIBatchClient
from draper.construction.batch.types import BatchClient

# Prefixes identifying each provider. The order matters only for
# diagnostics — `provider_for_model` short-circuits on first match.
_OPENAI_PREFIXES: tuple[str, ...] = (
    "gpt-",
    "gpt4",
    "o1",
    "o3",
    "o4",
    "chatgpt",
    "text-",
)
_ANTHROPIC_PREFIXES: tuple[str, ...] = ("claude",)
_GEMINI_PREFIXES: tuple[str, ...] = ("gemini",)

# Models known NOT to be eligible for batch APIs at the listed
# provider. Empty for now; add entries here when a model is verified
# chat-only or otherwise unsupported by the batch endpoint. Keys are
# logical provider names returned by ``provider_for_model``.
_BATCH_DENYLIST: dict[str, frozenset[str]] = {
    "openai": frozenset(),
    "anthropic": frozenset(),
    "gemini": frozenset(),
}


def provider_for_model(model: str) -> str:
    """Return ``"openai"``, ``"anthropic"``, or ``"gemini"`` for a model."""
    lowered = model.lower()
    if any(lowered.startswith(p) for p in _ANTHROPIC_PREFIXES):
        return "anthropic"
    if any(lowered.startswith(p) for p in _OPENAI_PREFIXES):
        return "openai"
    if any(lowered.startswith(p) for p in _GEMINI_PREFIXES):
        return "gemini"
    msg = (
        f"Cannot determine batch provider for model '{model}'. "
        f"Supported prefixes: gpt-*, claude-*, gemini-*."
    )
    raise ValueError(msg)


def validate_batch_model(model: str) -> None:
    """Raise ``ValueError`` if ``model`` is not eligible for batch.

    Call this from any code path that submits a batch BEFORE the
    network round trip. The check is intentionally cheap: it asserts
    the model maps to a known provider and is not in the per-provider
    denylist. Adding a model to ``_BATCH_DENYLIST`` is the way to
    block a known-chat-only model at submit time.
    """
    provider = provider_for_model(model)
    denied = _BATCH_DENYLIST.get(provider, frozenset())
    if model in denied:
        msg = (
            f"Model '{model}' is not eligible for the {provider} batch "
            f"API. Pick a batch-eligible model from the {provider} "
            f"docs (e.g. 'claude-sonnet-4-6', 'gpt-5.4-mini', "
            f"'gemini-3.1-pro-preview')."
        )
        raise ValueError(msg)


def make_batch_client(model: str) -> BatchClient:
    """Return the right `BatchClient` for a given model name."""
    provider = provider_for_model(model)
    if provider == "openai":
        return OpenAIBatchClient()
    if provider == "anthropic":
        return AnthropicBatchClient()
    return GeminiBatchClient()
