"""Unit tests for `validate_batch_model` (factory whitelist)."""

from __future__ import annotations

import pytest

from draper.construction.batch.factory import (
    _BATCH_DENYLIST,
    provider_for_model,
    validate_batch_model,
)


def test_validate_batch_model_accepts_known_providers() -> None:
    """A model whose prefix maps to a known provider passes."""
    validate_batch_model("claude-sonnet-4-6")
    validate_batch_model("claude-haiku-4-5")
    validate_batch_model("gpt-5.4-mini")
    validate_batch_model("gemini-3.1-pro-preview")


def test_validate_batch_model_unknown_prefix_raises() -> None:
    """Models without a recognized provider prefix fail at lookup."""
    with pytest.raises(ValueError, match="Cannot determine batch provider"):
        validate_batch_model("mistral-large")


def test_validate_batch_model_honors_denylist(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding a model to the denylist makes validation reject it."""
    # Patch a single entry instead of mutating the module-level frozenset.
    patched = dict(_BATCH_DENYLIST)
    patched["openai"] = frozenset({"gpt-5.4"})
    monkeypatch.setattr("draper.construction.batch.factory._BATCH_DENYLIST", patched)
    with pytest.raises(ValueError, match="not eligible for the openai batch API"):
        validate_batch_model("gpt-5.4")
    # gpt-5.4-mini stays valid.
    validate_batch_model("gpt-5.4-mini")


def test_provider_for_model_round_trip() -> None:
    """provider_for_model agrees with validate_batch_model's mapping."""
    assert provider_for_model("claude-sonnet-4-6") == "anthropic"
    assert provider_for_model("gpt-5.4-mini") == "openai"
    assert provider_for_model("gemini-3.1-pro-preview") == "gemini"
