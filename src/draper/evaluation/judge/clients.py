"""Shared provider clients, predicates, and utilities for all judge callers.

All three judge modules (pairwise, validation, batch) needed the same lazy
client singletons, model-prefix predicates, schema stripper, score clipper,
and provider classifier. This module is the single source of truth for all of
them.

Token-budget constants
-----------------------
The three providers have meaningfully different token needs for judge calls:

  - OpenAI (``OPENAI_MAX_TOKENS``): 512 tokens covers the JSON body with room
    to spare. OpenAI's strict json_schema mode guarantees valid JSON output so
    truncation is very unlikely.

  - Anthropic (``ANTHROPIC_MAX_TOKENS``): 1024. The tool-use path generates an
    outer message wrapper around the tool-input payload; the model also tends
    to add brief in-context reasoning before the tool call even when
    temperature=0. Extra headroom prevents truncation of the schema fields.

  - Gemini (``GEMINI_MAX_OUTPUT_TOKENS``): 2048. Gemini 2.5 Pro / Flash spend
    a variable number of tokens on internal reasoning ("thinking tokens") that
    count against ``max_output_tokens`` before the visible JSON body appears.
    Too low a budget causes the JSON body to be silently truncated, producing
    a parse error. 2048 is the empirically safe floor for 5-dimension judgments.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import anthropic
import openai
from google import genai

# ---------------------------------------------------------------------------
# Token budget constants
# ---------------------------------------------------------------------------

OPENAI_MAX_TOKENS: int = 512
"""OpenAI judge calls: strict json_schema mode; 512 is ample."""

ANTHROPIC_MAX_TOKENS: int = 1024
"""Anthropic judge calls: tool-use path; 1024 covers wrapper + reasoning."""

GEMINI_MAX_OUTPUT_TOKENS: int = 2048
"""Gemini judge calls: reasoning tokens compete for this budget; 2048 is safe."""

# ---------------------------------------------------------------------------
# Lazy-init client singletons
# ---------------------------------------------------------------------------

_openai_client: openai.AsyncOpenAI | None = None
_gemini_client: genai.Client | None = None
_anthropic_client: anthropic.AsyncAnthropic | None = None


def openai_client() -> openai.AsyncOpenAI:
    """Return a module-level AsyncOpenAI singleton (lazy-init)."""
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def gemini_client() -> genai.Client:
    """Return a module-level Gemini client singleton (lazy-init)."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _gemini_client


def anthropic_client() -> anthropic.AsyncAnthropic:
    """Return a module-level AsyncAnthropic singleton (lazy-init)."""
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


# ---------------------------------------------------------------------------
# Model-prefix predicates
# ---------------------------------------------------------------------------


def is_gemini(model: str) -> bool:
    """True when the model string routes to the Google Gemini provider."""
    return model.startswith("gemini")


def is_claude(model: str) -> bool:
    """True when the model string routes to the Anthropic provider."""
    return model.startswith("claude")


# ---------------------------------------------------------------------------
# Provider classifier (batch routing)
# ---------------------------------------------------------------------------

BatchProvider = Literal["openai", "anthropic"]


def provider_for_model(judge_model: str) -> BatchProvider:
    """Map a judge model string to its batch provider.

    Gemini intentionally raises — we don't ship a Gemini batch path because
    it lacks a flat 50% discount and the API shape is dissimilar. Use the live
    (sync) judge path for Gemini-based panel members.
    """
    if is_claude(judge_model):
        return "anthropic"
    if is_gemini(judge_model):
        raise ValueError(
            f"Gemini models ({judge_model!r}) are not supported by batch eval; "
            "use the live judge path for them."
        )
    return "openai"


# ---------------------------------------------------------------------------
# Schema utilities
# ---------------------------------------------------------------------------


def gemini_compat_schema(schema: Any) -> Any:
    """Strip OpenAPI keywords Gemini's response_schema doesn't accept.

    Gemini accepts a strict subset of JSON Schema; ``additionalProperties``,
    ``$schema``, and a few siblings raise INVALID_ARGUMENT. Recurse and prune.
    """
    drop = {"additionalProperties", "$schema", "definitions", "$defs"}
    if isinstance(schema, dict):
        return {k: gemini_compat_schema(v) for k, v in schema.items() if k not in drop}
    if isinstance(schema, list):
        return [gemini_compat_schema(v) for v in schema]
    return schema


# ---------------------------------------------------------------------------
# Score utilities
# ---------------------------------------------------------------------------


def clip_score(x: int) -> int:
    """Clip a per-dimension judge score to the valid range [-2, +2]."""
    return max(-2, min(2, int(x)))
