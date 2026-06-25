"""Thin async wrapper around Anthropic and OpenAI SDKs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import anthropic
import openai
from dotenv import load_dotenv
from google import genai
from google.genai import types as gtypes

# Thinking-token budget for Gemini 2.5/3.x thinking models. Kept in sync with
# construction.batch.gemini_client.DEFAULT_GEMINI_THINKING_BUDGET — duplicated
# here to keep llm_client at the bottom of the import DAG (construction.batch
# already depends on llm_client for Anthropic/OpenAI client getters).
DEFAULT_GEMINI_THINKING_BUDGET = 1024

load_dotenv()

_anthropic_client: anthropic.AsyncAnthropic | None = None
_openai_client: openai.AsyncOpenAI | None = None
_openrouter_client: openai.AsyncOpenAI | None = None
_gemini_client: genai.Client | None = None


def _get_anthropic() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
    return _anthropic_client


def _get_openai() -> openai.AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
        )
    return _openai_client


_OPENROUTER_PREFIXES = (
    "meta-llama/",
    "mistralai/",
    "qwen/",
    "google/",
    "deepseek/",
    "microsoft/",
    "nvidia/",
)


def _get_openrouter() -> openai.AsyncOpenAI:
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    return _openrouter_client


def _get_gemini() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _gemini_client


def _is_anthropic_model(model: str) -> bool:
    return model.startswith("claude")


def _is_gemini_model(model: str) -> bool:
    return model.startswith("gemini")


def _is_openrouter_model(model: str) -> bool:
    return any(model.startswith(p) for p in _OPENROUTER_PREFIXES)


_GEMINI_ROLE_MAP = {"assistant": "model", "user": "user"}


async def _complete_gemini_native(
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int,
    temperature: float,
    system: str | None,
) -> tuple[str, int, int, str]:
    """Call Gemini via the native google-genai SDK.

    Mirrors the batch-client thinking-budget handling: for 2.5/3.x thinking
    models, ``max_output_tokens`` is a hard cap on (thinking + visible), so
    we pad by ``DEFAULT_GEMINI_THINKING_BUDGET`` and explicitly cap thinking
    via ``ThinkingConfig`` to preserve the caller's visible-output budget.
    """
    contents: list[gtypes.Content] = [
        gtypes.Content(
            role=_GEMINI_ROLE_MAP.get(msg["role"], msg["role"]),
            parts=[gtypes.Part(text=msg["content"])],
        )
        for msg in messages
    ]

    if DEFAULT_GEMINI_THINKING_BUDGET > 0:
        effective_max = max_tokens + DEFAULT_GEMINI_THINKING_BUDGET
        thinking_cfg: gtypes.ThinkingConfig | None = gtypes.ThinkingConfig(
            thinking_budget=DEFAULT_GEMINI_THINKING_BUDGET,
        )
    else:
        effective_max = max_tokens
        thinking_cfg = None

    cfg_kwargs: dict[str, object] = {
        "temperature": temperature,
        "max_output_tokens": effective_max,
    }
    if thinking_cfg is not None:
        cfg_kwargs["thinking_config"] = thinking_cfg
    if system:
        cfg_kwargs["system_instruction"] = system
    cfg = gtypes.GenerateContentConfig(**cfg_kwargs)  # type: ignore[arg-type]

    client = _get_gemini()
    resp = await client.aio.models.generate_content(
        model=model,
        contents=contents,  # type: ignore[arg-type]
        config=cfg,
    )

    content_text = ""
    candidates = resp.candidates or []
    if candidates and candidates[0].content:
        parts = candidates[0].content.parts or []
        content_text = "".join(p.text for p in parts if p.text is not None)

    usage = resp.usage_metadata
    in_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    out_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    model_ver = str(getattr(resp, "model_version", "") or model)
    return content_text, in_tokens, out_tokens, model_ver


@dataclass(frozen=True)
class CompletionResult:
    """Response from ``complete_with_usage`` including token counts."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str


async def complete_with_usage(
    messages: list[dict[str, str]],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    system: str | None = None,
    **kwargs: Any,
) -> CompletionResult:
    """Like ``complete`` but returns token usage alongside content."""
    if _is_anthropic_model(model):
        client = _get_anthropic()
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
            **kwargs,
        }
        if system:
            params["system"] = system
        response = await client.messages.create(**params)
        return CompletionResult(
            content=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=model,
        )
    if _is_gemini_model(model):
        content, in_tok, out_tok, model_ver = await _complete_gemini_native(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
        )
        return CompletionResult(
            content=content,
            input_tokens=in_tok,
            output_tokens=out_tok,
            model=model_ver,
        )
    # OpenRouter or OpenAI path
    is_openrouter = _is_openrouter_model(model)
    client_oai = _get_openrouter() if is_openrouter else _get_openai()
    oai_messages: list[dict[str, str]] = []
    if system:
        oai_messages.append({"role": "system", "content": system})
    oai_messages.extend(messages)
    token_param = "max_tokens" if is_openrouter else "max_completion_tokens"
    response_oai = await client_oai.chat.completions.create(  # type: ignore[call-overload]
        model=model,
        messages=oai_messages,
        temperature=temperature,
        **{token_param: max_tokens},
        **kwargs,
    )
    usage = response_oai.usage
    return CompletionResult(
        content=response_oai.choices[0].message.content or "",
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        model=model,
    )


async def complete(
    messages: list[dict[str, str]],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    system: str | None = None,
    **kwargs: Any,
) -> str:
    """Send a completion request to the appropriate provider.

    Args:
        messages: List of {"role": "user"|"assistant", "content": "..."} dicts.
        model: Model identifier. Claude models route to Anthropic, others to OpenAI.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
        system: System prompt (Anthropic-style). For OpenAI, prepended as a system message.
        **kwargs: Additional provider-specific parameters.

    Returns:
        The assistant's response text.
    """
    if _is_anthropic_model(model):
        return await _complete_anthropic(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            **kwargs,
        )
    if _is_gemini_model(model):
        content, _, _, _ = await _complete_gemini_native(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
        )
        return content
    return await _complete_openai(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        client_override=_get_openrouter() if _is_openrouter_model(model) else None,
        **kwargs,
    )


async def _complete_anthropic(
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int,
    temperature: float,
    system: str | None,
    **kwargs: Any,
) -> str:
    client = _get_anthropic()
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
        **kwargs,
    }
    if system:
        params["system"] = system
    response = await client.messages.create(**params)
    block = response.content[0]
    return block.text  # type: ignore[no-any-return]


async def _complete_openai(
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int,
    temperature: float,
    system: str | None,
    client_override: openai.AsyncOpenAI | None = None,
    **kwargs: Any,
) -> str:
    client = client_override or _get_openai()
    is_openrouter = client_override is not None
    oai_messages: list[dict[str, str]] = []
    if system:
        oai_messages.append({"role": "system", "content": system})
    oai_messages.extend(messages)
    token_param = "max_tokens" if is_openrouter else "max_completion_tokens"
    response = await client.chat.completions.create(  # type: ignore[call-overload]
        model=model,
        messages=oai_messages,
        temperature=temperature,
        **{token_param: max_tokens},
        **kwargs,
    )
    content: str | None = response.choices[0].message.content
    if content is None:
        return ""
    return content
