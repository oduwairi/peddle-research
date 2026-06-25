"""Direct OpenAI single-shot runner — owns config A in Arm 1."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime

import openai

from ..schemas import Brief, Inference, UrlScenario

_client: openai.AsyncOpenAI | None = None


def _get_client() -> openai.AsyncOpenAI:
    global _client
    if _client is None:
        _client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


class OpenAIRunner:
    """Single-shot completion against OpenAI (no tools).

    Used for config A in Arm 1 — frontier baseline judged against the
    fine-tuned model on the same brief.
    """

    arm = "arm1"

    def __init__(
        self,
        config_name: str,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self.config_name = config_name
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def run_brief(self, brief: Brief) -> Inference:
        client = _get_client()
        messages = [
            {"role": "system", "content": brief.system},
            {"role": "user", "content": brief.user},
        ]
        started = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                max_completion_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            return Inference(
                example_id=brief.example_id,
                config=self.config_name,
                arm="arm1",
                brief=brief.user,
                system=brief.system,
                assistant_text="",
                latency_ms=int((time.perf_counter() - started) * 1000),
                model_id=self.model,
                error=f"{type(e).__name__}: {e}",
                created_at=datetime.now(UTC),
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        return Inference(
            example_id=brief.example_id,
            config=self.config_name,
            arm="arm1",
            brief=brief.user,
            system=brief.system,
            assistant_text=text,
            latency_ms=latency_ms,
            input_tokens=(usage.prompt_tokens if usage else 0),
            output_tokens=(usage.completion_tokens if usage else 0),
            model_id=self.model,
            created_at=datetime.now(UTC),
        )

    async def run_scenario(self, scenario: UrlScenario) -> Inference:
        # OpenAI single-shot has no tools — Arm 2 routes through FrontendRunner
        # for tool-using configs. This stub exists for symmetry; callers that
        # need single-shot scenario inference can construct a synthetic Brief.
        raise NotImplementedError("OpenAIRunner is single-shot only; use FrontendRunner for Arm 2.")
