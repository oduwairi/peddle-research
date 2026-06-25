"""vLLM single-shot runner — owns configs B / C (v1) and B_v2 / C_v2 (v2).

vLLM exposes an OpenAI-compatible /v1/chat/completions endpoint. We reuse
the AsyncOpenAI client with a custom ``base_url`` so the same code path
serves the base model and the fine-tuned LoRA on a single vLLM process
(switched per-request via the ``model`` field).

Config mapping:
  * ``B``    — base Qwen3-8B via OpenRouter (v1 split, no fine-tune)
  * ``C``    — Draper-r16 fine-tune via Modal vLLM (v1 split, no tools)
  * ``B_v2`` — base Qwen3-8B via OpenRouter (v2 split, no fine-tune)
  * ``C_v2`` — Draper v2 Qwen3-8B fine-tune (v2 split, no tools; max_tokens=2048
               to accommodate the ``<think>`` block before the deliverable)
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime

import openai

from ..schemas import Brief, Inference, UrlScenario


class VLLMRunner:
    """Single-shot completion against a local/remote vLLM endpoint."""

    arm = "arm1"

    def __init__(
        self,
        config_name: str,
        model: str,
        base_url_env: str = "VLLM_BASE_URL",
        api_key_env: str = "VLLM_API_KEY",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self.config_name = config_name
        self.model = model
        # Default to a permissive local URL — overridden by env when serving
        # remote (RunPod / Vast.ai exposes a public HTTPS endpoint).
        base_url = os.environ.get(base_url_env, "http://localhost:8000/v1")
        # vLLM does not require a real key but the OpenAI client insists on
        # one. Default "EMPTY" matches vLLM's docs.
        api_key = os.environ.get(api_key_env, "EMPTY")
        self._client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def run_brief(self, brief: Brief) -> Inference:
        messages = [
            {"role": "system", "content": brief.system},
            {"role": "user", "content": brief.user},
        ]
        started = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=self.max_tokens,  # vLLM uses max_tokens, not max_completion_tokens
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
        text = resp.choices[0].message.content or ""
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
        raise NotImplementedError(
            "VLLMRunner is single-shot only; Arm 2 routes through FrontendRunner."
        )
