"""Frontend pipeline runner — drives the Next.js multi-stage agent.

Owns configs A_pipe (GPT-5.5 + tools), C_pipe (FT 7B + tools), and B_pipe (base Qwen + tools).
Each config corresponds to a separately-launched frontend process with
``MODEL_ID`` / ``OPENAI_BASE_URL`` set for that backend; we POST to the
service-token-gated ``/api/eval/run`` route on the appropriate port.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from ..schemas import Brief, Inference, UrlScenario


class FrontendRunner:
    """POST briefs/scenarios to a frontend ``/api/eval/run`` endpoint."""

    arm = "arm2"

    def __init__(
        self,
        config_name: str,
        base_url_env: str,
        token_env: str = "EVAL_SERVICE_TOKEN",
        timeout_s: int = 180,
    ) -> None:
        self.config_name = config_name
        self.base_url = os.environ.get(base_url_env, "http://localhost:3000").rstrip("/")
        self.token = os.environ.get(token_env, "")
        if not self.token:
            raise RuntimeError(
                f"FrontendRunner requires {token_env} to be set (eval service token)."
            )
        self.timeout_s = timeout_s

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/api/eval/run"
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self.token}"},
            )
            resp.raise_for_status()
            data = resp.json()
            assert isinstance(data, dict)
            return data

    async def run_brief(self, brief: Brief) -> Inference:
        # In Arm 2 we run *some* briefs through the pipeline as a sanity
        # check; the user prompt is just the brief itself.
        started = time.perf_counter()
        payload = {
            "userPrompt": brief.user,
            "platform": brief.platform,
            "exampleId": brief.example_id,
        }
        try:
            data = await self._post(payload)
        except Exception as e:
            return Inference(
                example_id=brief.example_id,
                config=self.config_name,
                arm="arm2",
                brief=brief.user,
                system=brief.system,
                assistant_text="",
                latency_ms=int((time.perf_counter() - started) * 1000),
                model_id=self.config_name,
                error=f"{type(e).__name__}: {e}",
                created_at=datetime.now(UTC),
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        return _from_envelope(
            envelope=data,
            example_id=brief.example_id,
            config_name=self.config_name,
            brief_text=brief.user,
            system=brief.system,
            latency_ms=latency_ms,
        )

    async def run_scenario(self, scenario: UrlScenario) -> Inference:
        started = time.perf_counter()
        payload = {
            "userPrompt": scenario.user_prompt,
            "platform": scenario.platform,
            "exampleId": scenario.scenario_id,
        }
        try:
            data = await self._post(payload)
        except Exception as e:
            return Inference(
                example_id=scenario.scenario_id,
                config=self.config_name,
                arm="arm2",
                brief=scenario.user_prompt,
                system="",
                assistant_text="",
                latency_ms=int((time.perf_counter() - started) * 1000),
                model_id=self.config_name,
                error=f"{type(e).__name__}: {e}",
                created_at=datetime.now(UTC),
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        return _from_envelope(
            envelope=data,
            example_id=scenario.scenario_id,
            config_name=self.config_name,
            brief_text=scenario.user_prompt,
            system="",
            latency_ms=latency_ms,
        )


def _from_envelope(
    envelope: dict[str, Any],
    example_id: str,
    config_name: str,
    brief_text: str,
    system: str,
    latency_ms: int,
) -> Inference:
    """Convert the JSON envelope returned by /api/eval/run into an Inference."""
    campaign = envelope.get("campaign")
    assistant = str(envelope.get("assistantText") or "")
    # When a campaign is emitted, prefer its primary copy block as the text
    # the judge will compare against — that's what an end-user would see.
    if campaign:
        primary = _primary_copy_from_campaign(campaign)
        if primary:
            assistant = primary
    traces = envelope.get("traces")
    return Inference(
        example_id=example_id,
        config=config_name,
        arm="arm2",
        brief=brief_text,
        system=system,
        assistant_text=assistant,
        tool_calls=_tool_calls_from_traces(traces),
        raw_traces=traces if isinstance(traces, list) else None,
        campaign=campaign if isinstance(campaign, dict) else None,
        latency_ms=latency_ms,
        input_tokens=int(envelope.get("inputTokens") or 0),
        output_tokens=int(envelope.get("outputTokens") or 0),
        model_id=str(envelope.get("modelId") or config_name),
        created_at=datetime.now(UTC),
    )


def _primary_copy_from_campaign(campaign: dict[str, Any]) -> str | None:
    """Pull the primary copy block out of a CampaignOutput payload.

    The frontend emits one of several platform-shaped payloads (Meta has
    headline+body, X has tweet_body, TikTok has hook_line+caption, etc.).
    Concatenate the visible copy fields in display order so the judge sees
    the same thing a buyer would.
    """
    parts: list[str] = []
    for key in ("headline", "hook_line", "tweet_body", "body", "caption", "cta"):
        val = campaign.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return "\n\n".join(parts) if parts else None


def _tool_calls_from_traces(traces: Any) -> list[dict[str, Any]]:
    """Extract tool-call summaries from agent_trace rows for analysis."""
    if not isinstance(traces, list):
        return []
    out: list[dict[str, Any]] = []
    for t in traces:
        if not isinstance(t, dict):
            continue
        tool_calls = t.get("toolCalls")
        if tool_calls:
            out.append(
                {
                    "stage": t.get("stage"),
                    "stepIndex": t.get("stepIndex"),
                    "toolCalls": tool_calls,
                    "finishReason": t.get("finishReason"),
                }
            )
    return out
