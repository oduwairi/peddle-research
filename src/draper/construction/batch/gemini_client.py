"""Gemini Batch API client.

Uses the ``google-genai`` SDK (``google-genai>=1.0``; the legacy
``google-generativeai`` package reached EOL 2025-11-30).

Inline requests are submitted directly — no GCS or Vertex AI required
for payloads under 20 MB (~500 bundles at typical token sizes). Same 50%
batch discount as OpenAI/Anthropic with a 48h completion window.

See https://ai.google.dev/gemini-api/docs/batch-api
"""

from __future__ import annotations

import logging
import os

from google import genai
from google.genai import types as gtypes

from draper.construction.batch.content_blocks import translate_gemini_parts
from draper.construction.batch.types import (
    BatchJobInfo,
    BatchRequest,
    BatchResponse,
    BatchStatus,
)

logger = logging.getLogger("draper")


# Thinking-token budget for Gemini 2.5 Pro / 2.5 Flash.
#
# max_output_tokens on 2.5 thinking models is a HARD cap on (thinking + visible
# response) combined. Observed in production: 2.5 Pro burns 2000-2800 thinking
# tokens silently, so an uncapped config with max_output_tokens=4096 truncates
# ~40% of responses mid-sentence (finish_reason=MAX_TOKENS) before closing tags
# render. Capping thinking at 1024 preserves the reasoning boost over non-
# thinking calls while leaving the caller's requested max_tokens fully
# available for visible output (we pad max_output_tokens by the thinking
# budget so the caller never loses visible-output headroom to thinking).
#
# Set to 0 to disable thinking entirely, -1 to let the model decide (dynamic,
# what we had before and what caused the truncations).
DEFAULT_GEMINI_THINKING_BUDGET = 1024


# Gemini 2.5 Pro produces noticeably shorter rationales than Claude at matched
# max_tokens (observed: median assistant len 1075 vs 1570 chars on copywriting
# pilot). The model isn't hitting the ceiling — it's choosing to compress —
# but a roomier cap removes any upper-bound pressure and gives the model the
# signal that verbose output is permitted. Applied only to Gemini requests so
# other providers keep their cheaper, caller-specified budget.
GEMINI_MAX_TOKENS_MULTIPLIER = 2


# Gemini JobState → normalized BatchStatus.
_STATUS_MAP: dict[str, BatchStatus] = {
    "JOB_STATE_UNSPECIFIED": BatchStatus.PENDING,
    "JOB_STATE_QUEUED": BatchStatus.PENDING,
    "JOB_STATE_PENDING": BatchStatus.PENDING,
    "JOB_STATE_RUNNING": BatchStatus.IN_PROGRESS,
    "JOB_STATE_CANCELLING": BatchStatus.IN_PROGRESS,
    "JOB_STATE_UPDATING": BatchStatus.IN_PROGRESS,
    "JOB_STATE_PAUSED": BatchStatus.IN_PROGRESS,
    "JOB_STATE_SUCCEEDED": BatchStatus.COMPLETED,
    "JOB_STATE_PARTIALLY_SUCCEEDED": BatchStatus.COMPLETED,
    "JOB_STATE_FAILED": BatchStatus.FAILED,
    "JOB_STATE_CANCELLED": BatchStatus.CANCELLED,
    "JOB_STATE_EXPIRED": BatchStatus.EXPIRED,
}

# Gemini uses "model" for assistant turns; our BatchRequest uses "assistant".
_ROLE_MAP = {"assistant": "model", "user": "user"}


def _get_gemini() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


class GeminiBatchClient:
    """Concrete ``BatchClient`` for the Gemini Batch API (inline mode)."""

    provider = "gemini"

    def __init__(
        self,
        client: genai.Client | None = None,
        thinking_budget: int = DEFAULT_GEMINI_THINKING_BUDGET,
    ) -> None:
        self._client: genai.Client | None = client
        self._thinking_budget = thinking_budget
        # Deferred: _get_gemini() only called on first network operation so
        # that tests can construct the client without GEMINI_API_KEY present.

    @property
    def _sdk(self) -> genai.Client:
        if self._client is None:
            self._client = _get_gemini()
        return self._client

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    async def submit(self, requests: list[BatchRequest]) -> BatchJobInfo:
        if not requests:
            msg = "GeminiBatchClient.submit: requests must not be empty"
            raise ValueError(msg)

        inlined = [self._to_inlined(req, thinking_budget=self._thinking_budget) for req in requests]
        batch = await self._sdk.aio.batches.create(
            model=requests[0].model,
            src=inlined,
            config=gtypes.CreateBatchJobConfig(display_name="draper-batch"),
        )
        logger.info(
            "Gemini batch: submitted %s with %d requests (state=%s)",
            batch.name,
            len(requests),
            batch.state,
        )
        return self._to_info(batch, fallback_request_count=len(requests))

    # ------------------------------------------------------------------
    # Poll / cancel
    # ------------------------------------------------------------------

    async def poll(self, batch_id: str) -> BatchJobInfo:
        batch = await self._sdk.aio.batches.get(name=batch_id)
        return self._to_info(batch)

    async def cancel(self, batch_id: str) -> BatchJobInfo:
        # Gemini cancel returns None; poll immediately for updated state.
        await self._sdk.aio.batches.cancel(name=batch_id)
        batch = await self._sdk.aio.batches.get(name=batch_id)
        return self._to_info(batch)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    async def fetch_results(self, batch_id: str) -> list[BatchResponse]:
        batch = await self._sdk.aio.batches.get(name=batch_id)
        state_str = batch.state.value if batch.state else "JOB_STATE_UNSPECIFIED"
        mapped = _STATUS_MAP.get(state_str, BatchStatus.PENDING)
        if mapped != BatchStatus.COMPLETED:
            logger.warning(
                "Gemini batch %s not complete (state=%s); returning empty",
                batch_id,
                state_str,
            )
            return []

        dest = batch.dest
        if dest is None or not dest.inlined_responses:
            logger.warning("Gemini batch %s has no inlined_responses in dest", batch_id)
            return []

        return [self._parse_inlined(r) for r in dest.inlined_responses]

    # ------------------------------------------------------------------
    # Wire-format helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_inlined(
        req: BatchRequest,
        thinking_budget: int = DEFAULT_GEMINI_THINKING_BUDGET,
    ) -> gtypes.InlinedRequest:
        """Convert a provider-agnostic ``BatchRequest`` to a Gemini
        ``InlinedRequest``.

        - System prompt → ``config.system_instruction``
        - Messages → ``contents`` (role "assistant" → "model")
        - ``metadata`` carries the ``custom_id`` for round-trip tracking

        For 2.5 thinking models, ``max_output_tokens`` is a hard cap on
        (thinking + visible) combined. We scale visible output by
        ``GEMINI_MAX_TOKENS_MULTIPLIER`` (Gemini underproduces vs. Claude at
        matched caps), then pad by ``thinking_budget`` so thinking doesn't
        eat into the visible budget, and cap thinking via ``ThinkingConfig``
        so the model can't silently exceed the reservation.
        ``thinking_budget <= 0`` disables padding and thinking config.
        """
        contents: list[gtypes.Content] = []
        for msg in req.messages:
            role = _ROLE_MAP.get(msg["role"], msg["role"])
            content = msg.get("content", "")
            if isinstance(content, list):
                parts: list[gtypes.Part] = []
                for part_kwargs in translate_gemini_parts(content):
                    if "text" in part_kwargs:
                        parts.append(gtypes.Part(text=part_kwargs["text"]))
                    else:
                        parts.append(
                            gtypes.Part.from_uri(
                                file_uri=part_kwargs["file_uri"],
                                mime_type=part_kwargs["mime_type"],
                            )
                        )
                contents.append(gtypes.Content(role=role, parts=parts))
            else:
                contents.append(gtypes.Content(role=role, parts=[gtypes.Part(text=content)]))

        visible_max = req.max_tokens * GEMINI_MAX_TOKENS_MULTIPLIER
        if thinking_budget > 0:
            effective_max = visible_max + thinking_budget
            thinking_cfg = gtypes.ThinkingConfig(thinking_budget=thinking_budget)
        else:
            effective_max = visible_max
            thinking_cfg = None

        cfg_kwargs: dict[str, object] = {
            "temperature": req.temperature,
            "max_output_tokens": effective_max,
        }
        if thinking_cfg is not None:
            cfg_kwargs["thinking_config"] = thinking_cfg
        if req.system:
            cfg_kwargs["system_instruction"] = req.system
        cfg = gtypes.GenerateContentConfig(**cfg_kwargs)  # type: ignore[arg-type]

        return gtypes.InlinedRequest(
            contents=contents,  # type: ignore[arg-type]
            config=cfg,
            metadata={"custom_id": req.custom_id},
        )

    @staticmethod
    def _parse_inlined(item: gtypes.InlinedResponse) -> BatchResponse:
        """Turn a ``InlinedResponse`` into a provider-agnostic
        ``BatchResponse``.
        """
        metadata = item.metadata or {}
        custom_id = metadata.get("custom_id", "")

        if item.error is not None:
            return BatchResponse(
                custom_id=custom_id,
                content="",
                error=str(item.error),
            )

        resp = item.response
        if resp is None:
            return BatchResponse(
                custom_id=custom_id,
                content="",
                error="no response field",
            )

        # Extract text from first candidate's first text part.
        content_text = ""
        candidates = resp.candidates or []
        if candidates:
            parts = candidates[0].content.parts if candidates[0].content else []
            content_text = "".join(p.text for p in (parts or []) if p.text is not None)

        usage = resp.usage_metadata
        in_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        out_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        model_ver = str(getattr(resp, "model_version", "") or "")

        return BatchResponse(
            custom_id=custom_id,
            content=content_text,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            model=model_ver,
        )

    def _to_info(
        self,
        batch: gtypes.BatchJob,
        fallback_request_count: int = 0,
    ) -> BatchJobInfo:
        state_str = batch.state.value if batch.state else "JOB_STATE_UNSPECIFIED"
        status = _STATUS_MAP.get(state_str, BatchStatus.PENDING)

        stats = batch.completion_stats
        succeeded = int(getattr(stats, "successful_count", 0) or 0) if stats else 0
        failed = int(getattr(stats, "failed_count", 0) or 0) if stats else 0
        incomplete = int(getattr(stats, "incomplete_count", 0) or 0) if stats else 0

        # Gemini's completion_stats is frequently null/zero even after the job
        # reaches JOB_STATE_SUCCEEDED. When the job is terminal-completed and
        # stats are empty, count from dest.inlined_responses directly so the
        # CLI doesn't show "0 done / 0 failed" for a finished batch.
        if status == BatchStatus.COMPLETED and succeeded + failed + incomplete == 0:
            dest = batch.dest
            inlined = dest.inlined_responses if dest is not None else None
            if inlined:
                for r in inlined:
                    if r.error is not None or r.response is None:
                        failed += 1
                    else:
                        succeeded += 1

        total = succeeded + failed + incomplete or fallback_request_count
        return BatchJobInfo(
            batch_id=str(batch.name or ""),
            provider=self.provider,
            status=status,
            request_count=total,
            completed_count=succeeded,
            failed_count=failed,
            created_at=str(batch.create_time or ""),
            expires_at=str(batch.end_time or ""),
            raw={"gemini_state": state_str},
        )
