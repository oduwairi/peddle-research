"""Anthropic Message Batches API client.

Uses the Messages Batches endpoint (`client.messages.batches.*`). Same
50% pricing discount as OpenAI with a 24h window. See
https://docs.anthropic.com/en/docs/build-with-claude/batch-processing.
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from draper.construction.batch.content_blocks import translate_anthropic_content
from draper.construction.batch.types import (
    BatchJobInfo,
    BatchRequest,
    BatchResponse,
    BatchStatus,
)
from draper.utils.llm_client import _get_anthropic

logger = logging.getLogger("draper")


# Anthropic processing_status → normalized status.
# Values: in_progress, canceling, ended.
# Once `ended`, per-request result.type = succeeded / errored / canceled / expired.
_STATUS_MAP: dict[str, BatchStatus] = {
    "in_progress": BatchStatus.IN_PROGRESS,
    "canceling": BatchStatus.IN_PROGRESS,
    "ended": BatchStatus.COMPLETED,
}


class AnthropicBatchClient:
    """Concrete `BatchClient` for Anthropic Message Batches."""

    provider = "anthropic"

    def __init__(self, client: anthropic.AsyncAnthropic | None = None) -> None:
        self._client = client or _get_anthropic()

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    async def submit(self, requests: list[BatchRequest]) -> BatchJobInfo:
        if not requests:
            msg = "AnthropicBatchClient.submit: requests must not be empty"
            raise ValueError(msg)

        payload: list[dict[str, Any]] = []
        for req in requests:
            ant_messages: list[dict[str, Any]] = []
            for message in req.messages:
                content = message.get("content", "")
                if isinstance(content, list):
                    ant_messages.append(
                        {
                            "role": message["role"],
                            "content": translate_anthropic_content(content),
                        }
                    )
                else:
                    ant_messages.append({"role": message["role"], "content": content})
            params: dict[str, Any] = {
                "model": req.model,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "messages": ant_messages,
            }
            if req.system:
                params["system"] = req.system
            payload.append({"custom_id": req.custom_id, "params": params})

        batch = await self._client.messages.batches.create(requests=payload)  # type: ignore[arg-type]
        logger.info(
            "Anthropic batch: submitted %s with %d requests (status=%s)",
            batch.id,
            len(requests),
            batch.processing_status,
        )
        return self._to_info(batch, fallback_request_count=len(requests))

    # ------------------------------------------------------------------
    # Poll / cancel
    # ------------------------------------------------------------------

    async def poll(self, batch_id: str) -> BatchJobInfo:
        batch = await self._client.messages.batches.retrieve(batch_id)
        return self._to_info(batch)

    async def cancel(self, batch_id: str) -> BatchJobInfo:
        batch = await self._client.messages.batches.cancel(batch_id)
        return self._to_info(batch)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    async def fetch_results(self, batch_id: str) -> list[BatchResponse]:
        batch = await self._client.messages.batches.retrieve(batch_id)
        mapped_status = _STATUS_MAP.get(batch.processing_status, BatchStatus.PENDING)
        if mapped_status != BatchStatus.COMPLETED:
            logger.warning(
                "Anthropic batch %s not ended (status=%s); returning empty",
                batch_id,
                batch.processing_status,
            )
            return []

        results: list[BatchResponse] = []
        iterator = await self._client.messages.batches.results(batch_id)
        async for item in iterator:
            results.append(self._parse_result(item))
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_result(item: Any) -> BatchResponse:
        """Turn a `MessageBatchIndividualResponse` into a BatchResponse."""
        custom_id = getattr(item, "custom_id", "")
        result = getattr(item, "result", None)
        if result is None:
            return BatchResponse(custom_id=custom_id, content="", error="no result field")
        result_type = getattr(result, "type", "unknown")
        if result_type == "succeeded":
            message = getattr(result, "message", None)
            content_text = ""
            model_name = ""
            in_tokens = 0
            out_tokens = 0
            if message is not None:
                blocks = getattr(message, "content", []) or []
                parts: list[str] = []
                for block in blocks:
                    text = getattr(block, "text", None)
                    if text:
                        parts.append(text)
                content_text = "".join(parts)
                model_name = getattr(message, "model", "") or ""
                usage = getattr(message, "usage", None)
                if usage is not None:
                    in_tokens = int(getattr(usage, "input_tokens", 0) or 0)
                    out_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            return BatchResponse(
                custom_id=custom_id,
                content=content_text,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                model=model_name,
            )
        # errored / canceled / expired all surface through `error`
        error_payload = getattr(result, "error", None) or result_type
        return BatchResponse(
            custom_id=custom_id,
            content="",
            error=str(error_payload),
        )

    def _to_info(
        self,
        batch: Any,
        fallback_request_count: int = 0,
    ) -> BatchJobInfo:
        counts = getattr(batch, "request_counts", None)
        processing = int(getattr(counts, "processing", 0) or 0) if counts else 0
        succeeded = int(getattr(counts, "succeeded", 0) or 0) if counts else 0
        errored = int(getattr(counts, "errored", 0) or 0) if counts else 0
        canceled = int(getattr(counts, "canceled", 0) or 0) if counts else 0
        expired = int(getattr(counts, "expired", 0) or 0) if counts else 0
        total = processing + succeeded + errored + canceled + expired
        return BatchJobInfo(
            batch_id=batch.id,
            provider=self.provider,
            status=_STATUS_MAP.get(batch.processing_status, BatchStatus.PENDING),
            request_count=total or fallback_request_count,
            completed_count=succeeded,
            failed_count=errored + canceled + expired,
            created_at=str(getattr(batch, "created_at", "") or ""),
            expires_at=str(getattr(batch, "expires_at", "") or ""),
            raw={"anthropic_status": batch.processing_status},
        )
