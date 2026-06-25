"""OpenAI Batch API client.

Uses the `/v1/files` + `/v1/batches` endpoints. ~50% cheaper than sync
calls with a 24h completion window. See
https://platform.openai.com/docs/guides/batch for the wire format.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import openai

from draper.construction.batch.content_blocks import translate_openai_content
from draper.construction.batch.types import (
    BatchJobInfo,
    BatchRequest,
    BatchResponse,
    BatchStatus,
)
from draper.utils.llm_client import _get_openai

logger = logging.getLogger("draper")


# Provider statuses → normalized statuses.
# OpenAI: validating, failed, in_progress, finalizing, completed,
# expired, cancelling, cancelled.
_STATUS_MAP: dict[str, BatchStatus] = {
    "validating": BatchStatus.PENDING,
    "in_progress": BatchStatus.IN_PROGRESS,
    "finalizing": BatchStatus.IN_PROGRESS,
    "completed": BatchStatus.COMPLETED,
    "failed": BatchStatus.FAILED,
    "cancelling": BatchStatus.IN_PROGRESS,
    "cancelled": BatchStatus.CANCELLED,
    "expired": BatchStatus.EXPIRED,
}


class OpenAIBatchClient:
    """Concrete `BatchClient` for OpenAI's Batch API."""

    provider = "openai"

    def __init__(self, client: openai.AsyncOpenAI | None = None) -> None:
        self._client = client or _get_openai()

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    async def submit(self, requests: list[BatchRequest]) -> BatchJobInfo:
        if not requests:
            msg = "OpenAIBatchClient.submit: requests must not be empty"
            raise ValueError(msg)

        jsonl_bytes = self._render_input_jsonl(requests).encode("utf-8")
        # OpenAI expects a file-like object; wrap as a named BytesIO so
        # the SDK can detect the content type without touching disk.
        file_like = io.BytesIO(jsonl_bytes)
        file_like.name = "draper_batch.jsonl"

        file_obj = await self._client.files.create(
            file=file_like,
            purpose="batch",
        )
        logger.info("OpenAI batch: uploaded input file %s", file_obj.id)

        batch = await self._client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"draper_source": "construction_pipeline"},
        )
        logger.info(
            "OpenAI batch: submitted %s with %d requests (status=%s)",
            batch.id,
            len(requests),
            batch.status,
        )
        return self._to_info(batch, fallback_request_count=len(requests))

    # ------------------------------------------------------------------
    # Poll / cancel
    # ------------------------------------------------------------------

    async def poll(self, batch_id: str) -> BatchJobInfo:
        batch = await self._client.batches.retrieve(batch_id)
        return self._to_info(batch)

    async def cancel(self, batch_id: str) -> BatchJobInfo:
        batch = await self._client.batches.cancel(batch_id)
        return self._to_info(batch)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    async def fetch_results(self, batch_id: str) -> list[BatchResponse]:
        batch = await self._client.batches.retrieve(batch_id)
        mapped_status = _STATUS_MAP.get(batch.status, BatchStatus.PENDING)
        if mapped_status != BatchStatus.COMPLETED:
            logger.warning(
                "OpenAI batch %s not completed (status=%s); returning empty",
                batch_id,
                batch.status,
            )
            return []

        responses: list[BatchResponse] = []

        # Successful lines live in output_file_id. Parse failures live in
        # error_file_id — we surface both as BatchResponse with error set.
        output_file_id = getattr(batch, "output_file_id", None)
        if output_file_id:
            content = await self._client.files.content(output_file_id)
            responses.extend(self._parse_output_jsonl(content.text))

        error_file_id = getattr(batch, "error_file_id", None)
        if error_file_id:
            content = await self._client.files.content(error_file_id)
            responses.extend(self._parse_error_jsonl(content.text))

        return responses

    # ------------------------------------------------------------------
    # Wire-format helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _render_input_jsonl(requests: list[BatchRequest]) -> str:
        """Render the batch input JSONL per OpenAI's spec.

        Each line is one request:
        ``{"custom_id": ..., "method": "POST", "url": "/v1/chat/completions",
        "body": {"model": ..., "messages": [...], ...}}``
        """
        lines: list[str] = []
        for req in requests:
            oai_messages: list[dict[str, Any]] = []
            if req.system:
                oai_messages.append({"role": "system", "content": req.system})
            for msg in req.messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    oai_messages.append(
                        {"role": msg["role"], "content": translate_openai_content(content)}
                    )
                else:
                    oai_messages.append({"role": msg["role"], "content": content})
            body: dict[str, object] = {
                "model": req.model,
                "messages": oai_messages,
                "max_completion_tokens": req.max_tokens,
            }
            # Reasoning-family models (gpt-5.x, o1, o3) reject any
            # non-default temperature with HTTP 400. Omit the field for
            # these families and let the API apply its default.
            lowered = req.model.lower()
            is_reasoning = (
                lowered.startswith("o1") or lowered.startswith("o3") or lowered.startswith("gpt-5")
            )
            if not is_reasoning:
                body["temperature"] = req.temperature
            lines.append(
                json.dumps(
                    {
                        "custom_id": req.custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    },
                    ensure_ascii=False,
                )
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _parse_output_jsonl(text: str) -> list[BatchResponse]:
        """Parse `output_file` JSONL from a completed batch."""
        out: list[BatchResponse] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed output line: %s", exc)
                continue
            custom_id = str(obj.get("custom_id", ""))
            error_payload = obj.get("error")
            if error_payload:
                out.append(
                    BatchResponse(
                        custom_id=custom_id,
                        content="",
                        error=str(error_payload),
                    )
                )
                continue
            response = obj.get("response") or {}
            body = response.get("body") or {}
            choices = body.get("choices") or []
            content = ""
            if choices:
                content = choices[0].get("message", {}).get("content") or ""
            usage = body.get("usage") or {}
            out.append(
                BatchResponse(
                    custom_id=custom_id,
                    content=content,
                    input_tokens=int(usage.get("prompt_tokens", 0)),
                    output_tokens=int(usage.get("completion_tokens", 0)),
                    model=str(body.get("model", "")),
                )
            )
        return out

    @staticmethod
    def _parse_error_jsonl(text: str) -> list[BatchResponse]:
        """Parse `error_file` JSONL — each line is a per-request failure."""
        out: list[BatchResponse] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.append(
                BatchResponse(
                    custom_id=str(obj.get("custom_id", "")),
                    content="",
                    error=json.dumps(obj.get("error") or obj),
                )
            )
        return out

    def _to_info(
        self,
        batch: Any,
        fallback_request_count: int = 0,
    ) -> BatchJobInfo:
        """Convert an OpenAI ``Batch`` object into our normalized info type."""
        counts = getattr(batch, "request_counts", None)
        total = int(getattr(counts, "total", 0)) if counts else 0
        completed = int(getattr(counts, "completed", 0)) if counts else 0
        failed = int(getattr(counts, "failed", 0)) if counts else 0
        return BatchJobInfo(
            batch_id=batch.id,
            provider=self.provider,
            status=_STATUS_MAP.get(batch.status, BatchStatus.PENDING),
            request_count=total or fallback_request_count,
            completed_count=completed,
            failed_count=failed,
            created_at=str(getattr(batch, "created_at", "") or ""),
            expires_at=str(getattr(batch, "expires_at", "") or ""),
            error=str(getattr(batch, "errors", "") or ""),
            raw={"openai_status": batch.status},
        )
