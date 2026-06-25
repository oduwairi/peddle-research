"""Provider-agnostic batch transport types and client protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class BatchStatus(StrEnum):
    """Normalized batch lifecycle states.

    Provider-specific statuses are mapped into this enum so downstream
    code (registry, CLI) stays provider-agnostic.
    """

    PENDING = "pending"  # Submitted, awaiting processing
    IN_PROGRESS = "in_progress"  # Validating or running
    COMPLETED = "completed"  # All requests finished (partial failures allowed)
    FAILED = "failed"  # Whole batch failed
    CANCELLED = "cancelled"  # User-initiated cancel
    EXPIRED = "expired"  # Provider timeout (e.g. 24h for OpenAI)


@dataclass(frozen=True)
class BatchRequest:
    """Single request within a batch submission.

    ``custom_id`` must be unique per batch and is the only link between
    the request and its sidecar metadata when results come back. Each
    provider echoes the custom_id in its output.

    Each message is ``{"role": ..., "content": <str | list[ContentBlock]>}``.
    A ``str`` content stays text-only (the legacy shape, unchanged).
    A ``list`` content is a sequence of internal ContentBlock dicts; each
    provider client translates them into its own wire format. Two block
    forms are defined:

    - ``{"type": "text", "text": str}``
    - ``{"type": "image_url", "url": str, "mime_type": str | None}``
      (``mime_type`` defaults to ``"image/jpeg"`` when omitted)
    """

    custom_id: str
    system: str | None
    messages: list[dict[str, Any]]
    model: str
    max_tokens: int = 4096
    temperature: float = 0.7


@dataclass(frozen=True)
class BatchResponse:
    """Single response line from a completed batch.

    ``error`` is empty on success; on failure ``content`` is empty and
    ``error`` carries the provider's error message. Callers should check
    ``error`` before parsing ``content``.
    """

    custom_id: str
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    error: str = ""


@dataclass
class BatchJobInfo:
    """Polling snapshot of a batch submission."""

    batch_id: str
    provider: str
    status: BatchStatus
    request_count: int
    completed_count: int = 0
    failed_count: int = 0
    created_at: str = ""
    expires_at: str = ""
    error: str = ""
    raw: dict[str, object] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """True once the batch will not change state further."""
        return self.status in (
            BatchStatus.COMPLETED,
            BatchStatus.FAILED,
            BatchStatus.CANCELLED,
            BatchStatus.EXPIRED,
        )


class BatchClient(Protocol):
    """Provider-agnostic batch API client.

    Implementations:

    - ``OpenAIBatchClient`` — uses `/v1/files` + `/v1/batches`
    - ``AnthropicBatchClient`` — uses Messages Batches API
    - ``GeminiBatchClient`` — inline mode, no GCS required (<20 MB batches)

    The Protocol is runtime-checkable so tests can drop in fake clients
    without subclassing.
    """

    provider: str
    """Short provider key: ``"openai"``, ``"anthropic"``, ``"gemini"``."""

    async def submit(self, requests: list[BatchRequest]) -> BatchJobInfo:
        """Upload requests and create a batch job. Returns initial status."""
        ...

    async def poll(self, batch_id: str) -> BatchJobInfo:
        """Fetch current status of a batch job."""
        ...

    async def fetch_results(self, batch_id: str) -> list[BatchResponse]:
        """Download and parse the batch output file.

        Only valid once status is terminal. Implementations may return an
        empty list for non-completed batches rather than raising, so the
        caller can short-circuit.
        """
        ...

    async def cancel(self, batch_id: str) -> BatchJobInfo:
        """Attempt to cancel an in-flight batch."""
        ...
