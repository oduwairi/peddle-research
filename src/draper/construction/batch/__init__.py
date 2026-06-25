"""Provider-agnostic batch-generation interface for construction.

Primary workflow lives in the chat-first `prepare`/`ingest` CLI commands.
This package adds a parallel API-based workflow using each provider's
native Batch API (OpenAI Batch, Anthropic Message Batches, Gemini Batch
Mode). The economics are ~50% cheaper than sync calls with 24h turnaround.

The public surface is intentionally small:

- ``BatchClient`` — Protocol each provider implements
- ``BatchRequest`` / ``BatchResponse`` / ``BatchJobInfo`` — transport types
- ``BatchRegistry`` — per-format JSON registry of pending jobs
- ``make_batch_client`` — factory that picks the right client for a model
"""

from __future__ import annotations

from draper.construction.batch.factory import (
    make_batch_client,
    provider_for_model,
    validate_batch_model,
)
from draper.construction.batch.registry import (
    BatchRegistry,
    PendingBatch,
    PendingBatchSidecar,
)
from draper.construction.batch.types import (
    BatchClient,
    BatchJobInfo,
    BatchRequest,
    BatchResponse,
    BatchStatus,
)

__all__ = [
    "BatchClient",
    "BatchJobInfo",
    "BatchRegistry",
    "BatchRequest",
    "BatchResponse",
    "BatchStatus",
    "PendingBatch",
    "PendingBatchSidecar",
    "make_batch_client",
    "provider_for_model",
    "validate_batch_model",
]
