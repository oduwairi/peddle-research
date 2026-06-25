"""Unit tests for ``pipeline.collect_batch`` (single-pass).

Covers the happy round-trip (briefs + responses_raw written, registry
emptied), provider-error counting, partial-failure threshold trip,
stuck-batch auto-cancel, and empty-sidecar skip.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from draper.construction.batch import (
    PendingBatch,
    PendingBatchSidecar,
)
from draper.construction.batch.types import (
    BatchJobInfo,
    BatchRequest,
    BatchResponse,
    BatchStatus,
)
from draper.construction_v2 import pipeline
from draper.construction_v2.config import ConstructionV2Config, ProviderConfig

_BRIEF: dict[str, object] = {
    "task": "Write a Reddit post for a payroll-compliance tool.",
    "product": {"name": "Compliantly", "tone_signals": ["clipped"]},
    "bridge": {
        "angle": "problem-aware skeptic",
        "buyer_pain": "compliance review blocks weekly hires",
    },
    "platform": "reddit",
}
_THINK = (
    "The brief says reddit and the angle is problem-aware skeptic, "
    "so I lead with the friction the buyer already feels."
)
_DELIVERABLE = "Hire fast. Stay compliant."


def _single_pass_content(
    brief: dict[str, object] = _BRIEF,
    think: str = _THINK,
    deliverable: str = _DELIVERABLE,
) -> str:
    return f"<brief>\n{json.dumps(brief)}\n</brief>\n\n<think>\n{think}\n</think>\n\n{deliverable}"


class _FakeBatchClient:
    """Records calls and returns hand-crafted ``BatchResponse`` lists."""

    provider = "openai"

    def __init__(
        self,
        *,
        poll_info: BatchJobInfo,
        results: list[BatchResponse],
    ) -> None:
        self._poll_info = poll_info
        self._results = results
        self.cancel_calls: list[str] = []

    async def submit(self, requests: list[BatchRequest]) -> BatchJobInfo:  # noqa: ARG002
        raise AssertionError("submit not exercised in collect tests")

    async def poll(self, batch_id: str) -> BatchJobInfo:  # noqa: ARG002
        return self._poll_info

    async def fetch_results(self, batch_id: str) -> list[BatchResponse]:  # noqa: ARG002
        return list(self._results)

    async def cancel(self, batch_id: str) -> BatchJobInfo:
        self.cancel_calls.append(batch_id)
        return BatchJobInfo(
            batch_id=batch_id,
            provider=self.provider,
            status=BatchStatus.CANCELLED,
            request_count=self._poll_info.request_count,
        )


def _fresh_config(tmp_path: Path) -> ConstructionV2Config:
    cfg = ConstructionV2Config(
        output_dir=str(tmp_path / "constructed_v2"),
        final_dir=str(tmp_path / "final_v2"),
        audit_dir=str(tmp_path / "constructed_v2" / "_audit"),
        providers={
            "openai": ProviderConfig(model="gpt-5.4-mini"),
        },
    )
    cfg.single_pass.briefs_cache_path = str(
        tmp_path / "constructed_v2" / "copywriting" / "briefs.jsonl"
    )
    return cfg


def _register_batch(
    cfg: ConstructionV2Config,
    *,
    batch_id: str,
    ad_ids: list[str],
    submitted_at: str | None = None,
    model: str = "gpt-5.4-mini",
) -> None:
    registry = pipeline.registry_for(cfg)
    if submitted_at is None:
        submitted_at = datetime.now(UTC).isoformat()
    sidecars = [
        PendingBatchSidecar(custom_id=f"teacher-{aid}", prompt_index=i, source_ad_ids=[aid])
        for i, aid in enumerate(ad_ids)
    ]
    registry.add(
        PendingBatch(
            batch_id=batch_id,
            provider="openai",
            model=model,
            task_format=pipeline.TASK_FORMAT,
            submitted_at=submitted_at,
            status=BatchStatus.IN_PROGRESS.value,
            request_count=len(sidecars),
            sidecars=sidecars,
        )
    )


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    client: _FakeBatchClient,
) -> None:
    monkeypatch.setattr(
        "draper.construction_v2.pipeline.make_batch_client",
        lambda _model: client,
    )


def test_collect_writes_briefs_and_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _fresh_config(tmp_path)
    batch_id = "batch-abc"
    ad_ids = ["ad-1", "ad-2"]
    _register_batch(cfg, batch_id=batch_id, ad_ids=ad_ids)

    poll_info = BatchJobInfo(
        batch_id=batch_id,
        provider="openai",
        status=BatchStatus.COMPLETED,
        request_count=len(ad_ids),
        completed_count=len(ad_ids),
    )
    results = [
        BatchResponse(
            custom_id=f"teacher-{aid}",
            content=_single_pass_content(),
            model="gpt-5.4-mini",
        )
        for aid in ad_ids
    ]
    _install_fake_client(monkeypatch, _FakeBatchClient(poll_info=poll_info, results=results))

    import asyncio

    res = asyncio.run(pipeline.collect_batch(cfg, batch_id))
    assert res.terminal is True
    assert res.briefs_written == 2
    assert res.rationales_written == 2
    assert res.provider_errors == 0
    assert res.parse_failures == 0

    briefs_lines = pipeline.briefs_path(cfg).read_text(encoding="utf-8").splitlines()
    assert len(briefs_lines) == 2
    assert {json.loads(line)["ad_id"] for line in briefs_lines} == set(ad_ids)

    responses_lines = pipeline.responses_path(cfg).read_text(encoding="utf-8").splitlines()
    assert len(responses_lines) == 2
    first = json.loads(responses_lines[0])
    assert "<think>" in first["content"]
    assert _DELIVERABLE in first["content"]
    assert first["model"] == "gpt-5.4-mini"

    # Registry should be emptied after a successful collect.
    assert pipeline.registry_for(cfg).get(batch_id) is None


def test_collect_counts_provider_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _fresh_config(tmp_path)
    # Tighten the threshold so a single error is enough.
    cfg.batch.max_partial_error_rate = 0.10

    batch_id = "batch-err"
    ad_ids = ["ad-a", "ad-b", "ad-c", "ad-d"]
    _register_batch(cfg, batch_id=batch_id, ad_ids=ad_ids)

    poll_info = BatchJobInfo(
        batch_id=batch_id,
        provider="openai",
        status=BatchStatus.COMPLETED,
        request_count=len(ad_ids),
        completed_count=len(ad_ids) - 1,
        failed_count=1,
    )
    results = [
        BatchResponse(
            custom_id="teacher-ad-a",
            content=_single_pass_content(),
            model="gpt-5.4-mini",
        ),
        BatchResponse(
            custom_id="teacher-ad-b",
            content="",
            model="gpt-5.4-mini",
            error="upstream rate limit",
        ),
        BatchResponse(
            custom_id="teacher-ad-c",
            content=_single_pass_content(),
            model="gpt-5.4-mini",
        ),
        BatchResponse(
            custom_id="teacher-ad-d",
            content=_single_pass_content(),
            model="gpt-5.4-mini",
        ),
    ]
    _install_fake_client(monkeypatch, _FakeBatchClient(poll_info=poll_info, results=results))

    import asyncio

    # 1 error out of 4 = 25% > 10% → threshold trip, raises after writing rows.
    with pytest.raises(pipeline.PartialFailureThreshold):
        asyncio.run(pipeline.collect_batch(cfg, batch_id))

    # Successful rows still persisted.
    briefs = pipeline.briefs_path(cfg).read_text(encoding="utf-8").splitlines()
    assert len(briefs) == 3


def test_collect_skips_parse_failures_and_writes_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _fresh_config(tmp_path)
    batch_id = "batch-parse"
    ad_ids = ["ad-x", "ad-y"]
    _register_batch(cfg, batch_id=batch_id, ad_ids=ad_ids)

    poll_info = BatchJobInfo(
        batch_id=batch_id,
        provider="openai",
        status=BatchStatus.COMPLETED,
        request_count=len(ad_ids),
        completed_count=len(ad_ids),
    )
    results = [
        BatchResponse(
            custom_id="teacher-ad-x",
            content=_single_pass_content(),
            model="gpt-5.4-mini",
        ),
        # Malformed: no <brief> region, no <think>, no deliverable structure.
        BatchResponse(
            custom_id="teacher-ad-y",
            content="raw garbage output",
            model="gpt-5.4-mini",
        ),
    ]
    _install_fake_client(monkeypatch, _FakeBatchClient(poll_info=poll_info, results=results))

    import asyncio

    res = asyncio.run(pipeline.collect_batch(cfg, batch_id))
    assert res.briefs_written == 1
    assert res.parse_failures == 1
    rejections_path = pipeline.audit_path(cfg, "collect_rejections.jsonl")
    rows = rejections_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    rec = json.loads(rows[0])
    assert rec["ad_id"] == "ad-y"
    assert rec["stage"] == "parse"


def test_collect_force_cancels_stuck_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _fresh_config(tmp_path)
    cfg.batch.stuck_timeout_minutes = 5
    batch_id = "batch-stuck"
    long_ago = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
    _register_batch(cfg, batch_id=batch_id, ad_ids=["ad-z"], submitted_at=long_ago)

    poll_info = BatchJobInfo(
        batch_id=batch_id,
        provider="openai",
        status=BatchStatus.PENDING,  # not terminal
        request_count=1,
    )
    fake = _FakeBatchClient(poll_info=poll_info, results=[])
    _install_fake_client(monkeypatch, fake)

    import asyncio

    res = asyncio.run(pipeline.collect_batch(cfg, batch_id))
    assert res.stuck_cancelled is True
    assert res.status == "cancelled_stuck"
    assert fake.cancel_calls == [batch_id]
    # Registry must be emptied so the slot is freed.
    assert pipeline.registry_for(cfg).get(batch_id) is None


def test_collect_skips_results_with_unknown_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _fresh_config(tmp_path)
    batch_id = "batch-empty-side"
    ad_ids = ["ad-real"]
    _register_batch(cfg, batch_id=batch_id, ad_ids=ad_ids)

    poll_info = BatchJobInfo(
        batch_id=batch_id,
        provider="openai",
        status=BatchStatus.COMPLETED,
        request_count=2,
        completed_count=2,
    )
    results = [
        BatchResponse(
            custom_id="teacher-ad-real",
            content=_single_pass_content(),
            model="gpt-5.4-mini",
        ),
        # custom_id not in the sidecars — must be skipped, not crash.
        BatchResponse(
            custom_id="teacher-ad-stray",
            content=_single_pass_content(),
            model="gpt-5.4-mini",
        ),
    ]
    _install_fake_client(monkeypatch, _FakeBatchClient(poll_info=poll_info, results=results))

    import asyncio

    res = asyncio.run(pipeline.collect_batch(cfg, batch_id))
    assert res.briefs_written == 1
    assert res.rationales_written == 1


def test_collect_returns_missing_when_not_in_registry(tmp_path: Path) -> None:
    cfg = _fresh_config(tmp_path)
    # No batch registered.
    import asyncio

    res = asyncio.run(pipeline.collect_batch(cfg, "nonexistent"))
    assert res.missing is True
    assert res.terminal is False
