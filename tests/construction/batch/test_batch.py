"""Tests for the provider-agnostic batch infrastructure.

Covers:

- ``provider_for_model`` routing + factory error cases
- ``BatchRegistry`` persistence round-trip
- OpenAI batch input/output JSONL wire format
- OpenAI client `submit`/`poll`/`fetch_results` flow via SDK stubs
- Anthropic client result parsing for the three terminal types
- CLI helpers: `_custom_id`, `_sidecar_from_prepared`
"""

from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from draper.construction.batch import (
    BatchRegistry,
    BatchRequest,
    BatchStatus,
    PendingBatch,
    PendingBatchSidecar,
    make_batch_client,
    provider_for_model,
)
from draper.construction.batch.anthropic_client import AnthropicBatchClient
from draper.construction.batch.gemini_client import GeminiBatchClient
from draper.construction.batch.openai_client import OpenAIBatchClient

# ---------------------------------------------------------------------------
# provider_for_model / make_batch_client
# ---------------------------------------------------------------------------


class TestProviderRouting:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gpt-4o", "openai"),
            ("gpt-4o-mini", "openai"),
            ("o1-preview", "openai"),
            ("o3-mini", "openai"),
            ("claude-sonnet-4-6", "anthropic"),
            ("claude-haiku-4-5", "anthropic"),
            ("gemini-2.5-pro", "gemini"),
        ],
    )
    def test_provider_detection(self, model: str, expected: str) -> None:
        assert provider_for_model(model) == expected

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="batch provider"):
            provider_for_model("mistral-large")

    def test_factory_returns_right_client(self) -> None:
        # Don't actually hit the network — the constructors just cache
        # the SDK client; instantiation is enough.
        assert isinstance(make_batch_client("gpt-4o-mini"), OpenAIBatchClient)
        assert isinstance(make_batch_client("claude-sonnet-4-6"), AnthropicBatchClient)
        assert isinstance(make_batch_client("gemini-2.5-pro"), GeminiBatchClient)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestBatchRegistry:
    def test_add_get_round_trip(self, tmp_path: Path) -> None:
        registry = BatchRegistry(tmp_path, "positioning")
        sidecars = [
            PendingBatchSidecar(
                custom_id="positioning-00000",
                prompt_index=0,
                source_ad_ids=["a1", "a2"],
                prompt_style="data_grounded",
                persona_id="smb_founder_first_ads",
                seed_idx=2,
                evol_op="deepen",
                difficulty="standard",
                turn_structure="single",
                followup_type="",
                provider_label="gpt",
            )
        ]
        batch = PendingBatch(
            batch_id="batch_abc",
            provider="openai",
            model="gpt-4o-mini",
            task_format="positioning",
            submitted_at="2026-04-14T00:00:00+00:00",
            request_count=1,
            sidecars=sidecars,
        )
        registry.add(batch)

        # Reload from disk
        reloaded = BatchRegistry(tmp_path, "positioning")
        got = reloaded.get("batch_abc")
        assert got is not None
        assert got.model == "gpt-4o-mini"
        assert got.sidecars[0].source_ad_ids == ["a1", "a2"]
        assert got.sidecar_by_custom_id("positioning-00000") is not None
        assert got.sidecar_by_custom_id("missing") is None

    def test_update_status_and_pending_filter(self, tmp_path: Path) -> None:
        registry = BatchRegistry(tmp_path, "positioning")
        registry.add(
            PendingBatch(
                batch_id="batch_done",
                provider="openai",
                model="gpt-4o-mini",
                task_format="positioning",
                submitted_at="2026-04-14T00:00:00+00:00",
                request_count=1,
                status=BatchStatus.PENDING.value,
            )
        )
        registry.add(
            PendingBatch(
                batch_id="batch_running",
                provider="openai",
                model="gpt-4o-mini",
                task_format="positioning",
                submitted_at="2026-04-14T00:01:00+00:00",
                request_count=1,
                status=BatchStatus.IN_PROGRESS.value,
            )
        )
        registry.update_status(
            "batch_done",
            status=BatchStatus.COMPLETED.value,
            completed_count=1,
            failed_count=0,
        )

        pending = registry.pending()
        assert [b.batch_id for b in pending] == ["batch_running"]

    def test_reserved_ad_ids_excludes_terminal_batches(self, tmp_path: Path) -> None:
        registry = BatchRegistry(tmp_path, "copywriting")
        # Pending batch with two ad IDs
        registry.add(
            PendingBatch(
                batch_id="batch_pending",
                provider="openai",
                model="gpt-4o",
                task_format="copywriting",
                submitted_at="2026-04-14T00:00:00+00:00",
                status=BatchStatus.PENDING.value,
                request_count=1,
                sidecars=[
                    PendingBatchSidecar(
                        custom_id="copywriting-00000",
                        prompt_index=0,
                        source_ad_ids=["ad_a", "ad_b"],
                        prompt_style="data_grounded",
                        persona_id="smb_founder_first_ads",
                        seed_idx=0,
                        evol_op="",
                        difficulty="standard",
                        turn_structure="single",
                        followup_type="",
                        provider_label="gpt",
                    )
                ],
            )
        )
        # Completed-but-not-yet-ingested batch — its ad IDs MUST still be
        # reserved so a concurrent batch-submit doesn't draw the same ads
        # before batch-collect removes this entry.
        registry.add(
            PendingBatch(
                batch_id="batch_done",
                provider="openai",
                model="gpt-4o",
                task_format="copywriting",
                submitted_at="2026-04-14T00:01:00+00:00",
                status=BatchStatus.COMPLETED.value,
                request_count=1,
                sidecars=[
                    PendingBatchSidecar(
                        custom_id="copywriting-00001",
                        prompt_index=1,
                        source_ad_ids=["ad_c"],
                        prompt_style="natural",
                        persona_id="brand_cmo",
                        seed_idx=1,
                        evol_op="",
                        difficulty="standard",
                        turn_structure="single",
                        followup_type="",
                        provider_label="gpt",
                    )
                ],
            )
        )
        reserved = registry.reserved_ad_ids()
        assert reserved == {"ad_a", "ad_b", "ad_c"}

        # Only failed/cancelled/expired batches should be excluded
        registry.add(
            PendingBatch(
                batch_id="batch_cancelled",
                provider="openai",
                model="gpt-4o",
                task_format="copywriting",
                submitted_at="2026-04-14T00:02:00+00:00",
                status=BatchStatus.CANCELLED.value,
                request_count=1,
                sidecars=[
                    PendingBatchSidecar(
                        custom_id="copywriting-00002",
                        prompt_index=2,
                        source_ad_ids=["ad_d"],
                        prompt_style="natural",
                        persona_id="brand_cmo",
                        seed_idx=2,
                        evol_op="",
                        difficulty="standard",
                        turn_structure="single",
                        followup_type="",
                        provider_label="gpt",
                    )
                ],
            )
        )
        reserved = registry.reserved_ad_ids()
        assert "ad_d" not in reserved

    def test_pending_request_count(self, tmp_path: Path) -> None:
        registry = BatchRegistry(tmp_path, "copywriting")
        registry.add(
            PendingBatch(
                batch_id="batch_a",
                provider="openai",
                model="gpt-4o",
                task_format="copywriting",
                submitted_at="2026-04-14T00:00:00+00:00",
                status=BatchStatus.IN_PROGRESS.value,
                request_count=20,
            )
        )
        registry.add(
            PendingBatch(
                batch_id="batch_b",
                provider="gemini",
                model="gemini-2.5-pro",
                task_format="copywriting",
                submitted_at="2026-04-14T00:01:00+00:00",
                status=BatchStatus.PENDING.value,
                request_count=15,
            )
        )
        # Cancelled — should not count
        registry.add(
            PendingBatch(
                batch_id="batch_c",
                provider="anthropic",
                model="claude-haiku-4-5",
                task_format="copywriting",
                submitted_at="2026-04-14T00:02:00+00:00",
                status=BatchStatus.CANCELLED.value,
                request_count=10,
            )
        )
        # Completed-but-not-ingested — MUST count (same as in-flight for RNG offset)
        registry.add(
            PendingBatch(
                batch_id="batch_d",
                provider="openai",
                model="gpt-4o",
                task_format="copywriting",
                submitted_at="2026-04-14T00:03:00+00:00",
                status=BatchStatus.COMPLETED.value,
                request_count=12,
            )
        )
        assert registry.pending_request_count() == 47  # 20 + 15 + 12, not 10

    def test_remove_clears_entry(self, tmp_path: Path) -> None:
        registry = BatchRegistry(tmp_path, "positioning")
        registry.add(
            PendingBatch(
                batch_id="batch_x",
                provider="openai",
                model="gpt-4o-mini",
                task_format="positioning",
                submitted_at="2026-04-14T00:00:00+00:00",
            )
        )
        registry.remove("batch_x")
        assert registry.get("batch_x") is None
        # Persistence confirms it's gone
        assert BatchRegistry(tmp_path, "positioning").get("batch_x") is None


# ---------------------------------------------------------------------------
# OpenAI wire-format helpers
# ---------------------------------------------------------------------------


class TestOpenAIWireFormat:
    def test_input_jsonl_shape(self) -> None:
        req = BatchRequest(
            custom_id="positioning-00000",
            system="you are helpful",
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
            max_tokens=2048,
            temperature=0.5,
        )
        rendered = OpenAIBatchClient._render_input_jsonl([req])
        line = rendered.strip().splitlines()[0]
        obj = json.loads(line)
        assert obj["custom_id"] == "positioning-00000"
        assert obj["method"] == "POST"
        assert obj["url"] == "/v1/chat/completions"
        body = obj["body"]
        assert body["model"] == "gpt-4o-mini"
        # GPT-5.* require max_completion_tokens; we send it universally.
        assert body["max_completion_tokens"] == 2048
        assert body["temperature"] == 0.5
        # System prepended
        assert body["messages"][0] == {
            "role": "system",
            "content": "you are helpful",
        }
        assert body["messages"][1] == {"role": "user", "content": "hi"}

    def test_input_jsonl_omits_system_when_none(self) -> None:
        req = BatchRequest(
            custom_id="x-00000",
            system=None,
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
        )
        rendered = OpenAIBatchClient._render_input_jsonl([req])
        obj = json.loads(rendered.strip())
        assert obj["body"]["messages"][0]["role"] == "user"
        assert len(obj["body"]["messages"]) == 1

    def test_output_jsonl_happy_path(self) -> None:
        text = json.dumps(
            {
                "custom_id": "positioning-00000",
                "response": {
                    "status_code": 200,
                    "body": {
                        "model": "gpt-4o-mini-2024-07-18",
                        "choices": [{"message": {"content": "<user_prompt>q</user_prompt>"}}],
                        "usage": {"prompt_tokens": 42, "completion_tokens": 7},
                    },
                },
                "error": None,
            }
        )
        [resp] = OpenAIBatchClient._parse_output_jsonl(text)
        assert resp.custom_id == "positioning-00000"
        assert resp.content == "<user_prompt>q</user_prompt>"
        assert resp.input_tokens == 42
        assert resp.output_tokens == 7
        assert resp.model == "gpt-4o-mini-2024-07-18"
        assert resp.error == ""

    def test_output_jsonl_captures_line_error(self) -> None:
        text = json.dumps(
            {
                "custom_id": "positioning-00001",
                "response": None,
                "error": {"code": "invalid_request", "message": "bad"},
            }
        )
        [resp] = OpenAIBatchClient._parse_output_jsonl(text)
        assert resp.content == ""
        assert "invalid_request" in resp.error


# ---------------------------------------------------------------------------
# OpenAI client flow (mocked SDK)
# ---------------------------------------------------------------------------


@dataclass
class _FakeFileObject:
    id: str = "file-123"


@dataclass
class _FakeRequestCounts:
    total: int = 2
    completed: int = 2
    failed: int = 0


@dataclass
class _FakeBatch:
    id: str = "batch_abc"
    status: str = "completed"
    output_file_id: str | None = "file-out"
    error_file_id: str | None = None
    created_at: int = 123
    expires_at: int = 456
    request_counts: _FakeRequestCounts = field(default_factory=_FakeRequestCounts)
    errors: Any = None


@dataclass
class _FakeContent:
    text: str


class _FakeFiles:
    def __init__(self, output_text: str) -> None:
        self._output_text = output_text
        self.last_upload: bytes | None = None

    async def create(self, *, file: Any, purpose: str) -> _FakeFileObject:
        # file is a BytesIO — read it to verify the caller wrote the JSONL
        data = file.read() if hasattr(file, "read") else bytes(file)
        self.last_upload = data
        assert purpose == "batch"
        return _FakeFileObject()

    async def content(self, file_id: str) -> _FakeContent:
        return _FakeContent(text=self._output_text)


class _FakeBatches:
    def __init__(self, batch: _FakeBatch) -> None:
        self._batch = batch
        self.create_calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeBatch:
        self.create_calls.append(kwargs)
        return self._batch

    async def retrieve(self, batch_id: str) -> _FakeBatch:
        assert batch_id == self._batch.id
        return self._batch

    async def cancel(self, batch_id: str) -> _FakeBatch:
        self._batch.status = "cancelled"
        return self._batch


class _FakeOpenAI:
    def __init__(self, batch: _FakeBatch, output_text: str) -> None:
        self.files = _FakeFiles(output_text)
        self.batches = _FakeBatches(batch)


class TestOpenAIBatchClientFlow:
    @pytest.mark.asyncio
    async def test_submit_uploads_and_creates(self) -> None:
        fake = _FakeOpenAI(_FakeBatch(status="validating"), output_text="")
        client = OpenAIBatchClient(client=fake)  # type: ignore[arg-type]

        info = await client.submit(
            [
                BatchRequest(
                    custom_id="x-00000",
                    system=None,
                    messages=[{"role": "user", "content": "hi"}],
                    model="gpt-4o-mini",
                )
            ]
        )
        assert info.batch_id == "batch_abc"
        assert info.status == BatchStatus.PENDING
        # Uploaded bytes parse as valid JSONL
        assert fake.files.last_upload is not None
        line = fake.files.last_upload.decode("utf-8").strip().splitlines()[0]
        assert json.loads(line)["custom_id"] == "x-00000"
        # /v1/chat/completions with 24h window
        [create_kwargs] = fake.batches.create_calls
        assert create_kwargs["endpoint"] == "/v1/chat/completions"
        assert create_kwargs["completion_window"] == "24h"

    @pytest.mark.asyncio
    async def test_submit_rejects_empty(self) -> None:
        fake = _FakeOpenAI(_FakeBatch(), "")
        client = OpenAIBatchClient(client=fake)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="must not be empty"):
            await client.submit([])

    @pytest.mark.asyncio
    async def test_poll_maps_status(self) -> None:
        fake = _FakeOpenAI(_FakeBatch(status="in_progress"), "")
        client = OpenAIBatchClient(client=fake)  # type: ignore[arg-type]
        info = await client.poll("batch_abc")
        assert info.status == BatchStatus.IN_PROGRESS
        assert info.is_terminal is False

    @pytest.mark.asyncio
    async def test_fetch_results_returns_parsed(self) -> None:
        output = json.dumps(
            {
                "custom_id": "x-00000",
                "response": {
                    "body": {
                        "choices": [{"message": {"content": "hello"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                        "model": "gpt-4o-mini",
                    }
                },
                "error": None,
            }
        )
        fake = _FakeOpenAI(_FakeBatch(status="completed"), output)
        client = OpenAIBatchClient(client=fake)  # type: ignore[arg-type]
        [resp] = await client.fetch_results("batch_abc")
        assert resp.content == "hello"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 3

    @pytest.mark.asyncio
    async def test_fetch_results_empty_when_not_completed(self) -> None:
        fake = _FakeOpenAI(_FakeBatch(status="in_progress"), "")
        client = OpenAIBatchClient(client=fake)  # type: ignore[arg-type]
        assert await client.fetch_results("batch_abc") == []


# ---------------------------------------------------------------------------
# Anthropic result parser (no network)
# ---------------------------------------------------------------------------


@dataclass
class _FakeAnthropicBlock:
    text: str


@dataclass
class _FakeAnthropicUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class _FakeAnthropicMessage:
    content: list[_FakeAnthropicBlock]
    model: str
    usage: _FakeAnthropicUsage


@dataclass
class _FakeAnthropicResult:
    type: str
    message: _FakeAnthropicMessage | None = None
    error: Any = None


@dataclass
class _FakeAnthropicItem:
    custom_id: str
    result: _FakeAnthropicResult


class TestAnthropicResultParser:
    def test_succeeded_extracts_text_and_usage(self) -> None:
        item = _FakeAnthropicItem(
            custom_id="x-00000",
            result=_FakeAnthropicResult(
                type="succeeded",
                message=_FakeAnthropicMessage(
                    content=[_FakeAnthropicBlock("hello"), _FakeAnthropicBlock(" world")],
                    model="claude-sonnet-4-6",
                    usage=_FakeAnthropicUsage(50, 12),
                ),
            ),
        )
        resp = AnthropicBatchClient._parse_result(item)
        assert resp.content == "hello world"
        assert resp.input_tokens == 50
        assert resp.output_tokens == 12
        assert resp.model == "claude-sonnet-4-6"

    @pytest.mark.parametrize(
        ("result_type", "payload"),
        [
            ("errored", {"message": "invalid_request"}),
            ("canceled", None),
            ("expired", None),
        ],
    )
    def test_non_success_surfaces_error(self, result_type: str, payload: object) -> None:
        item = _FakeAnthropicItem(
            custom_id="x-00001",
            result=_FakeAnthropicResult(type=result_type, error=payload),
        )
        resp = AnthropicBatchClient._parse_result(item)
        assert resp.content == ""
        assert resp.error


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


class TestCLIHelpers:
    def test_custom_id_is_stable_and_zero_padded(self) -> None:
        # Import the helper from the CLI module
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "construct_cli",
            Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "construct.py",
        )
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        assert module._custom_id("positioning", 0) == "positioning-00000"
        assert module._custom_id("positioning", 12345) == "positioning-12345"


# ---------------------------------------------------------------------------
# Gemini wire-format helpers + client flow (mocked SDK)
# ---------------------------------------------------------------------------


@dataclass
class _FakeJobError:
    message: str = "something went wrong"

    def __str__(self) -> str:
        return self.message


@dataclass
class _FakeGeminiUsage:
    prompt_token_count: int = 10
    candidates_token_count: int = 5


@dataclass
class _FakeGeminiPart:
    text: str


@dataclass
class _FakeGeminiContent:
    parts: list[_FakeGeminiPart]
    role: str = "model"


@dataclass
class _FakeCandidate:
    content: _FakeGeminiContent


@dataclass
class _FakeGenResponse:
    candidates: list[_FakeCandidate]
    usage_metadata: _FakeGeminiUsage
    model_version: str = "gemini-2.5-pro-001"


@dataclass
class _FakeInlinedResponse:
    metadata: dict[str, str]
    response: _FakeGenResponse | None = None
    error: _FakeJobError | None = None


@dataclass
class _FakeCompletionStats:
    successful_count: int = 0
    failed_count: int = 0
    incomplete_count: int = 0


@dataclass
class _FakeDest:
    inlined_responses: list[_FakeInlinedResponse]


@dataclass
class _FakeBatchJob:
    name: str = "batches/abc123"
    state: Any = None
    completion_stats: _FakeCompletionStats = field(default_factory=_FakeCompletionStats)
    dest: _FakeDest | None = None
    create_time: Any = None
    end_time: Any = None

    def __post_init__(self) -> None:
        if self.state is None:
            from google.genai.types import JobState

            self.state = JobState.JOB_STATE_SUCCEEDED


class _FakeGeminiBatches:
    def __init__(self, job: _FakeBatchJob) -> None:
        self._job = job
        self.create_calls: list[dict[str, Any]] = []
        self.cancel_calls: list[str] = []

    async def create(self, *, model: str, src: Any, config: Any = None) -> _FakeBatchJob:
        self.create_calls.append({"model": model, "src": src, "config": config})
        return self._job

    async def get(self, *, name: str, config: Any = None) -> _FakeBatchJob:
        assert name == self._job.name
        return self._job

    async def cancel(self, *, name: str, config: Any = None) -> None:
        self.cancel_calls.append(name)
        from google.genai.types import JobState

        self._job.state = JobState.JOB_STATE_CANCELLED


class _FakeGeminiAio:
    def __init__(self, job: _FakeBatchJob) -> None:
        self.batches = _FakeGeminiBatches(job)


class _FakeGeminiClient:
    def __init__(self, job: _FakeBatchJob) -> None:
        self.aio = _FakeGeminiAio(job)


class TestGeminiWireFormat:
    def test_to_inlined_with_system(self) -> None:
        from draper.construction.batch.gemini_client import GeminiBatchClient

        req = BatchRequest(
            custom_id="copywriting-00000",
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Write me an ad."}],
            model="gemini-2.5-pro",
            max_tokens=2048,
            temperature=0.7,
        )
        inlined = GeminiBatchClient._to_inlined(req)
        assert inlined.metadata == {"custom_id": "copywriting-00000"}
        assert inlined.config.system_instruction == "You are a helpful assistant."
        assert inlined.config.temperature == 0.7
        # Gemini 2.5 thinking models share max_output_tokens between thinking
        # and visible output. Visible budget is scaled by
        # GEMINI_MAX_TOKENS_MULTIPLIER (=2) because Gemini underproduces at
        # matched caps; thinking_budget (default 1024) is added on top.
        assert inlined.config.max_output_tokens == 2048 * 2 + 1024
        assert inlined.config.thinking_config is not None
        assert inlined.config.thinking_config.thinking_budget == 1024
        assert len(inlined.contents) == 1
        assert inlined.contents[0].role == "user"
        assert inlined.contents[0].parts[0].text == "Write me an ad."

    def test_to_inlined_maps_assistant_role(self) -> None:
        from draper.construction.batch.gemini_client import GeminiBatchClient

        req = BatchRequest(
            custom_id="x-00000",
            system=None,
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            model="gemini-2.5-pro",
        )
        inlined = GeminiBatchClient._to_inlined(req)
        assert inlined.contents[0].role == "user"
        assert inlined.contents[1].role == "model"

    def test_to_inlined_without_system(self) -> None:
        from draper.construction.batch.gemini_client import GeminiBatchClient

        req = BatchRequest(
            custom_id="x-00001",
            system=None,
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-2.5-pro",
        )
        inlined = GeminiBatchClient._to_inlined(req)
        assert inlined.config.system_instruction is None

    def test_to_inlined_thinking_budget_zero_disables_padding(self) -> None:
        from draper.construction.batch.gemini_client import GeminiBatchClient

        req = BatchRequest(
            custom_id="x-00002",
            system=None,
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-2.5-pro",
            max_tokens=2048,
        )
        inlined = GeminiBatchClient._to_inlined(req, thinking_budget=0)
        # When thinking is disabled, no padding and no thinking_config, but
        # the visible-output multiplier still applies.
        assert inlined.config.max_output_tokens == 2048 * 2
        assert inlined.config.thinking_config is None

    def test_parse_inlined_success(self) -> None:
        from draper.construction.batch.gemini_client import GeminiBatchClient

        item = _FakeInlinedResponse(
            metadata={"custom_id": "copywriting-00000"},
            response=_FakeGenResponse(
                candidates=[_FakeCandidate(_FakeGeminiContent([_FakeGeminiPart("hello")]))],
                usage_metadata=_FakeGeminiUsage(prompt_token_count=20, candidates_token_count=8),
                model_version="gemini-2.5-pro-001",
            ),
        )
        resp = GeminiBatchClient._parse_inlined(item)  # type: ignore[arg-type]
        assert resp.custom_id == "copywriting-00000"
        assert resp.content == "hello"
        assert resp.input_tokens == 20
        assert resp.output_tokens == 8
        assert resp.model == "gemini-2.5-pro-001"
        assert resp.error == ""

    def test_parse_inlined_error(self) -> None:
        from draper.construction.batch.gemini_client import GeminiBatchClient

        item = _FakeInlinedResponse(
            metadata={"custom_id": "copywriting-00001"},
            error=_FakeJobError("safety filter triggered"),
        )
        resp = GeminiBatchClient._parse_inlined(item)  # type: ignore[arg-type]
        assert resp.content == ""
        assert "safety filter" in resp.error

    def test_parse_inlined_concatenates_parts(self) -> None:
        from draper.construction.batch.gemini_client import GeminiBatchClient

        item = _FakeInlinedResponse(
            metadata={"custom_id": "x"},
            response=_FakeGenResponse(
                candidates=[
                    _FakeCandidate(
                        _FakeGeminiContent([_FakeGeminiPart("foo"), _FakeGeminiPart(" bar")])
                    )
                ],
                usage_metadata=_FakeGeminiUsage(),
            ),
        )
        resp = GeminiBatchClient._parse_inlined(item)  # type: ignore[arg-type]
        assert resp.content == "foo bar"


class TestGeminiBatchClientFlow:
    @pytest.mark.asyncio
    async def test_submit_calls_create_and_returns_info(self) -> None:
        from google.genai.types import JobState

        from draper.construction.batch.gemini_client import GeminiBatchClient

        job = _FakeBatchJob(
            name="batches/abc123",
            state=JobState.JOB_STATE_PENDING,
        )
        fake_sdk = _FakeGeminiClient(job)
        client = GeminiBatchClient(client=fake_sdk)  # type: ignore[arg-type]
        info = await client.submit(
            [
                BatchRequest(
                    custom_id="copywriting-00000",
                    system="sys",
                    messages=[{"role": "user", "content": "hi"}],
                    model="gemini-2.5-pro",
                )
            ]
        )
        assert info.batch_id == "batches/abc123"
        assert info.status == BatchStatus.PENDING
        assert len(fake_sdk.aio.batches.create_calls) == 1

    @pytest.mark.asyncio
    async def test_submit_rejects_empty(self) -> None:
        from draper.construction.batch.gemini_client import GeminiBatchClient

        job = _FakeBatchJob()
        client = GeminiBatchClient(client=_FakeGeminiClient(job))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="must not be empty"):
            await client.submit([])

    @pytest.mark.asyncio
    async def test_poll_maps_running(self) -> None:
        from google.genai.types import JobState

        from draper.construction.batch.gemini_client import GeminiBatchClient

        job = _FakeBatchJob(state=JobState.JOB_STATE_RUNNING)
        client = GeminiBatchClient(client=_FakeGeminiClient(job))  # type: ignore[arg-type]
        info = await client.poll("batches/abc123")
        assert info.status == BatchStatus.IN_PROGRESS
        assert info.is_terminal is False

    @pytest.mark.asyncio
    async def test_fetch_results_returns_parsed(self) -> None:
        from google.genai.types import JobState

        from draper.construction.batch.gemini_client import GeminiBatchClient

        job = _FakeBatchJob(
            state=JobState.JOB_STATE_SUCCEEDED,
            dest=_FakeDest(
                inlined_responses=[
                    _FakeInlinedResponse(
                        metadata={"custom_id": "copywriting-00000"},
                        response=_FakeGenResponse(
                            candidates=[
                                _FakeCandidate(_FakeGeminiContent([_FakeGeminiPart("result")]))
                            ],
                            usage_metadata=_FakeGeminiUsage(15, 6),
                        ),
                    )
                ]
            ),
        )
        client = GeminiBatchClient(client=_FakeGeminiClient(job))  # type: ignore[arg-type]
        results = await client.fetch_results("batches/abc123")
        assert len(results) == 1
        assert results[0].content == "result"
        assert results[0].input_tokens == 15
        assert results[0].output_tokens == 6

    @pytest.mark.asyncio
    async def test_fetch_results_empty_when_not_complete(self) -> None:
        from google.genai.types import JobState

        from draper.construction.batch.gemini_client import GeminiBatchClient

        job = _FakeBatchJob(state=JobState.JOB_STATE_RUNNING)
        client = GeminiBatchClient(client=_FakeGeminiClient(job))  # type: ignore[arg-type]
        assert await client.fetch_results("batches/abc123") == []

    @pytest.mark.asyncio
    async def test_poll_counts_from_dest_when_stats_empty(self) -> None:
        # Gemini frequently leaves completion_stats null/zero even after the
        # job reaches JOB_STATE_SUCCEEDED. Ensure poll() falls back to counting
        # dest.inlined_responses so the CLI doesn't show "0 done" on a
        # finished batch.
        from google.genai.types import JobState

        from draper.construction.batch.gemini_client import GeminiBatchClient

        job = _FakeBatchJob(
            state=JobState.JOB_STATE_SUCCEEDED,
            completion_stats=_FakeCompletionStats(),  # all zeros
            dest=_FakeDest(
                inlined_responses=[
                    _FakeInlinedResponse(
                        metadata={"custom_id": "x-0"},
                        response=_FakeGenResponse(
                            candidates=[
                                _FakeCandidate(_FakeGeminiContent([_FakeGeminiPart("ok")]))
                            ],
                            usage_metadata=_FakeGeminiUsage(),
                        ),
                    ),
                    _FakeInlinedResponse(
                        metadata={"custom_id": "x-1"},
                        response=_FakeGenResponse(
                            candidates=[
                                _FakeCandidate(_FakeGeminiContent([_FakeGeminiPart("ok2")]))
                            ],
                            usage_metadata=_FakeGeminiUsage(),
                        ),
                    ),
                    _FakeInlinedResponse(
                        metadata={"custom_id": "x-2"},
                        error=_FakeJobError("safety filter"),
                    ),
                ]
            ),
        )
        client = GeminiBatchClient(client=_FakeGeminiClient(job))  # type: ignore[arg-type]
        info = await client.poll("batches/abc123")
        assert info.status == BatchStatus.COMPLETED
        assert info.completed_count == 2
        assert info.failed_count == 1
        assert info.request_count == 3

    @pytest.mark.asyncio
    async def test_cancel_polls_after_cancel(self) -> None:
        from google.genai.types import JobState

        from draper.construction.batch.gemini_client import GeminiBatchClient

        job = _FakeBatchJob(state=JobState.JOB_STATE_RUNNING)
        fake_client = _FakeGeminiClient(job)
        client = GeminiBatchClient(client=fake_client)  # type: ignore[arg-type]
        info = await client.cancel("batches/abc123")
        assert info.status == BatchStatus.CANCELLED
        assert "batches/abc123" in fake_client.aio.batches.cancel_calls


# ---------------------------------------------------------------------------
# BatchStatus terminality
# ---------------------------------------------------------------------------


class TestBatchStatus:
    def test_terminal_states(self) -> None:
        from draper.construction.batch.types import BatchJobInfo

        for status in (
            BatchStatus.COMPLETED,
            BatchStatus.FAILED,
            BatchStatus.CANCELLED,
            BatchStatus.EXPIRED,
        ):
            info = BatchJobInfo(batch_id="x", provider="openai", status=status, request_count=0)
            assert info.is_terminal, f"{status} should be terminal"

    def test_non_terminal_states(self) -> None:
        from draper.construction.batch.types import BatchJobInfo

        for status in (BatchStatus.PENDING, BatchStatus.IN_PROGRESS):
            info = BatchJobInfo(batch_id="x", provider="openai", status=status, request_count=0)
            assert not info.is_terminal


# ---------------------------------------------------------------------------
# BytesIO buffer is well-formed JSONL for OpenAI submit
# ---------------------------------------------------------------------------


def test_input_jsonl_is_newline_terminated() -> None:
    req = BatchRequest(
        custom_id="x",
        system=None,
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-4o-mini",
    )
    rendered = OpenAIBatchClient._render_input_jsonl([req])
    assert rendered.endswith("\n")
    # Each non-empty line is valid JSON
    for line in rendered.splitlines():
        if line.strip():
            json.loads(line)


def test_input_jsonl_can_be_uploaded_as_bytesio() -> None:
    # Sanity check that the bytes we produce are what OpenAI expects.
    req = BatchRequest(
        custom_id="x",
        system=None,
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-4o-mini",
    )
    rendered = OpenAIBatchClient._render_input_jsonl([req])
    buffer = io.BytesIO(rendered.encode("utf-8"))
    buffer.name = "draper_batch.jsonl"
    assert buffer.read().decode("utf-8") == rendered
