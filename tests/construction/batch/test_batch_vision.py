"""Tests for vision-enabled batch content-block translation.

Covers:
- Per-provider content-block translators (text + image_url)
- End-to-end wire format through each provider's submit/render path
- Mixed-content requests round-trip alongside text-only requests in the
  same batch (regression guard for the list-vs-string content branch)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from draper.construction.batch.content_blocks import (
    translate_anthropic_content,
    translate_gemini_parts,
    translate_openai_content,
)
from draper.construction.batch.openai_client import OpenAIBatchClient
from draper.construction.batch.types import BatchRequest

# ---------------------------------------------------------------------------
# Internal translators
# ---------------------------------------------------------------------------


class TestTranslateOpenAI:
    def test_text_only_blocks(self) -> None:
        out = translate_openai_content([{"type": "text", "text": "hi"}])
        assert out == [{"type": "text", "text": "hi"}]

    def test_image_url_block(self) -> None:
        out = translate_openai_content([{"type": "image_url", "url": "https://example.com/x.jpg"}])
        assert out == [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/x.jpg", "detail": "auto"},
            }
        ]

    def test_mixed_blocks_preserve_order(self) -> None:
        out = translate_openai_content(
            [
                {"type": "text", "text": "describe:"},
                {"type": "image_url", "url": "https://example.com/a.jpg"},
                {"type": "text", "text": "in detail."},
            ]
        )
        assert [b["type"] for b in out] == ["text", "image_url", "text"]
        assert out[1]["image_url"]["url"] == "https://example.com/a.jpg"

    def test_rejects_unknown_block(self) -> None:
        with pytest.raises(ValueError, match="unknown content block"):
            translate_openai_content([{"type": "audio", "url": "x"}])

    def test_rejects_empty_list(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            translate_openai_content([])

    def test_rejects_non_list(self) -> None:
        with pytest.raises(TypeError, match="list content"):
            translate_openai_content("plain string")  # type: ignore[arg-type]


class TestTranslateAnthropic:
    def test_text_only_blocks(self) -> None:
        out = translate_anthropic_content([{"type": "text", "text": "hi"}])
        assert out == [{"type": "text", "text": "hi"}]

    def test_image_url_block(self) -> None:
        out = translate_anthropic_content(
            [{"type": "image_url", "url": "https://example.com/x.jpg"}]
        )
        assert out == [
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/x.jpg"},
            }
        ]

    def test_mixed_blocks_preserve_order(self) -> None:
        out = translate_anthropic_content(
            [
                {"type": "image_url", "url": "https://example.com/a.jpg"},
                {"type": "text", "text": "What is this?"},
            ]
        )
        assert out[0]["type"] == "image"
        assert out[1]["type"] == "text"
        assert out[0]["source"]["url"] == "https://example.com/a.jpg"

    def test_rejects_unknown_block(self) -> None:
        with pytest.raises(ValueError, match="unknown content block"):
            translate_anthropic_content([{"type": "audio", "url": "x"}])


class TestTranslateGeminiParts:
    def test_text_only_blocks(self) -> None:
        out = translate_gemini_parts([{"type": "text", "text": "hi"}])
        assert out == [{"text": "hi"}]

    def test_image_url_block_default_mime(self) -> None:
        out = translate_gemini_parts([{"type": "image_url", "url": "https://example.com/x.jpg"}])
        assert out == [{"file_uri": "https://example.com/x.jpg", "mime_type": "image/jpeg"}]

    def test_image_url_block_explicit_mime(self) -> None:
        out = translate_gemini_parts(
            [
                {
                    "type": "image_url",
                    "url": "https://example.com/x.png",
                    "mime_type": "image/png",
                }
            ]
        )
        assert out == [{"file_uri": "https://example.com/x.png", "mime_type": "image/png"}]

    def test_mixed_blocks_preserve_order(self) -> None:
        out = translate_gemini_parts(
            [
                {"type": "text", "text": "describe:"},
                {"type": "image_url", "url": "https://example.com/a.jpg"},
            ]
        )
        assert "text" in out[0]
        assert "file_uri" in out[1]


# ---------------------------------------------------------------------------
# OpenAI: end-to-end wire format with vision content
# ---------------------------------------------------------------------------


class TestOpenAIVisionWire:
    def test_image_request_renders_native_blocks(self) -> None:
        req = BatchRequest(
            custom_id="cap-00000",
            system=None,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Caption this."},
                        {
                            "type": "image_url",
                            "url": "https://cdn.adflex.io/x.jpg",
                        },
                    ],
                }
            ],
            model="gpt-5.4-mini",
            max_tokens=512,
        )
        line = OpenAIBatchClient._render_input_jsonl([req]).strip()
        obj = json.loads(line)
        content = obj["body"]["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "Caption this."}
        assert content[1] == {
            "type": "image_url",
            "image_url": {
                "url": "https://cdn.adflex.io/x.jpg",
                "detail": "auto",
            },
        }

    def test_mixed_text_and_vision_in_same_batch(self) -> None:
        text_req = BatchRequest(
            custom_id="t-0",
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-5.4-mini",
        )
        vision_req = BatchRequest(
            custom_id="v-0",
            system=None,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Look:"},
                        {"type": "image_url", "url": "https://x/y.jpg"},
                    ],
                }
            ],
            model="gpt-5.4-mini",
        )
        lines = OpenAIBatchClient._render_input_jsonl([text_req, vision_req]).strip().splitlines()
        text_obj = json.loads(lines[0])
        vision_obj = json.loads(lines[1])
        # Text request: legacy string content path
        assert text_obj["body"]["messages"][1]["content"] == "hi"
        # Vision request: list-of-blocks path
        assert isinstance(vision_obj["body"]["messages"][0]["content"], list)


# ---------------------------------------------------------------------------
# Anthropic: end-to-end wire format with vision content
# ---------------------------------------------------------------------------


class _RecordingMessagesBatches:
    """Captures the payload passed to ``messages.batches.create``."""

    def __init__(self) -> None:
        self.captured: list[dict[str, Any]] = []

    async def create(self, *, requests: list[dict[str, Any]]) -> Any:
        self.captured = requests

        class _Batch:
            id = "batch_recording_001"
            processing_status = "in_progress"
            request_counts = None
            created_at = ""
            expires_at = ""

        return _Batch()


class _RecordingAnthropicClient:
    def __init__(self) -> None:
        batches = _RecordingMessagesBatches()

        class _Messages:
            def __init__(self) -> None:
                self.batches = batches

        self.messages = _Messages()
        self._batches_obj = batches

    @property
    def captured(self) -> list[dict[str, Any]]:
        return self._batches_obj.captured


class TestAnthropicVisionWire:
    @pytest.mark.asyncio
    async def test_image_request_translates_to_anthropic_blocks(self) -> None:
        from draper.construction.batch.anthropic_client import AnthropicBatchClient

        fake = _RecordingAnthropicClient()
        client = AnthropicBatchClient(client=fake)  # type: ignore[arg-type]
        req = BatchRequest(
            custom_id="cap-0",
            system="you describe images",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe:"},
                        {
                            "type": "image_url",
                            "url": "https://cdn.adflex.io/x.jpg",
                        },
                    ],
                }
            ],
            model="claude-sonnet-4-6",
        )
        await client.submit([req])
        assert len(fake.captured) == 1
        params = fake.captured[0]["params"]
        content = params["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "Describe:"}
        assert content[1] == {
            "type": "image",
            "source": {"type": "url", "url": "https://cdn.adflex.io/x.jpg"},
        }
        assert params["system"] == "you describe images"


# ---------------------------------------------------------------------------
# Gemini: _to_inlined wire format with vision content
# ---------------------------------------------------------------------------


class TestGeminiVisionWire:
    def test_image_request_emits_file_uri_part(self) -> None:
        from draper.construction.batch.gemini_client import GeminiBatchClient

        req = BatchRequest(
            custom_id="cap-0",
            system=None,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Caption this."},
                        {
                            "type": "image_url",
                            "url": "https://cdn.adflex.io/x.jpg",
                            "mime_type": "image/jpeg",
                        },
                    ],
                }
            ],
            model="gemini-3.5-flash",
            max_tokens=512,
        )
        inlined = GeminiBatchClient._to_inlined(req, thinking_budget=0)
        # InlinedRequest.contents is list[Content]; first message user role.
        contents = inlined.contents
        assert contents is not None
        assert len(contents) == 1
        parts = contents[0].parts or []
        assert len(parts) == 2
        # First part is text
        assert parts[0].text == "Caption this."
        # Second part is a file-data ref (vision); the SDK normalizes
        # Part.from_uri into a Part with file_data populated.
        fd = getattr(parts[1], "file_data", None)
        assert fd is not None
        assert fd.file_uri == "https://cdn.adflex.io/x.jpg"
        assert fd.mime_type == "image/jpeg"

    def test_text_only_unchanged_path(self) -> None:
        """Regression: text-only requests still take the legacy fast path."""
        from draper.construction.batch.gemini_client import GeminiBatchClient

        req = BatchRequest(
            custom_id="t-0",
            system="sys",
            messages=[{"role": "user", "content": "just text"}],
            model="gemini-3.5-flash",
            max_tokens=512,
        )
        inlined = GeminiBatchClient._to_inlined(req, thinking_budget=0)
        contents = inlined.contents
        assert contents is not None
        parts = contents[0].parts or []
        assert len(parts) == 1
        assert parts[0].text == "just text"
