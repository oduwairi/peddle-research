"""Unit tests for ``draper.construction_v2.teacher.brief_extractor``.

The chat-mode path (:func:`extract_brief`) goes through the Anthropic
SDK; we exercise it via a fake async client that returns canned
``content`` blocks. The batch-mode helper
(:func:`build_brief_batch_requests`) and the raw-content parser
(:func:`parse_brief_response_content`) are pure functions and tested
directly.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from draper.construction_v2.config import BriefExtractionConfig
from draper.construction_v2.dataset.source_selector import SourceAd
from draper.construction_v2.schemas.brief import Brief
from draper.construction_v2.teacher.brief_extractor import (
    AD_COPY_TASK,
    BRIEF_EXTRACTION_SYSTEM_PROMPT,
    build_brief_batch_requests,
    extract_brief,
    parse_brief_response_content,
)

_VALID_BRIEF_PAYLOAD: dict[str, Any] = {
    "product": {
        "name": "Compliantly",
        "description": "background checks for ops teams",
        "category": "HR-tech",
        "key_features": ["SOC 2 audit trail"],
        "unique_selling_points": ["72-hour turnaround"],
        "tone_signals": ["professional", "clipped"],
    },
    "bridge": {
        "positioning": "speed-first vs legacy HR vendors",
        "target_audience": "Series-A operations leads",
        "angle": "aspirational founder identity",
        "buyer_pain": "compliance review blocks weekly hires",
    },
    "platform": "meta",
}


class _ToolUseBlock:
    """Mimics ``anthropic.types.ToolUseBlock`` minimally."""

    type = "tool_use"

    def __init__(self, tool_input: dict[str, Any]) -> None:
        self.input = tool_input


class _TextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResponse:
    def __init__(self, content: list[Any]) -> None:
        self.content = content


class _FakeMessages:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.last_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> _FakeResponse:
        self.last_kwargs = kwargs
        return self._response


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.messages = _FakeMessages(response)


# ---------------------------------------------------------------------------
# parse_brief_response_content
# ---------------------------------------------------------------------------


def test_parse_brief_response_content_raw_json(sample_source_ad: SourceAd) -> None:
    raw = json.dumps(_VALID_BRIEF_PAYLOAD)
    brief = parse_brief_response_content(raw, ad=sample_source_ad)
    assert isinstance(brief, Brief)
    assert brief.task == AD_COPY_TASK
    assert brief.platform == "meta"
    assert brief.product.tone_signals == ["professional", "clipped"]
    assert brief.bridge.angle == "aspirational founder identity"


def test_parse_brief_response_content_fenced_json(sample_source_ad: SourceAd) -> None:
    raw = "```json\n" + json.dumps(_VALID_BRIEF_PAYLOAD) + "\n```"
    brief = parse_brief_response_content(raw, ad=sample_source_ad)
    assert brief.platform == "meta"


def test_parse_brief_response_content_platform_alias(sample_source_ad: SourceAd) -> None:
    """Source ad platforms should be coerced to the v2 surface enum."""
    payload = {**_VALID_BRIEF_PAYLOAD, "platform": "facebook"}
    brief = parse_brief_response_content(json.dumps(payload), ad=sample_source_ad)
    assert brief.platform == "meta"


def test_parse_brief_response_content_rejects_empty(sample_source_ad: SourceAd) -> None:
    with pytest.raises(ValueError, match="Empty brief"):
        parse_brief_response_content("   ", ad=sample_source_ad)


def test_parse_brief_response_content_rejects_invalid_json(
    sample_source_ad: SourceAd,
) -> None:
    with pytest.raises(ValueError, match="non-JSON"):
        parse_brief_response_content("not a json object", ad=sample_source_ad)


def test_parse_brief_response_content_rejects_missing_tone_signals(
    sample_source_ad: SourceAd,
) -> None:
    """tone_signals is required and non-empty at the schema layer."""
    bad = {
        **_VALID_BRIEF_PAYLOAD,
        "product": {**_VALID_BRIEF_PAYLOAD["product"], "tone_signals": []},
    }
    with pytest.raises(ValueError, match="Brief validation"):
        parse_brief_response_content(json.dumps(bad), ad=sample_source_ad)


def test_parse_brief_response_content_rejects_missing_bridge_required(
    sample_source_ad: SourceAd,
) -> None:
    bad = {
        **_VALID_BRIEF_PAYLOAD,
        "bridge": {"positioning": "x", "target_audience": "y"},  # no angle/buyer_pain
    }
    with pytest.raises(ValueError, match="Brief validation"):
        parse_brief_response_content(json.dumps(bad), ad=sample_source_ad)


# ---------------------------------------------------------------------------
# build_brief_batch_requests
# ---------------------------------------------------------------------------


def test_build_brief_batch_requests_shape(sample_source_ad: SourceAd) -> None:
    cfg = BriefExtractionConfig(model="claude-haiku-4-5", max_tokens=500, temperature=0.2)
    requests = build_brief_batch_requests([sample_source_ad, sample_source_ad], cfg)
    assert len(requests) == 2
    for r in requests:
        assert r.custom_id == f"brief-{sample_source_ad.ad_id}"
        assert r.system == BRIEF_EXTRACTION_SYSTEM_PROMPT
        assert r.model == "claude-haiku-4-5"
        assert r.max_tokens == 500
        assert r.temperature == pytest.approx(0.2)
        assert len(r.messages) == 1
        assert r.messages[0]["role"] == "user"
        # User payload should mention the ad's ad_id.
        assert sample_source_ad.ad_id in r.messages[0]["content"]


def test_build_brief_batch_requests_custom_id_safe_charset(
    sample_source_ad: SourceAd,
) -> None:
    """custom_id must satisfy Anthropic's ``^[a-zA-Z0-9_-]{1,64}$``."""
    import re

    cfg = BriefExtractionConfig()
    requests = build_brief_batch_requests([sample_source_ad], cfg)
    pattern = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
    for r in requests:
        assert pattern.match(r.custom_id), r.custom_id


# ---------------------------------------------------------------------------
# extract_brief (chat mode)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_brief_tool_use_path(sample_source_ad: SourceAd) -> None:
    """When the teacher emits a tool_use block we parse from .input directly."""
    response = _FakeResponse(content=[_ToolUseBlock(_VALID_BRIEF_PAYLOAD)])
    client = _FakeClient(response)
    cfg = BriefExtractionConfig()
    brief = await extract_brief(sample_source_ad, cfg, client=client)  # type: ignore[arg-type]
    assert isinstance(brief, Brief)
    assert brief.platform == "meta"
    assert client.messages.last_kwargs is not None
    assert client.messages.last_kwargs["system"] == BRIEF_EXTRACTION_SYSTEM_PROMPT
    assert client.messages.last_kwargs["tool_choice"]["name"] == "emit_brief"


@pytest.mark.asyncio
async def test_extract_brief_text_fallback(sample_source_ad: SourceAd) -> None:
    """When the teacher ignores tool_choice we fall back to text parsing."""
    raw = json.dumps(_VALID_BRIEF_PAYLOAD)
    response = _FakeResponse(content=[_TextBlock(raw)])
    client = _FakeClient(response)
    cfg = BriefExtractionConfig()
    brief = await extract_brief(sample_source_ad, cfg, client=client)  # type: ignore[arg-type]
    assert brief.platform == "meta"


@pytest.mark.asyncio
async def test_extract_brief_no_content_raises(sample_source_ad: SourceAd) -> None:
    response = _FakeResponse(content=[])
    client = _FakeClient(response)
    cfg = BriefExtractionConfig()
    with pytest.raises(ValueError, match="no content"):
        await extract_brief(sample_source_ad, cfg, client=client)  # type: ignore[arg-type]
