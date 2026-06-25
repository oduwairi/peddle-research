"""Content-block translators for vision-enabled batch requests.

Each ``BatchRequest`` message has ``content`` that is either a plain string
(text-only, legacy shape) or a list of internal ContentBlock dicts. Two
block forms are defined in :mod:`draper.construction.batch.types`:

- ``{"type": "text", "text": str}``
- ``{"type": "image_url", "url": str, "mime_type": str | None}``

The functions below convert from our internal block shape to each
provider's native wire format. The text-only string path is left alone:
callers that pass a string never touch this module.
"""

from __future__ import annotations

from typing import Any

# Canonical default for AdFlex hotlinks (the inventory we care about now).
DEFAULT_IMAGE_MIME = "image/jpeg"


def _ensure_list(content: Any) -> list[dict[str, Any]]:
    """Internal: validate that a message content is a non-empty list of blocks."""
    if not isinstance(content, list):
        msg = "translate_*_content requires list content; string content is provider-native"
        raise TypeError(msg)
    if not content:
        msg = "content block list must not be empty"
        raise ValueError(msg)
    return content


def translate_openai_content(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert internal blocks to OpenAI Chat Completions vision blocks.

    OpenAI wire shape:
      ``{"type": "text", "text": ...}``
      ``{"type": "image_url", "image_url": {"url": ..., "detail": "auto"}}``
    """
    _ensure_list(content)
    out: list[dict[str, Any]] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            out.append({"type": "text", "text": block["text"]})
        elif btype == "image_url":
            out.append(
                {
                    "type": "image_url",
                    "image_url": {"url": block["url"], "detail": "auto"},
                }
            )
        else:
            msg = f"unknown content block type for OpenAI: {btype!r}"
            raise ValueError(msg)
    return out


def translate_anthropic_content(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert internal blocks to Anthropic Messages vision blocks.

    Anthropic wire shape:
      ``{"type": "text", "text": ...}``
      ``{"type": "image", "source": {"type": "url", "url": ...}}``
    """
    _ensure_list(content)
    out: list[dict[str, Any]] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            out.append({"type": "text", "text": block["text"]})
        elif btype == "image_url":
            out.append(
                {
                    "type": "image",
                    "source": {"type": "url", "url": block["url"]},
                }
            )
        else:
            msg = f"unknown content block type for Anthropic: {btype!r}"
            raise ValueError(msg)
    return out


def translate_gemini_parts(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert internal blocks to a list of ``Part``-constructor kwargs.

    Returns plain dicts that the caller passes into ``gtypes.Part``-equivalent
    constructors. Keeping the SDK type out of this module avoids forcing
    ``google-genai`` to be importable wherever this helper is used (tests
    that exercise this function don't need the SDK).

    Two output shapes:
      ``{"text": "..."}``                      -> ``Part(text=...)``
      ``{"file_uri": "...", "mime_type": ...}`` -> ``Part.from_uri(...)``

    Note on external URLs: Gemini's ``file_uri`` is documented for GCS and
    File API references. AdFlex hotlinks may or may not work depending on
    server-side fetch behavior; Phase 2 of the image-brief plan validates
    this against real creatives. If they don't, callers can pre-download
    bytes and pass ``image_url`` blocks routed through a future
    ``image_bytes`` variant; that's deliberately out of scope here to keep
    the Phase-1 refactor minimal.
    """
    _ensure_list(content)
    out: list[dict[str, Any]] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            out.append({"text": block["text"]})
        elif btype == "image_url":
            out.append(
                {
                    "file_uri": block["url"],
                    "mime_type": block.get("mime_type") or DEFAULT_IMAGE_MIME,
                }
            )
        else:
            msg = f"unknown content block type for Gemini: {btype!r}"
            raise ValueError(msg)
    return out
