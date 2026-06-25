"""Unit tests for ``pipeline.ingest_responses``.

Covers three branches that had zero direct test coverage:

C1. invalid_brief path — ``build_brief`` raises ``ValueError``/``TypeError``
    on a malformed brief dict, incrementing ``stats.missing_brief`` and
    appending a ``stage="parse"`` rejection with reason ``"invalid_brief:…"``.

C2. leak skipped for image_brief — ``bundle.leak is None``, so a deliverable
    that reproduces the copy verbatim must NOT increment ``stats.leak_failed``.

C3. content_bridge gate failure — monkeypatching the bundle's
    ``content_bridge`` callable to return a failing result increments
    ``stats.content_bridge_failed`` and appends a
    ``stage="content_bridge"`` rejection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from draper.construction_v2 import pipeline
from draper.construction_v2.config import ConstructionV2Config, ProviderConfig
from draper.construction_v2.ingest.image_brief_fidelity import ContentBridgeResult
from draper.construction_v2.teacher.image_brief_single_pass import CAPTION_RAW_KEY

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# A minimal teacher response that parse_response (the general single-pass
# parser) will accept: <think> block of sufficient length + a deliverable.
_THINK_TEXT = (
    "I'll anchor to the calm premium product-photography brand feel "
    "called out in brand_guidelines and lead with the founder-hands hero."
)
_IMAGE_BRIEF_PROSE = (
    "Create a founder-hands hero shot of the bottle over a warm wooden desk "
    "in natural morning light."
)

# The deliverable for image_brief includes the <image_brief> tag so the
# fidelity gate (check_image_brief_fidelity) can find it.
_DELIVERABLE = f"<image_brief>{_IMAGE_BRIEF_PROSE}</image_brief>"

# A caption whose content words overlap with the deliverable prose, keeping
# the fidelity gate happy (MIN_CAPTION_OVERLAP = 0.3).
_CAPTION = "founder hands holding bottle over a warm wooden desk in natural light"


def _make_response_content(think: str = _THINK_TEXT, deliverable: str = _DELIVERABLE) -> str:
    """Assemble a raw teacher response as stored in ``responses_raw.jsonl``."""
    return f"<think>\n{think}\n</think>\n\n{deliverable}"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dicts as JSONL (parents created automatically)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------


def _image_brief_config(tmp_path: Path) -> ConstructionV2Config:
    """Build a minimal ConstructionV2Config wired for the image_brief skill."""
    cfg = ConstructionV2Config(
        skill="image_brief",
        output_dir=str(tmp_path / "constructed_v2"),
        final_dir=str(tmp_path / "final_v2"),
        audit_dir=str(tmp_path / "constructed_v2" / "_audit"),
        providers={"openai": ProviderConfig(model="gpt-5.4-mini")},
    )
    # Point the briefs cache at a controlled path inside tmp_path.
    cfg.single_pass.briefs_cache_path = str(
        tmp_path / "constructed_v2" / "image_brief" / "briefs.jsonl"
    )
    # Point scored_ads at a controlled path.
    cfg.selection.scored_ads_path = str(tmp_path / "scored_ads.jsonl")
    return cfg


# ---------------------------------------------------------------------------
# Scored-ad JSONL row helper
# ---------------------------------------------------------------------------


def _scored_ad_row(
    ad_id: str,
    *,
    platform: str = "meta",
    headline: str = "Hero headline",
    body: str = "Body copy for the hero product.",
    caption: str | None = _CAPTION,
) -> dict[str, Any]:
    """Build a minimal scored-ad JSONL row readable by ``load_source_ads_by_id``."""
    raw: dict[str, Any] = {
        "ad_id": ad_id,
        "platform": platform,
        "ad_copy": {
            "headline": headline,
            "body": body,
            "description": "",
            "cta": "Shop now",
        },
    }
    if caption is not None:
        raw[CAPTION_RAW_KEY] = caption
    return {"ad": raw, "composite_score": 0.75}


# ---------------------------------------------------------------------------
# Brief dict helpers
# ---------------------------------------------------------------------------


def _valid_image_brief_dict(ad_id: str) -> dict[str, Any]:
    """A brief dict the teacher would emit — no injected fields yet."""
    return {
        "task": "Give me the image brief for the meta ad below.",
        "objective": "awareness",
        "product": {"tone_signals": ["calm-premium"]},
        "creative": {
            "brand_guidelines": (
                "Calm, premium founder-brand feel; soft natural-light "
                "product photography."
            ),
            "on_creative_text": [],
            "key_facts": [],
        },
    }


def _invalid_image_brief_dict() -> dict[str, Any]:
    """A brief dict that will cause ImageBriefInput validation to fail.

    ``creative.brand_guidelines`` is empty (stripped), which triggers the
    non-empty validator in the schema.
    """
    return {
        "task": "Give me the image brief.",
        "objective": "awareness",
        "product": {"tone_signals": ["warm"]},
        "creative": {
            "brand_guidelines": "   ",  # fails non-empty validator
        },
    }


# ---------------------------------------------------------------------------
# C1 — invalid_brief path
# ---------------------------------------------------------------------------


class TestIngestInvalidBrief:
    def test_invalid_brief_increments_missing_brief_and_records_rejection(
        self, tmp_path: Path
    ) -> None:
        """A brief dict that fails the skill's model validation is counted as
        ``missing_brief`` and generates a ``stage="parse"`` rejection whose
        reason starts with ``"invalid_brief:"``.
        """
        ad_id = "ad-invalid-brief-01"
        cfg = _image_brief_config(tmp_path)

        # Write scored ad so load_source_ads_by_id finds it.
        _write_jsonl(
            Path(cfg.selection.scored_ads_path),
            [_scored_ad_row(ad_id)],
        )

        # Write the bad brief dict into the briefs cache.
        briefs_path = Path(cfg.single_pass.briefs_cache_path)
        _write_jsonl(
            briefs_path,
            [{"ad_id": ad_id, "brief": _invalid_image_brief_dict()}],
        )

        # Write a valid-looking response (the row must survive parse_response
        # before reaching build_brief).
        responses_path = pipeline.responses_path(cfg)
        _write_jsonl(
            responses_path,
            [{"ad_id": ad_id, "content": _make_response_content(), "model": "gpt-5.4-mini"}],
        )

        result = pipeline.ingest_responses(cfg, input_path=responses_path)

        assert result.stats.total_input == 1
        assert result.stats.missing_brief == 1
        assert result.stats.passed == 0
        assert len(result.rejections) == 1
        rejection = result.rejections[0]
        assert rejection.ad_id == ad_id
        assert rejection.stage == "parse"
        assert rejection.reason.startswith("invalid_brief:")


# ---------------------------------------------------------------------------
# C2 — leak guard skipped for image_brief
# ---------------------------------------------------------------------------


class TestIngestLeakSkippedForImageBrief:
    def test_verbatim_copy_in_deliverable_does_not_trip_leak_for_image_brief(
        self, tmp_path: Path
    ) -> None:
        """The image_brief skill has ``bundle.leak = None``.

        Even when the deliverable reproduces the ad copy verbatim, the leak
        counter must remain 0 because the None-leak branch is not reached.
        """
        ad_id = "ad-leak-skip-01"
        headline = "Hero headline"
        cfg = _image_brief_config(tmp_path)

        _write_jsonl(
            Path(cfg.selection.scored_ads_path),
            [_scored_ad_row(ad_id, headline=headline)],
        )
        _write_jsonl(
            Path(cfg.single_pass.briefs_cache_path),
            [{"ad_id": ad_id, "brief": _valid_image_brief_dict(ad_id)}],
        )

        # Deliverable quotes the headline verbatim — would fail a leak check
        # if one were wired.  The image_brief skill has bundle.leak = None
        # so this must pass through without tripping stats.leak_failed.
        deliverable_with_copy = (
            f"<image_brief>Create a hero shot showing the message "
            f'"{headline}" over the product.</image_brief>'
        )
        # The think must reference brand_guidelines for the grounding gate.
        think_with_guidelines = (
            "I'll lean on the calm premium product-photography brand feel "
            "called out in the brand_guidelines block and hero the bottle."
        )
        responses_path = pipeline.responses_path(cfg)
        _write_jsonl(
            responses_path,
            [
                {
                    "ad_id": ad_id,
                    "content": _make_response_content(
                        think=think_with_guidelines,
                        deliverable=deliverable_with_copy,
                    ),
                    "model": "gpt-5.4-mini",
                }
            ],
        )

        result = pipeline.ingest_responses(cfg, input_path=responses_path)

        assert result.stats.leak_failed == 0
        # The row either passed all gates or was rejected for a different reason.
        # The important invariant is that leak was never the cause of rejection.
        assert not any(r.stage == "leak" for r in result.rejections)


# ---------------------------------------------------------------------------
# C3 — content_bridge gate failure
# ---------------------------------------------------------------------------


class TestIngestContentBridgeGate:
    def test_content_bridge_failure_increments_counter_and_records_rejection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the content_bridge callable returns a failing result, the
        ingest loop increments ``stats.content_bridge_failed`` and appends a
        rejection with ``stage="content_bridge"``.
        """
        ad_id = "ad-cb-fail-01"
        cfg = _image_brief_config(tmp_path)

        _write_jsonl(
            Path(cfg.selection.scored_ads_path),
            [_scored_ad_row(ad_id)],
        )
        _write_jsonl(
            Path(cfg.single_pass.briefs_cache_path),
            [{"ad_id": ad_id, "brief": _valid_image_brief_dict(ad_id)}],
        )

        # A well-formed response that passes fidelity and grounding.
        think_good = (
            "I'll lean on the calm premium product-photography brand feel "
            "called out in the brand_guidelines block and hero the bottle."
        )
        responses_path = pipeline.responses_path(cfg)
        _write_jsonl(
            responses_path,
            [
                {
                    "ad_id": ad_id,
                    "content": _make_response_content(think=think_good),
                    "model": "gpt-5.4-mini",
                }
            ],
        )

        # Monkeypatch: replace the real content_bridge callable on the bundle
        # so it always returns a deterministic failure.
        failing_cb = MagicMock(
            return_value=ContentBridgeResult(
                passed=False,
                reason="content_bridge_ungrounded",
                detail="a fabricated item",
            )
        )

        original_get_bundle = pipeline.get_bundle

        def _patched_get_bundle(skill: str) -> Any:
            bundle = original_get_bundle(skill)
            # Return a modified copy with the failing content_bridge injected.
            from dataclasses import replace

            return replace(bundle, content_bridge=failing_cb)

        monkeypatch.setattr(pipeline, "get_bundle", _patched_get_bundle)

        result = pipeline.ingest_responses(cfg, input_path=responses_path)

        assert result.stats.total_input == 1
        assert result.stats.content_bridge_failed == 1
        assert result.stats.passed == 0
        assert len(result.rejections) == 1
        rejection = result.rejections[0]
        assert rejection.ad_id == ad_id
        assert rejection.stage == "content_bridge"
        assert "content_bridge_ungrounded" in rejection.reason
