"""Brief schema: round-trip + canonical JSON byte-stability."""

from __future__ import annotations

import json

import pytest

from draper.construction_v2.schemas.brief import (
    SUPPORTED_PLATFORMS,
    Brief,
    BriefBridge,
    BriefProduct,
    canonical_json,
)


def test_brief_round_trip(sample_brief: Brief) -> None:
    payload = sample_brief.model_dump(mode="json")
    restored = Brief.model_validate(payload)
    assert restored == sample_brief


def test_brief_rejects_extra_fields() -> None:
    with pytest.raises(ValueError, match="extra_forbidden|Extra inputs"):
        Brief.model_validate(
            {
                "product": {
                    "name": "x",
                    "description": "y",
                    "tone_signals": ["crisp"],
                },
                "bridge": {
                    "positioning": "p",
                    "target_audience": "t",
                    "angle": "a",
                    "buyer_pain": "b",
                },
                "platform": "meta",
                "extra_field": "should be rejected",
            }
        )


def test_product_rejects_empty_tone_signals() -> None:
    with pytest.raises(ValueError, match="tone_signals"):
        BriefProduct(
            name="x",
            description="y",
            tone_signals=[],
        )


def test_product_rejects_whitespace_only_tone_signals() -> None:
    with pytest.raises(ValueError, match="tone_signals"):
        BriefProduct(
            name="x",
            description="y",
            tone_signals=["  ", ""],
        )


def test_product_trims_tone_signals() -> None:
    product = BriefProduct(
        name="x",
        description="y",
        tone_signals=["  crisp  ", " bold"],
    )
    assert product.tone_signals == ["crisp", "bold"]


@pytest.mark.parametrize("platform", SUPPORTED_PLATFORMS)
def test_brief_accepts_supported_platforms(platform: str) -> None:
    brief = Brief(
        task="Write ad copy.",
        product=BriefProduct(name="x", description="y", tone_signals=["crisp"]),
        bridge=BriefBridge(positioning="p", target_audience="t", angle="a", buyer_pain="b"),
        platform=platform,  # type: ignore[arg-type]
    )
    assert brief.platform == platform


def test_brief_rejects_unsupported_platform() -> None:
    with pytest.raises(ValueError):
        Brief(
            task="Write ad copy.",
            product=BriefProduct(name="x", description="y", tone_signals=["crisp"]),
            bridge=BriefBridge(positioning="p", target_audience="t", angle="a", buyer_pain="b"),
            platform="snapchat",  # type: ignore[arg-type]
        )


def test_canonical_json_stable_across_field_declaration_order(
    sample_brief: Brief,
) -> None:
    """Reordering field-init order must not change canonical bytes."""
    payload = sample_brief.model_dump(mode="json")
    permuted: dict[str, object] = dict(reversed(list(payload.items())))
    # Also reverse nested dicts.
    permuted["product"] = dict(reversed(list(payload["product"].items())))
    permuted["bridge"] = dict(reversed(list(payload["bridge"].items())))
    rebuilt = Brief.model_validate(permuted)
    assert canonical_json(rebuilt) == canonical_json(sample_brief)


def test_canonical_json_parses_back(sample_brief: Brief) -> None:
    canonical = canonical_json(sample_brief)
    parsed = json.loads(canonical)
    assert parsed["product"]["name"] == sample_brief.product.name
    assert parsed["platform"] == sample_brief.platform


def test_canonical_json_no_trailing_newline(sample_brief: Brief) -> None:
    canonical = canonical_json(sample_brief)
    assert not canonical.endswith("\n")
    assert not canonical.endswith(" ")


def test_brief_accepts_minimal_grounded_brief() -> None:
    """Sparse ad (hook only) must validate with optional fields null.

    Grounding contract: only ``tone_signals``, ``angle``, ``buyer_pain``,
    and ``platform`` are required. A teacher faced with a one-line hook
    ad must be able to emit a valid brief without fabricating product
    facts.
    """
    brief = Brief(
        task="Write ad copy.",
        product=BriefProduct(tone_signals=["playful"]),
        bridge=BriefBridge(angle="DTC-snark", buyer_pain="menu fatigue"),
        platform="meta",
    )
    assert brief.product.name is None
    assert brief.product.description is None
    assert brief.product.key_features == []
    assert brief.bridge.positioning is None
    assert brief.bridge.target_audience is None


def test_brief_bridge_requires_angle_and_buyer_pain() -> None:
    """``angle`` and ``buyer_pain`` are the only required bridge fields."""
    with pytest.raises(ValueError, match="angle"):
        BriefBridge(buyer_pain="b")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="buyer_pain"):
        BriefBridge(angle="a")  # type: ignore[call-arg]
