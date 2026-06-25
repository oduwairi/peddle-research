"""Tests for the GOLD reference-eval sentinel."""

from __future__ import annotations

from draper.evaluation.gold import (
    GOLD_CONFIG,
    gold_inference_from_brief,
    gold_inferences_from_briefs,
    is_gold,
)
from draper.evaluation.schemas import Brief


def _brief(example_id: str, gold_text: str) -> Brief:
    return Brief(
        example_id=example_id,
        task_format="copywriting",
        platform="facebook",
        vertical="cpg",
        source_tiers=["high"],
        construction_model="claude-3.5-sonnet",
        system="You are an ad copywriter.",
        user="Write a Meta ad for IL MAKIAGE primer.",
        reference_assistant=gold_text,
    )


def test_is_gold_matches_bare_and_suffix() -> None:
    """is_gold accepts the bare sentinel and any GOLD_<suffix> split variant."""
    # Positive cases
    assert is_gold("GOLD")        # bare sentinel (v1 split)
    assert is_gold("GOLD_v2")     # v2 split variant
    assert is_gold("GOLD_v3")     # hypothetical future split variant
    # Negative cases
    assert not is_gold("gold")    # case-sensitive
    assert not is_gold("GOLDEN")  # prefix collision — must have underscore separator
    assert not is_gold("NOTGOLD") # unrelated name
    assert not is_gold("C")


def test_gold_constant() -> None:
    assert GOLD_CONFIG == "GOLD"


def test_gold_inference_carries_reference_assistant() -> None:
    b = _brief("ex1", "Try Before You Buy. Flawless finish.")
    inf = gold_inference_from_brief(b)
    assert inf.example_id == "ex1"
    assert inf.config == GOLD_CONFIG
    assert inf.assistant_text == "Try Before You Buy. Flawless finish."
    assert inf.arm == "arm1"
    # GOLD didn't go through a runner — these should be zero/empty.
    assert inf.latency_ms == 0
    assert inf.input_tokens == 0
    assert inf.output_tokens == 0
    assert inf.tool_calls == []
    assert inf.campaign is None


def test_gold_inferences_keyed_by_example_id() -> None:
    briefs = [
        _brief("ex1", "Ad copy 1."),
        _brief("ex2", "Ad copy 2."),
        _brief("ex3", "Ad copy 3."),
    ]
    out = gold_inferences_from_briefs(briefs)
    assert set(out) == {"ex1", "ex2", "ex3"}
    assert out["ex2"].assistant_text == "Ad copy 2."
    assert all(inf.config == GOLD_CONFIG for inf in out.values())


def test_gold_inference_serializes_roundtrip() -> None:
    """GOLD-synthesized Inferences must serialize like any other Inference
    (the judge driver may save them via the same code path)."""
    from draper.evaluation.schemas import Inference

    b = _brief("ex1", "Glow All Day.")
    inf = gold_inference_from_brief(b)
    payload = inf.model_dump_json()
    parsed = Inference.model_validate_json(payload)
    assert parsed.config == GOLD_CONFIG
    assert parsed.assistant_text == "Glow All Day."


def test_gold_inference_custom_config_name() -> None:
    """config_name param propagates to Inference.config and Inference.model_id."""
    b = _brief("ex1", "Real ad copy.")
    inf = gold_inference_from_brief(b, "GOLD_v2")
    assert inf.config == "GOLD_v2"
    assert inf.model_id == "GOLD_v2"
    assert inf.assistant_text == "Real ad copy."
    assert inf.example_id == "ex1"


def test_gold_inference_default_config_name_is_gold() -> None:
    """Calling gold_inference_from_brief without config_name defaults to GOLD."""
    b = _brief("ex2", "Default gold.")
    inf = gold_inference_from_brief(b)
    assert inf.config == GOLD_CONFIG
    assert inf.model_id == GOLD_CONFIG


def test_gold_inferences_from_briefs_custom_config_name() -> None:
    """config_name propagates to every Inference in the dict."""
    briefs = [_brief("ex1", "Copy 1."), _brief("ex2", "Copy 2.")]
    out = gold_inferences_from_briefs(briefs, "GOLD_v2")
    assert all(inf.config == "GOLD_v2" for inf in out.values())
    assert all(inf.model_id == "GOLD_v2" for inf in out.values())
    assert out["ex1"].assistant_text == "Copy 1."
    assert out["ex2"].assistant_text == "Copy 2."
