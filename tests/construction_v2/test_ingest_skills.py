"""Tests for the skill registry that dispatches the v2 pipeline.

The bundle spans the submit/collect dispatch (``prepare_source_ads`` /
``build_request`` / ``parse_response``) plus the ingest hooks
(``build_brief`` + the ``fidelity`` / ``grounding`` gates and the
optional ``leak`` / ``labels`` gates).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from draper.construction.batch.types import BatchRequest
from draper.construction_v2.ingest.fidelity import FidelityResult, GroundingResult
from draper.construction_v2.ingest.skills import (
    SkillGateBundle,
    get_bundle,
    register,
    registered_skills,
)
from draper.construction_v2.platform_labels import LabelResult


def _ok_fidelity(_d: str, _ad: object) -> FidelityResult:
    return FidelityResult(
        passed=True,
        coverage=1.0,
        ad_word_count=1,
        signature_passed=True,
        reason="ok",
    )


def _ok_grounding(_t: str, _b: object) -> GroundingResult:
    return GroundingResult(passed=True, bridge_match="angle", reason="ok")


def _ok_labels(_d: str, _ad: object) -> LabelResult:
    return LabelResult(passed=True, expected=(), missing=(), reason="ok")


def _noop_prepare(ads: list[Any], _config: object) -> tuple[list[Any], list[str]]:
    return ads, []


def _stub_build_brief(brief_dict: dict[str, Any], _ad: object) -> dict[str, Any]:
    return brief_dict


def _stub_build(ad: Any, *, model: str, max_tokens: int, temperature: float) -> BatchRequest:
    return BatchRequest(
        custom_id=f"teacher-{getattr(ad, 'ad_id', 'x')}",
        system=None,
        messages=[{"role": "user", "content": "stub"}],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


@dataclass(frozen=True)
class _StubParse:
    brief: dict[str, Any] | None = None
    think: str | None = None
    deliverable: str | None = None
    errors: list[str] = field(default_factory=list)


def _stub_parse(_content: str) -> _StubParse:
    return _StubParse()


class TestSkillRegistry:
    def test_copywriting_bundle_registered_on_import(self) -> None:
        bundle = get_bundle("copywriting")
        assert bundle.name == "copywriting"
        # Copywriting wires: submit hooks (prepare_source_ads, build_request,
        # parse_response) + ingest hooks (build_brief, fidelity, grounding,
        # labels) + optional hooks (leak=check_bridge_leak, content_bridge=None).
        assert bundle.prepare_source_ads is not None
        assert bundle.build_request is not None
        assert bundle.parse_response is not None
        assert bundle.fidelity is not None
        assert bundle.grounding is not None
        assert bundle.labels is not None
        assert bundle.system_prompt
        assert "Draper" in bundle.system_prompt
        assert callable(bundle.build_brief)
        assert bundle.leak is not None  # copywriting uses check_bridge_leak
        assert bundle.content_bridge is None  # no content-bridge gate for copywriting

    def test_image_brief_bundle_registered_on_import(self) -> None:
        bundle = get_bundle("image_brief")
        assert bundle.name == "image_brief"
        assert bundle.prepare_source_ads is not None
        assert bundle.build_request is not None
        assert bundle.parse_response is not None
        assert bundle.fidelity is not None
        assert bundle.grounding is not None
        # Image-brief has no platform-native field labels.
        assert bundle.labels is None
        assert bundle.system_prompt
        assert "image brief" in bundle.system_prompt.lower()
        assert callable(bundle.build_brief)
        assert bundle.leak is None  # copy is legitimate verbatim input — no leak guard
        assert bundle.content_bridge is not None  # factual content-bridge gate is wired

    def test_copywriting_prepare_is_identity(self) -> None:
        """Copywriting submit-time prepare must pass ads through unchanged."""
        bundle = get_bundle("copywriting")

        class _FakeAd:
            ad_id = "a1"

        ads = [_FakeAd(), _FakeAd()]
        kept, missing = bundle.prepare_source_ads(ads, object())  # type: ignore[arg-type]
        assert kept == ads
        assert missing == []

    def test_unknown_skill_raises_with_available_list(self) -> None:
        with pytest.raises(KeyError, match="No skill bundle registered"):
            get_bundle("does_not_exist")

    def test_register_replaces_by_name(self) -> None:
        skill = "test_skill_replace_me"
        register(
            SkillGateBundle(
                name=skill,
                prepare_source_ads=_noop_prepare,
                build_request=_stub_build,
                parse_response=_stub_parse,
                build_brief=_stub_build_brief,
                fidelity=_ok_fidelity,
                grounding=_ok_grounding,
                leak=None,
                labels=_ok_labels,
                content_bridge=None,
                system_prompt="stub",
            )
        )
        register(
            SkillGateBundle(
                name=skill,
                prepare_source_ads=_noop_prepare,
                build_request=_stub_build,
                parse_response=_stub_parse,
                build_brief=_stub_build_brief,
                fidelity=_ok_fidelity,
                grounding=_ok_grounding,
                leak=None,
                labels=None,
                content_bridge=None,
                system_prompt="stub",
            )
        )
        assert get_bundle(skill).labels is None

    def test_labels_optional_skill_has_none(self) -> None:
        """Skills whose deliverable has no platform-native field labels
        leave ``labels`` as ``None`` and the ingest pipeline skips that stage."""
        skill = "test_no_labels"
        register(
            SkillGateBundle(
                name=skill,
                prepare_source_ads=_noop_prepare,
                build_request=_stub_build,
                parse_response=_stub_parse,
                build_brief=_stub_build_brief,
                fidelity=_ok_fidelity,
                grounding=_ok_grounding,
                leak=None,
                labels=None,
                content_bridge=None,
                system_prompt="stub",
            )
        )
        assert get_bundle(skill).labels is None

    def test_registered_skills_lists_known(self) -> None:
        skills = registered_skills()
        assert "copywriting" in skills
        assert "image_brief" in skills
