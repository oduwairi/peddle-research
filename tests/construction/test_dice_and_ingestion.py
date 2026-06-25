"""Tests for the shared dice-rolling + ingestion helpers.

Both helpers sit under the chat-client and API-batch CLI paths. Keeping
them covered independently of the CLI means either path can change
without drifting from the other.
"""

from __future__ import annotations

from pathlib import Path

from draper.construction.bundle import (
    ASSISTANT_RESPONSE_CLOSE,
    ASSISTANT_RESPONSE_OPEN,
    USER_PROMPT_CLOSE,
    USER_PROMPT_OPEN,
)
from draper.construction.dice import prepare_bundles
from draper.construction.formats.copywriting.constructor import CopywritingConstructor
from draper.construction.ingestion import ingest_response, load_ads_by_id
from draper.construction.personas import PersonaLibrary
from draper.construction.schemas import (
    ConstructionConfig,
    FormatConfig,
    PromptStyle,
    TaskFormat,
)
from draper.scoring.schemas import ScoredAd
from draper.scraping.schemas import AdCopy, AdSource, Platform, RawAd


def _make_scored_ad(
    ad_id: str,
    tier: str = "high",
    score: float = 0.85,
    platform: Platform = Platform.FACEBOOK,
    vertical: str = "dtc_skincare",
    headline: str = "Transform your skincare routine in 30 days",
    body: str = (
        "Our clinically-tested serum delivers visible results. Real customers "
        "report smoother texture, fewer breakouts, and radiant skin after "
        "four weeks of daily use."
    ),
    cta: str = "Shop the collection today",
) -> ScoredAd:
    ad = RawAd(
        ad_id=ad_id,
        source=AdSource.META_LIBRARY,
        platform=platform,
        ad_copy=AdCopy(headline=headline, body=body, cta=cta),
        active_days=14,
        vertical=vertical,
        advertiser_name=f"Brand-{ad_id}",
    )
    return ScoredAd(
        ad=ad,
        composite_score=score,
        signal_scores={"longevity": 0.8, "early_death": 1.0},
        tier=tier,
    )


def _make_cfg(tmp_path: Path) -> ConstructionConfig:
    return ConstructionConfig(
        scored_ads_path=str(tmp_path / "scored.jsonl"),
        output_dir=str(tmp_path / "constructed"),
        clusters_dir=str(tmp_path / "clusters"),
        final_dir=str(tmp_path / "final"),
        formats={
            "copywriting": FormatConfig(
                target=10,
                valid_styles=[PromptStyle.BACKTRANSLATION],
                style_ratios={PromptStyle.BACKTRANSLATION.value: 1.0},
            ),
        },
    )


def _tagged(user_prompt: str, assistant_response: str) -> str:
    return (
        f"{USER_PROMPT_OPEN}\n{user_prompt}\n{USER_PROMPT_CLOSE}\n"
        f"{ASSISTANT_RESPONSE_OPEN}\n{assistant_response}\n"
        f"{ASSISTANT_RESPONSE_CLOSE}"
    )


class TestPrepareBundles:
    def test_copywriting_bundle_has_expected_sidecar(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        constructor = CopywritingConstructor(config=cfg)
        personas = PersonaLibrary.from_yaml("configs/personas.yaml")

        batches = [[_make_scored_ad(f"cw{i}")] for i in range(3)]
        styles = [PromptStyle.BACKTRANSLATION] * 3

        prepared = prepare_bundles(
            cfg=cfg,
            constructor=constructor,
            personas=personas,
            batches=batches,
            styles=styles,
            provider_label="gpt",
        )
        assert len(prepared) == 3
        assert [p.context.style for p in prepared] == styles
        assert all(p.context.provider == "gpt" for p in prepared)
        for i, p in enumerate(prepared):
            sc = p.sidecar
            assert sc["prompt_index"] == i
            assert sc["provider"] == "gpt"
            assert sc["prompt_style"] == "backtranslation"
            assert "difficulty" in sc
            # Copywriting is sparse-disallowed.
            assert sc["difficulty"] != "sparse"

    def test_deterministic_given_generated_count(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        constructor = CopywritingConstructor(config=cfg)
        personas = PersonaLibrary.from_yaml("configs/personas.yaml")
        batches = [[_make_scored_ad(f"cw{i}")] for i in range(2)]
        styles = [PromptStyle.BACKTRANSLATION, PromptStyle.BACKTRANSLATION]

        run_a = prepare_bundles(
            cfg=cfg,
            constructor=constructor,
            personas=personas,
            batches=batches,
            styles=styles,
            provider_label="gpt",
        )
        run_b = prepare_bundles(
            cfg=cfg,
            constructor=constructor,
            personas=personas,
            batches=batches,
            styles=styles,
            provider_label="gpt",
        )
        for a, b in zip(run_a, run_b, strict=False):
            assert a.sidecar["difficulty"] == b.sidecar["difficulty"]


class TestIngestResponse:
    def test_missing_tags_reports_error(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        constructor = CopywritingConstructor(config=cfg)
        result = ingest_response(
            response_text="just prose, no tags",
            sidecar={
                "source_ad_ids": ["ad0"],
                "prompt_style": "backtranslation",
            },
            constructor=constructor,
            ads_by_id={"ad0": _make_scored_ad("ad0")},
        )
        assert result.saved == 0
        assert "Missing required tags" in result.error

    def test_backtranslation_fidelity_accepts_matching_response(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_cfg(tmp_path)
        constructor = CopywritingConstructor(config=cfg)
        ad = _make_scored_ad("ad0")
        sidecar = {
            "prompt_index": 0,
            "source_ad_ids": ["ad0"],
            "prompt_style": "backtranslation",
            "persona_id": "smb_founder_first_ads",
            "difficulty": "standard",
            "turn_structure": "single",
            "followup_type": "",
            "provider": "gpt",
        }
        # Response must echo the source ad's headline/body/CTA verbatim for
        # the backtranslation fidelity check to pass.
        assistant_response = (
            f"{ad.ad.ad_copy.headline}\n\n{ad.ad.ad_copy.body}\n\n"
            f"{ad.ad.ad_copy.cta}"
        )
        response = _tagged(
            user_prompt="Write me ad copy for a 30-day skincare serum.",
            assistant_response=assistant_response,
        )
        result = ingest_response(
            response_text=response,
            sidecar=sidecar,
            constructor=constructor,
            ads_by_id={"ad0": ad},
            construction_model_override="gpt-4o-mini",
        )
        assert result.saved == 1, result.error
        examples = constructor.load_existing_examples()
        assert len(examples) == 1
        ex = examples[0]
        assert ex.task_format == TaskFormat.COPYWRITING
        assert ex.metadata.construction_model == "gpt-4o-mini"
        assert ex.metadata.prompt_style == PromptStyle.BACKTRANSLATION
        assert [m.role for m in ex.messages] == ["system", "user", "assistant"]


class TestLoadAdsById:
    def test_returns_dict_keyed_by_ad_id(self, tmp_path: Path) -> None:
        path = tmp_path / "scored.jsonl"
        ad = _make_scored_ad("ad0")
        with path.open("w") as f:
            f.write(ad.model_dump_json() + "\n")
        loaded = load_ads_by_id(str(path))
        assert "ad0" in loaded
        assert loaded["ad0"].ad.ad_id == "ad0"
