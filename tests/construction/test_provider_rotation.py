"""Tests for provider rotation logic."""

from __future__ import annotations

import json
from pathlib import Path

from draper.construction.batch.registry import BatchRegistry, PendingBatch
from draper.construction.provider_rotation import (
    classify_provider,
    format_provider_capacity,
    suggest_next_provider,
    tally_reserved_provider_counts,
)
from draper.construction.schemas import (
    ConstructionConfig,
    FormatConfig,
    ProviderRotationConfig,
    TaskFormat,
)


class TestClassify:
    def test_claude_variants(self) -> None:
        assert classify_provider("Claude Sonnet 4.6") == "claude"
        assert classify_provider("claude-haiku-4-5-20251001") == "claude"
        assert classify_provider("anthropic/claude") == "claude"

    def test_gpt_variants(self) -> None:
        assert classify_provider("GPT-4o") == "gpt"
        assert classify_provider("gpt-5") == "gpt"
        assert classify_provider("OpenAI o1-preview") == "gpt"

    def test_gemini_variants(self) -> None:
        assert classify_provider("Gemini 2.5 Pro") == "gemini"
        assert classify_provider("google-gemini") == "gemini"

    def test_empty_is_unknown(self) -> None:
        assert classify_provider("") == "unknown"

    def test_unrelated_is_unknown(self) -> None:
        assert classify_provider("mistral-7b-instruct") == "unknown"


class TestSuggest:
    def test_empty_history_returns_highest_target(self) -> None:
        rotation = ProviderRotationConfig(claude_ratio=0.40, gpt_ratio=0.35, gemini_ratio=0.25)
        counts = {"claude": 0, "gpt": 0, "gemini": 0, "unknown": 0}
        assert suggest_next_provider(counts, rotation) == "claude"

    def test_balances_toward_target(self) -> None:
        rotation = ProviderRotationConfig(claude_ratio=0.40, gpt_ratio=0.35, gemini_ratio=0.25)
        # Only claude has been used — suggest should move away from claude.
        counts = {"claude": 100, "gpt": 0, "gemini": 0, "unknown": 0}
        result = suggest_next_provider(counts, rotation)
        assert result in ("gpt", "gemini")

    def test_suggests_most_underrepresented(self) -> None:
        rotation = ProviderRotationConfig(claude_ratio=0.40, gpt_ratio=0.35, gemini_ratio=0.25)
        # claude: 60% (target 40%, +20pp over)
        # gpt: 40% (target 35%, +5pp over)
        # gemini: 0% (target 25%, -25pp under)
        counts = {"claude": 60, "gpt": 40, "gemini": 0, "unknown": 0}
        assert suggest_next_provider(counts, rotation) == "gemini"

    def test_ignores_unknown_in_denominator(self) -> None:
        rotation = ProviderRotationConfig()
        counts = {"claude": 10, "gpt": 10, "gemini": 10, "unknown": 1000}
        # With unknown excluded, each provider at ~33% share. Claude's
        # target (40%) is highest, so claude should be suggested.
        result = suggest_next_provider(counts, rotation)
        assert result == "claude"


class TestFormatProviderCapacity:
    def _write_examples(
        self,
        path: Path,
        per_provider: dict[str, int],
    ) -> None:
        """Seed examples.jsonl with N records tagged by construction_model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        model_for = {
            "claude": "claude-haiku-4-5-20251001",
            "gpt": "gpt-4o",
            "gemini": "gemini-2.5-pro",
        }
        with path.open("w") as f:
            for provider, n in per_provider.items():
                for _ in range(n):
                    f.write(
                        json.dumps(
                            {
                                "messages": [{"role": "user", "content": "x"}],
                                "metadata": {"construction_model": model_for[provider]},
                            }
                        )
                        + "\n"
                    )

    def test_capacity_reflects_used_and_reserved(self, tmp_path: Path) -> None:
        cfg = ConstructionConfig(
            output_dir=str(tmp_path),
            overgeneration_buffer=1.25,
            formats={"copywriting": FormatConfig(target=2500)},
        )
        # 100 used GPT examples already in the format file
        self._write_examples(
            tmp_path / "copywriting" / "examples.jsonl",
            {"claude": 0, "gpt": 100, "gemini": 0},
        )
        # 50 in-flight Gemini bundles in registry
        registry = BatchRegistry(tmp_path, "copywriting")
        registry.add(
            PendingBatch(
                batch_id="batches/abc",
                provider="gemini",
                model="gemini-2.5-pro",
                task_format="copywriting",
                submitted_at="2026-04-15T00:00:00Z",
                request_count=50,
            )
        )

        cap = format_provider_capacity(cfg, TaskFormat.COPYWRITING)
        # raw_target = 2500 * 1.25 = 3125
        assert cap["claude"].cap == int(3125 * 0.40)  # 1250
        assert cap["gpt"].cap == int(3125 * 0.35)  # 1093
        assert cap["gemini"].cap == int(3125 * 0.25)  # 781

        assert cap["gpt"].used == 100
        assert cap["gpt"].reserved == 0
        assert cap["gpt"].remaining == 1093 - 100

        assert cap["gemini"].used == 0
        assert cap["gemini"].reserved == 50
        assert cap["gemini"].remaining == 781 - 50

        assert cap["claude"].remaining == 1250

    def test_terminal_failed_batches_dont_reserve(self, tmp_path: Path) -> None:
        registry = BatchRegistry(tmp_path, "copywriting")
        registry.add(
            PendingBatch(
                batch_id="batches/dead",
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                task_format="copywriting",
                submitted_at="2026-04-15T00:00:00Z",
                request_count=200,
                status="failed",
            )
        )
        # Reload to pick up persisted status
        registry = BatchRegistry(tmp_path, "copywriting")
        reserved = tally_reserved_provider_counts(registry)
        assert reserved["claude"] == 0


class TestProviderRotationConfig:
    def test_targets_sum_validates(self) -> None:
        cfg = ProviderRotationConfig()
        cfg.validate_sum()  # should not raise

    def test_invalid_sum_raises(self) -> None:
        import pytest

        cfg = ProviderRotationConfig(claude_ratio=0.5, gpt_ratio=0.5, gemini_ratio=0.5)
        with pytest.raises(ValueError):
            cfg.validate_sum()
