"""Tests for the quality filter."""

from __future__ import annotations

from draper.construction.quality_filter import (
    QualityFilter,
    _get_assistant_content,
    _is_english,
    _is_structurally_valid,
)
from draper.construction.schemas import (
    ChatMessage,
    QualityFilterConfig,
    TaskFormat,
    TrainingExample,
)

_DEFAULT_CONTENT = (
    "## Category Read\nThe target audience in this category navigates social "
    "tensions on Facebook and TikTok. This is the core audience frame.\n"
    "## Angles Observed\nBenefit-led angle/frame with a strong hook. The "
    "positioning lands via concrete outcomes rather than abstract promise.\n"
    "Platform: Facebook carousel, TikTok video.\n"
    "## Heuristic\nPick the benefit-led angle first when entering this "
    "category — it consistently leads the strongest lift across platforms."
)


def _make_example(
    assistant_content: str = _DEFAULT_CONTENT,
    include_system: bool = True,
    include_user: bool = True,
    include_assistant: bool = True,
    example_id: str = "",
) -> TrainingExample:
    messages: list[ChatMessage] = []
    if include_system:
        messages.append(ChatMessage(role="system", content="You are helpful."))
    if include_user:
        messages.append(ChatMessage(role="user", content="Do something."))
    if include_assistant:
        messages.append(ChatMessage(role="assistant", content=assistant_content))
    kwargs: dict[str, object] = {
        "task_format": TaskFormat.COPYWRITING,
        "messages": messages,
    }
    if example_id:
        kwargs["example_id"] = example_id
    return TrainingExample(**kwargs)  # type: ignore[arg-type]


class TestHelpers:
    def test_get_assistant_content(self) -> None:
        ex = _make_example(assistant_content="Hello world")
        assert _get_assistant_content(ex) == "Hello world"

    def test_get_assistant_content_missing(self) -> None:
        ex = _make_example(include_assistant=False)
        assert _get_assistant_content(ex) == ""

    def test_structural_valid(self) -> None:
        ex = _make_example()
        assert _is_structurally_valid(ex) is True

    def test_structural_no_user(self) -> None:
        ex = _make_example(include_user=False)
        assert _is_structurally_valid(ex) is False

    def test_structural_no_assistant(self) -> None:
        ex = _make_example(include_assistant=False)
        assert _is_structurally_valid(ex) is False

    def test_is_english(self) -> None:
        assert _is_english("This is a test sentence in English.") is True

    def test_is_not_english(self) -> None:
        assert _is_english("Dies ist ein Testsatz auf Deutsch.") is False


class TestQualityFilter:
    def test_all_pass(self) -> None:
        # Each text includes all required positioning rubric keywords:
        # category/audience, angle/frame, platform, and heuristic/first.
        distinct_texts = [
            (
                "## Category Read\nFitness category audience on Facebook.\n"
                "## Angles\nAuthority via specificity is a strong angle/frame.\n"
                "Platform: Facebook reads well.\n"
                "## Heuristic\nPick the authority angle first for a new entrant."
            ),
            (
                "## Category Read\nSkincare audience on TikTok seeking proof.\n"
                "## Angles\nSocial-proof density as a landing frame/angle.\n"
                "Platform: TikTok short-form video.\n"
                "## Heuristic\nPick the proof angle to lead first with new audiences."
            ),
            (
                "## Category Read\nHome decor audience on Pinterest.\n"
                "## Angles\nAspirational lifestyle as a dominant angle/frame.\n"
                "Platform: Pinterest reads lifestyle carousel well.\n"
                "## Heuristic\nLead with aspirational angle first for discovery."
            ),
            (
                "## Category Read\nNiche hobbyist audience on Reddit.\n"
                "## Angles\nAuthentic community-led angle/frame wins here.\n"
                "Platform: Reddit rewards text-heavy authenticity.\n"
                "## Heuristic\nPick the community angle first; ads that look "
                "like ads get downvoted fast."
            ),
            (
                "## Category Read\nB2B technology decision-maker audience.\n"
                "## Angles\nThought-leadership framing is the primary angle.\n"
                "Platform: LinkedIn threads and long-form posts.\n"
                "## Heuristic\nLead with authority angle first for enterprise."
            ),
        ]
        examples = [_make_example(assistant_content=t) for t in distinct_texts]
        qf = QualityFilter()
        result = qf.filter_all(examples)
        assert result.stats.passed == 5
        assert result.stats.total_input == 5

    def test_rejects_structural(self) -> None:
        good = _make_example()
        bad = _make_example(include_user=False)
        qf = QualityFilter()
        result = qf.filter_all([good, bad])
        assert result.stats.passed == 1
        assert result.stats.rejected_structural == 1

    def test_rejects_short(self) -> None:
        short = _make_example(assistant_content="Too short")
        qf = QualityFilter(config=QualityFilterConfig(min_response_length=200))
        result = qf.filter_all([short])
        assert result.stats.rejected_min_length == 1
        assert result.stats.passed == 0

    def test_dedup_removes_near_duplicates(self) -> None:
        # All three include every required positioning rubric keyword so
        # only the dedup filter is responsible for rejections here.
        content = (
            "Category audience, angle frame, platform Facebook, heuristic pick first "
            "positioning strategy for millennial shoppers. " * 10
        )
        alt_content = (
            "Different category audience, alternative angle frame, platform Pinterest, "
            "heuristic pick the lifestyle angle first for home decor buyers. " * 10
        )
        ex1 = _make_example(assistant_content=content)
        ex2 = _make_example(assistant_content=content)  # exact duplicate
        ex3 = _make_example(assistant_content=alt_content)
        qf = QualityFilter(config=QualityFilterConfig(dedup_threshold=0.85))
        result = qf.filter_all([ex1, ex2, ex3])
        assert result.stats.rejected_duplicate == 1
        assert result.stats.passed == 2

    def test_empty_input(self) -> None:
        qf = QualityFilter()
        result = qf.filter_all([])
        assert result.stats.passed == 0
        assert result.stats.total_input == 0

    def test_cross_format_source_dedup(self) -> None:
        from draper.construction.schemas import ExampleMetadata

        # Each example needs distinct assistant content so response-dedup
        # doesn't swallow them before source-ad dedup runs.
        content_a = (
            "## Category Read\nMillennials on Facebook.\n"
            "## Angles\nBenefit-led angle/frame.\n"
            "Platform: Facebook.\n"
            "## Heuristic\nPick benefit angle first. " * 3
        )
        content_b = (
            "## Category Read\nGen Z on TikTok.\n"
            "## Angles\nAuthentic angle/frame.\n"
            "Platform: TikTok.\n"
            "## Heuristic\nPick authenticity first. " * 3
        )
        content_c = (
            "## Category Read\nProfessionals on LinkedIn.\n"
            "## Angles\nCredibility angle/frame.\n"
            "Platform: LinkedIn.\n"
            "## Heuristic\nPick credibility first. " * 3
        )
        ex1 = _make_example(example_id="first", assistant_content=content_a)
        ex1.metadata = ExampleMetadata(source_ad_ids=["a1", "a2", "a3"])
        ex2 = _make_example(example_id="dupe", assistant_content=content_b)
        ex2.metadata = ExampleMetadata(source_ad_ids=["a1", "a2", "a3"])
        ex3 = _make_example(example_id="distinct", assistant_content=content_c)
        ex3.metadata = ExampleMetadata(source_ad_ids=["a4", "a5"])
        qf = QualityFilter()
        result = qf.filter_all([ex1, ex2, ex3])
        # ex2 shares ad set with ex1 → rejected by cross-format source dedup.
        assert result.stats.rejected_source_ad_duplicate == 1
        assert result.stats.passed == 2

    def test_source_ad_dedup_skipped_when_disabled(self) -> None:
        from draper.construction.schemas import ExampleMetadata

        content_a = (
            "## Category Read\nMillennials on Facebook.\n"
            "## Angles\nBenefit-led angle/frame.\n"
            "Platform: Facebook.\n"
            "## Heuristic\nPick benefit angle first. " * 3
        )
        content_b = (
            "## Category Read\nGen Z on TikTok.\n"
            "## Angles\nAuthentic angle/frame.\n"
            "Platform: TikTok.\n"
            "## Heuristic\nPick authenticity first. " * 3
        )
        ex1 = _make_example(example_id="a", assistant_content=content_a)
        ex1.metadata = ExampleMetadata(source_ad_ids=["x"])
        ex2 = _make_example(example_id="b", assistant_content=content_b)
        ex2.metadata = ExampleMetadata(source_ad_ids=["x"])
        qf = QualityFilter(config=QualityFilterConfig(cross_format_source_dedup=False))
        result = qf.filter_all([ex1, ex2])
        assert result.stats.rejected_source_ad_duplicate == 0


class TestStyleBSpecificityFilter:
    """Coverage for the Style-B specificity guard (stage 6).

    Only CONTEXT_DISTILLED examples should be rejected for containing
    percentages, dollar amounts, or engagement counts. DATA_GROUNDED
    and NATURAL responses may legitimately cite these.
    """

    _GOOD_STYLE_B = (
        "## Category Read\nFitness-oriented audience responds to benefit-led "
        "framing on feed-based platforms.\n"
        "## Angles\nBenefit-led angle/frame with a hook up front.\n"
        "Platform: feed-based social — TikTok and Facebook.\n"
        "## Heuristic\nPick the benefit-led angle first for this audience; "
        "casual tonality wins where long-form product detail falls flat."
    )

    def test_rejects_percentage_in_style_b(self) -> None:
        from draper.construction.schemas import ExampleMetadata, PromptStyle

        bad = (
            "## Category Read\nFitness-oriented audience on feed-based platforms.\n"
            "## Angles\nBenefit-led angle/frame with a 32% lift observed in the hook.\n"
            "Platform: Facebook and Instagram.\n"
            "## Heuristic\nPick the benefit-led angle first."
        )
        ex = _make_example(example_id="pct", assistant_content=bad)
        ex.metadata = ExampleMetadata(prompt_style=PromptStyle.CONTEXT_DISTILLED)
        qf = QualityFilter()
        result = qf.filter_all([ex])
        assert result.stats.rejected_style_b_specificity == 1

    def test_rejects_dollar_amount_in_style_b(self) -> None:
        from draper.construction.schemas import ExampleMetadata, PromptStyle

        bad = (
            "## Category Read\nSMB audience exploring paid social across "
            "Facebook and LinkedIn — a category where buyers are still "
            "building confidence in paid channels and need to see clear "
            "operational proof before committing spend.\n"
            "## Angles\nOutcome-led angle/frame with a $4.50 CPM hook "
            "referenced from category benchmarks. Authority-via-specificity "
            "also lands because SMBs want to see concrete numbers before "
            "trusting a channel at all.\n"
            "Platform: Facebook and LinkedIn tend to over-index for this "
            "audience because feed-based discovery matches how they browse.\n"
            "## Heuristic\nPick the outcome-led angle first — SMBs respond "
            "to operational detail over aspirational framing."
        )
        ex = _make_example(example_id="dollar", assistant_content=bad)
        ex.metadata = ExampleMetadata(prompt_style=PromptStyle.CONTEXT_DISTILLED)
        qf = QualityFilter()
        result = qf.filter_all([ex])
        assert result.stats.rejected_style_b_specificity == 1

    def test_rejects_engagement_count_in_style_b(self) -> None:
        from draper.construction.schemas import ExampleMetadata, PromptStyle

        bad = (
            "## Category Read\nSkincare audience looking for before/after proof.\n"
            "## Angles\nSocial-proof angle/frame with a testimonial hook — "
            "one benchmark post racked up 50,000 likes on Instagram.\n"
            "Platform: Meta feed and TikTok.\n"
            "## Heuristic\nPick the social-proof angle first."
        )
        ex = _make_example(example_id="eng", assistant_content=bad)
        ex.metadata = ExampleMetadata(prompt_style=PromptStyle.CONTEXT_DISTILLED)
        qf = QualityFilter()
        result = qf.filter_all([ex])
        assert result.stats.rejected_style_b_specificity == 1

    def test_passes_clean_style_b(self) -> None:
        from draper.construction.schemas import ExampleMetadata, PromptStyle

        ex = _make_example(
            example_id="clean_b",
            assistant_content=self._GOOD_STYLE_B,
        )
        ex.metadata = ExampleMetadata(prompt_style=PromptStyle.CONTEXT_DISTILLED)
        qf = QualityFilter()
        result = qf.filter_all([ex])
        assert result.stats.rejected_style_b_specificity == 0

    def test_allows_percentage_in_data_grounded(self) -> None:
        """Style A (data-grounded) may cite percentages from provided ads."""
        from draper.construction.schemas import ExampleMetadata, PromptStyle

        content = (
            "## Category Read\nFitness enthusiasts on Meta platforms.\n"
            "## Angles\nBenefit-led angle/frame citing a 32% lift hook.\n"
            "Platform: Facebook and Instagram.\n"
            "## Heuristic\nPick the benefit-led angle first."
        )
        ex = _make_example(example_id="a_pct", assistant_content=content)
        ex.metadata = ExampleMetadata(prompt_style=PromptStyle.DATA_GROUNDED)
        qf = QualityFilter()
        result = qf.filter_all([ex])
        assert result.stats.rejected_style_b_specificity == 0

    def test_allows_percentage_in_natural(self) -> None:
        """Style C (natural) draws on parametric knowledge — no source to gate."""
        from draper.construction.schemas import ExampleMetadata, PromptStyle

        content = (
            "## Category Read\nDTC buyers comfortable with short-form video.\n"
            "## Angles\nUrgency-led angle/frame. Industry lifts hover ~20% "
            "around the hook.\n"
            "Platform: TikTok and Reels.\n"
            "## Heuristic\nPick the urgency angle first for conversion."
        )
        ex = _make_example(example_id="c_pct", assistant_content=content)
        ex.metadata = ExampleMetadata(prompt_style=PromptStyle.NATURAL)
        qf = QualityFilter()
        result = qf.filter_all([ex])
        assert result.stats.rejected_style_b_specificity == 0
