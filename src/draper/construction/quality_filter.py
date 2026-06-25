"""Post-generation quality filtering and deduplication.

Run after examples are generated (via either API or chat-client mode) to
remove structural issues, non-English content, near-duplicates, and
low-quality outputs.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langdetect import LangDetectException, detect
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from draper.construction.formats.registry import get_pipeline
from draper.construction.schemas import (
    PromptStyle,
    QualityFilterConfig,
    TrainingExample,
)

logger = logging.getLogger("draper")


class FilterStats(BaseModel):
    """Aggregate statistics from a quality-filter run."""

    total_input: int = 0
    passed: int = 0
    rejected_structural: int = 0
    rejected_min_length: int = 0
    rejected_language: int = 0
    rejected_duplicate: int = 0
    rejected_prompt_duplicate: int = 0
    rejected_source_ad_duplicate: int = 0
    rejected_rubric: int = 0
    rejected_style_b_specificity: int = 0
    rejected_format_specific: int = 0
    rejected_artifact_leak: int = 0
    artifacts_repaired: int = 0
    rejected_quality: int = 0
    quality_sample_size: int = 0
    quality_mean_score: float = 0.0


class FilterResult(BaseModel):
    """Output of ``QualityFilter.filter_all``."""

    passed: list[TrainingExample] = Field(default_factory=list)
    rejected: list[tuple[str, str]] = Field(default_factory=list)  # (example_id, reason)
    stats: FilterStats = Field(default_factory=FilterStats)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_assistant_content(example: TrainingExample) -> str:
    """Extract the assistant response text from an example."""
    for msg in example.messages:
        if msg.role == "assistant":
            return msg.content
    return ""


def _get_user_content(example: TrainingExample) -> str:
    """Extract the first user-prompt text from an example."""
    for msg in example.messages:
        if msg.role == "user":
            return msg.content
    return ""


def _is_structurally_valid(example: TrainingExample) -> bool:
    """Check basic structural requirements."""
    if not example.messages:
        return False
    roles = [m.role for m in example.messages]
    if "user" not in roles or "assistant" not in roles:
        return False
    # Assistant must come after user
    user_idx = roles.index("user")
    assistant_idx = roles.index("assistant")
    return assistant_idx > user_idx


def _is_english(text: str) -> bool:
    """Detect whether text is primarily English."""
    try:
        return bool(detect(text) == "en")
    except LangDetectException:
        return False


def _tfidf_dedup(
    examples: list[TrainingExample],
    text_fn: Any,  # Callable[[TrainingExample], str] — Any avoids import
    threshold: float,
    rejection_reason: str,
) -> tuple[list[TrainingExample], list[tuple[str, str]]]:
    """Mark later near-duplicates via TF-IDF cosine similarity."""
    if len(examples) < 2:  # noqa: PLR2004
        return examples, []

    texts = [text_fn(ex) for ex in examples]
    # Skip any example with empty text for the chosen field.
    nonempty_indices = [i for i, t in enumerate(texts) if t.strip()]
    if len(nonempty_indices) < 2:  # noqa: PLR2004
        return examples, []

    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform([texts[i] for i in nonempty_indices])
    except ValueError:
        # All documents were stop-words only — nothing to dedup on.
        return examples, []
    sim_matrix: Any = cosine_similarity(tfidf_matrix)

    # Flag duplicates within the nonempty subset; map back to full indices.
    is_dup = [False] * len(nonempty_indices)
    for a in range(len(nonempty_indices)):
        if is_dup[a]:
            continue
        for b in range(a + 1, len(nonempty_indices)):
            if is_dup[b]:
                continue
            if sim_matrix[a, b] > threshold:
                is_dup[b] = True

    flagged_full: set[int] = {nonempty_indices[a] for a, flag in enumerate(is_dup) if flag}
    passed: list[TrainingExample] = []
    rejected: list[tuple[str, str]] = []
    for i, ex in enumerate(examples):
        if i in flagged_full:
            rejected.append((ex.example_id, rejection_reason))
        else:
            passed.append(ex)
    return passed, rejected


# Style-B specificity guard. These patterns catch the most common
# hallucination vectors for context-distilled responses: percentages
# ("32%", "2x lift"), dollar amounts ("$4.50 CPM", "$10K"), and explicit
# engagement counts ("50,000 likes"). The bundle rules tell teachers not
# to include these in Style B; this filter is the enforcement backstop.
_PERCENT_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*%")
_DOLLAR_PATTERN = re.compile(r"\$\s*\d")
_ENGAGEMENT_COUNT_PATTERN = re.compile(
    r"\b\d{1,3}(?:,\d{3})+\s+(?:likes|shares|comments|views|impressions|clicks)\b",
    re.IGNORECASE,
)

# Teacher-artifact patterns. Reasoning-style teachers occasionally leak
# scaffolding tags into the rendered output: <thinking>...</thinking>
# blocks, <user_prompt>...</user_prompt> wrappers echoing the rendered
# prompt, etc. These are auto-repairable (extract inner content, strip
# the wrapper); anything that survives repair is rejected.
_THINKING_BLOCK = re.compile(r"<thinking[^>]*>.*?</thinking>", re.IGNORECASE | re.DOTALL)
_THINKING_ORPHAN = re.compile(r"</?thinking[^>]*>", re.IGNORECASE)
_USER_PROMPT_WRAPPER = re.compile(
    r"<user_prompt[^>]*>(.*?)</user_prompt>", re.IGNORECASE | re.DOTALL
)
_ASSISTANT_RESPONSE_WRAPPER = re.compile(
    r"<assistant_response[^>]*>(.*?)</assistant_response>", re.IGNORECASE | re.DOTALL
)
# Anything that looks like an XML/HTML tag and isn't a recognised prose
# tag. We treat residue under this pattern as a rejection signal *after*
# auto-repair has had a chance to clean known wrappers.
_PROSE_TAGS = "b|i|em|strong|br|p|a|ul|ol|li|h[1-6]"
_NONPROSE_TAG_RESIDUE = re.compile(
    rf"</?(?!(?:{_PROSE_TAGS})\b)[a-z_][a-z0-9_]*(?:\s[^>]*)?>",
    re.IGNORECASE,
)


def _repair_artifacts(text: str) -> str:
    """Strip known teacher scaffolding from a message body.

    Order matters: extract wrapper inner content first, then strip
    <thinking> blocks (which may live outside the wrapper).
    """
    if not text:
        return text
    m = _USER_PROMPT_WRAPPER.search(text)
    if m:
        text = m.group(1)
    m = _ASSISTANT_RESPONSE_WRAPPER.search(text)
    if m:
        text = m.group(1)
    text = _THINKING_BLOCK.sub("", text)
    text = _THINKING_ORPHAN.sub("", text)
    return text.strip()


def _has_artifact_residue(text: str) -> str:
    """Return the offending tag if any non-prose tag remains, else ''."""
    m = _NONPROSE_TAG_RESIDUE.search(text)
    return m.group(0) if m else ""

# Format-specific schema-leak and ad-centrality guards now live on the
# copywriting FormatPipeline; see
# ``draper.construction.formats.copywriting.quality_filter``. The shared
# QualityFilter invokes them via ``pipeline.extra_quality_filters``.


def _style_b_specificity_violations(text: str) -> list[str]:
    """Return labels for any Style-B-forbidden specificity markers in text.

    Empty list means the response is clean under the Style B rules.
    """
    violations: list[str] = []
    if _PERCENT_PATTERN.search(text):
        violations.append("percent")
    if _DOLLAR_PATTERN.search(text):
        violations.append("dollar")
    if _ENGAGEMENT_COUNT_PATTERN.search(text):
        violations.append("engagement_count")
    return violations


# ---------------------------------------------------------------------------
# Main filter
# ---------------------------------------------------------------------------


class QualityFilter:
    """Applies sequential quality filters to constructed training examples.

    Filters (in order):

    0. Teacher-artifact repair — strip <thinking> blocks and
       <user_prompt>/<assistant_response> wrappers; reject any example
       still containing a non-prose tag after repair
    1. Structural validation — correct message format
    2. Minimum length — assistant response >= configured threshold
    3. Language detection — English only
    4. Format rubric — per-format required sections present
    5. Style-B specificity guard — CONTEXT_DISTILLED responses may not
       contain percentages, dollar amounts, or engagement counts
       (Gekhman et al., EMNLP 2024 — defends against hallucination from
       fact injection in weights)
    6. Response dedup — TF-IDF cosine similarity > config.dedup_threshold
    7. Prompt dedup — TF-IDF cosine similarity > prompt_dedup_threshold
    8. Cross-format source-ad dedup — reject examples sharing source ads
    """

    def __init__(self, config: QualityFilterConfig | None = None) -> None:
        self._config = config or QualityFilterConfig()

    def filter_all(self, examples: list[TrainingExample]) -> FilterResult:
        """Run all filters and return results."""
        stats = FilterStats(total_input=len(examples))
        rejected: list[tuple[str, str]] = []
        current = list(examples)

        # 0. Teacher-artifact repair + residue rejection. Runs first so
        # downstream stages (length, language, dedup) operate on the
        # cleaned text rather than wrapper-padded duplicates.
        passed, fails, repaired = self._filter_artifact_leak(current)
        stats.rejected_artifact_leak = len(fails)
        stats.artifacts_repaired = repaired
        rejected.extend(fails)
        current = passed

        # 1. Structural validation
        passed, fails = self._filter_structural(current)
        stats.rejected_structural = len(fails)
        rejected.extend(fails)
        current = passed

        # 2. Minimum length
        passed, fails = self._filter_min_length(current)
        stats.rejected_min_length = len(fails)
        rejected.extend(fails)
        current = passed

        # 3. Language detection
        passed, fails = self._filter_language(current)
        stats.rejected_language = len(fails)
        rejected.extend(fails)
        current = passed

        # 4. Format rubric — per-format section coverage
        passed, fails = self._filter_rubric(current)
        stats.rejected_rubric = len(fails)
        rejected.extend(fails)
        current = passed

        # 4b. Per-format extra filters — each FormatPipeline contributes
        # its own checks. Copywriting adds schema-leak + ad-centrality
        # guards for backtranslation responses.
        passed, fails = self._filter_format_specific(current)
        stats.rejected_format_specific = len(fails)
        rejected.extend(fails)
        current = passed

        # 5. Style-B specificity guard — reject CONTEXT_DISTILLED responses
        # containing verifiable specifics (%, $, engagement counts) the
        # student can't derive from a natural prompt at inference.
        passed, fails = self._filter_style_b_specificity(current)
        stats.rejected_style_b_specificity = len(fails)
        rejected.extend(fails)
        current = passed

        # 6. Response-text dedup (TF-IDF cosine similarity)
        passed, fails = self._filter_duplicates(current)
        stats.rejected_duplicate = len(fails)
        rejected.extend(fails)
        current = passed

        # 7. User-prompt dedup (catches semantic prompt duplicates
        # even when responses diverge)
        passed, fails = self._filter_prompt_duplicates(current)
        stats.rejected_prompt_duplicate = len(fails)
        rejected.extend(fails)
        current = passed

        # 8. Cross-format source-ad dedup (reject two examples sharing
        # the exact same set of source ads across formats)
        if self._config.cross_format_source_dedup:
            passed, fails = self._filter_source_ad_duplicates(current)
            stats.rejected_source_ad_duplicate = len(fails)
            rejected.extend(fails)
            current = passed

        stats.passed = len(current)
        logger.info(
            "Quality filter: %d input → %d passed "
            "(artifact-repaired: +%d, artifact-leak: -%d, "
            "struct: -%d, length: -%d, lang: -%d, rubric: -%d, "
            "format-specific: -%d, style-b: -%d, dedup: -%d, "
            "prompt-dedup: -%d, src-dedup: -%d)",
            stats.total_input,
            stats.passed,
            stats.artifacts_repaired,
            stats.rejected_artifact_leak,
            stats.rejected_structural,
            stats.rejected_min_length,
            stats.rejected_language,
            stats.rejected_rubric,
            stats.rejected_format_specific,
            stats.rejected_style_b_specificity,
            stats.rejected_duplicate,
            stats.rejected_prompt_duplicate,
            stats.rejected_source_ad_duplicate,
        )
        return FilterResult(passed=current, rejected=rejected, stats=stats)

    # ------------------------------------------------------------------
    # Individual filters
    # ------------------------------------------------------------------

    def _filter_structural(
        self, examples: list[TrainingExample]
    ) -> tuple[list[TrainingExample], list[tuple[str, str]]]:
        passed: list[TrainingExample] = []
        rejected: list[tuple[str, str]] = []
        for ex in examples:
            if _is_structurally_valid(ex):
                passed.append(ex)
            else:
                rejected.append((ex.example_id, "structural"))
        return passed, rejected

    def _filter_artifact_leak(
        self, examples: list[TrainingExample]
    ) -> tuple[list[TrainingExample], list[tuple[str, str]], int]:
        """Repair teacher-scaffolding leaks in messages; reject residue.

        For each message in each example: extract any
        ``<user_prompt>``/``<assistant_response>`` wrapper inner content,
        then strip ``<thinking>`` blocks and orphan tags. If a non-prose
        tag still remains after repair, the example is rejected.
        """
        passed: list[TrainingExample] = []
        rejected: list[tuple[str, str]] = []
        repaired_count = 0
        for ex in examples:
            new_messages = []
            mutated = False
            residue = ""
            for msg in ex.messages:
                cleaned = _repair_artifacts(msg.content)
                if cleaned != msg.content:
                    mutated = True
                left = _has_artifact_residue(cleaned)
                if left and not residue:
                    residue = f"{msg.role}:{left}"
                if mutated and msg.content != cleaned:
                    new_messages.append(msg.model_copy(update={"content": cleaned}))
                else:
                    new_messages.append(msg)
            if residue:
                rejected.append((ex.example_id, f"artifact_leak:{residue}"))
                continue
            if mutated:
                ex = ex.model_copy(update={"messages": new_messages})
                repaired_count += 1
            passed.append(ex)
        return passed, rejected, repaired_count

    def _filter_min_length(
        self, examples: list[TrainingExample]
    ) -> tuple[list[TrainingExample], list[tuple[str, str]]]:
        """Enforce per-format minimum assistant-response length.

        The default floor comes from ``QualityFilterConfig``; each
        :class:`FormatPipeline` can lower it (copywriting drops to 80
        chars for naturally-tight backtranslation responses).
        """
        default_floor = self._config.min_response_length
        passed: list[TrainingExample] = []
        rejected: list[tuple[str, str]] = []
        for ex in examples:
            content = _get_assistant_content(ex)
            floor = get_pipeline(ex.task_format).min_length_floor(ex, default_floor)
            if len(content) >= floor:
                passed.append(ex)
            else:
                rejected.append((ex.example_id, "min_length"))
        return passed, rejected

    def _filter_language(
        self, examples: list[TrainingExample]
    ) -> tuple[list[TrainingExample], list[tuple[str, str]]]:
        passed: list[TrainingExample] = []
        rejected: list[tuple[str, str]] = []
        for ex in examples:
            content = _get_assistant_content(ex)
            # Only check if content is long enough for reliable detection
            if len(content) < 50 or _is_english(content):  # noqa: PLR2004
                passed.append(ex)
            else:
                rejected.append((ex.example_id, "language"))
        return passed, rejected

    def _filter_rubric(
        self, examples: list[TrainingExample]
    ) -> tuple[list[TrainingExample], list[tuple[str, str]]]:
        """Run per-format rubric check on each example's assistant response."""
        passed: list[TrainingExample] = []
        rejected: list[tuple[str, str]] = []
        for ex in examples:
            # Check the final assistant response (last assistant message
            # if multi-turn). That's where the full structure should appear.
            response_text = _get_assistant_content(ex)
            missing = get_pipeline(ex.task_format).rubric_check(response_text)
            if missing:
                rejected.append((ex.example_id, f"rubric:missing={','.join(missing)}"))
            else:
                passed.append(ex)
        return passed, rejected

    def _filter_format_specific(
        self,
        examples: list[TrainingExample],
    ) -> tuple[list[TrainingExample], list[tuple[str, str]]]:
        """Run each :class:`FormatPipeline`'s extra filters.

        Each format registers its own rejections via
        :meth:`FormatPipeline.extra_quality_filters`. Copywriting returns
        schema-leak + ad-centrality violations for backtranslation
        responses; other formats return nothing.
        """
        passed: list[TrainingExample] = []
        rejected: list[tuple[str, str]] = []
        for ex in examples:
            reasons = get_pipeline(ex.task_format).extra_quality_filters(ex)
            if reasons:
                rejected.extend(reasons)
            else:
                passed.append(ex)
        return passed, rejected

    def _filter_style_b_specificity(
        self, examples: list[TrainingExample]
    ) -> tuple[list[TrainingExample], list[tuple[str, str]]]:
        """Reject Style B responses that contain forbidden specific claims.

        Gated on style: only CONTEXT_DISTILLED examples are checked.
        Copywriting uses BACKTRANSLATION so this is effectively a no-op
        today; the filter stays in case other styles are reintroduced.
        """
        passed: list[TrainingExample] = []
        rejected: list[tuple[str, str]] = []
        for ex in examples:
            if ex.metadata.prompt_style != PromptStyle.CONTEXT_DISTILLED:
                passed.append(ex)
                continue
            response = _get_assistant_content(ex)
            violations = _style_b_specificity_violations(response)
            if violations:
                rejected.append(
                    (
                        ex.example_id,
                        f"style_b_specificity:{','.join(violations)}",
                    )
                )
            else:
                passed.append(ex)
        return passed, rejected

    def _filter_duplicates(
        self, examples: list[TrainingExample]
    ) -> tuple[list[TrainingExample], list[tuple[str, str]]]:
        """Remove response-text near-duplicates using TF-IDF cosine."""
        return _tfidf_dedup(
            examples,
            text_fn=_get_assistant_content,
            threshold=self._config.dedup_threshold,
            rejection_reason="duplicate",
        )

    def _filter_prompt_duplicates(
        self, examples: list[TrainingExample]
    ) -> tuple[list[TrainingExample], list[tuple[str, str]]]:
        """Remove user-prompt near-duplicates using TF-IDF cosine."""
        return _tfidf_dedup(
            examples,
            text_fn=_get_user_content,
            threshold=self._config.prompt_dedup_threshold,
            rejection_reason="prompt_duplicate",
        )

    def _filter_source_ad_duplicates(
        self, examples: list[TrainingExample]
    ) -> tuple[list[TrainingExample], list[tuple[str, str]]]:
        """Reject examples that share the exact same set of source ads.

        Works across formats: if two examples (potentially different formats)
        have the same ``source_ad_ids`` set, only the first is kept.
        Prevents over-reliance on a small subset of ads.
        """
        seen: dict[frozenset[str], str] = {}  # ad_id_set → first example_id
        passed: list[TrainingExample] = []
        rejected: list[tuple[str, str]] = []
        for ex in examples:
            key = frozenset(ex.metadata.source_ad_ids)
            if not key:
                # No source ads (e.g., pure natural style) — can't dedup.
                passed.append(ex)
                continue
            if key in seen:
                rejected.append((ex.example_id, f"source_ad_duplicate:first={seen[key]}"))
            else:
                seen[key] = ex.example_id
                passed.append(ex)
        return passed, rejected
