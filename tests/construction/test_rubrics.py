"""Tests for the copywriting rubric.

Copywriting's rubric is intentionally empty — backtranslation mode carries
its structural fidelity in the ingestion word-coverage / verbatim-signature
checks rather than required sections in the response.
"""

from __future__ import annotations

from draper.construction.formats.registry import get_pipeline
from draper.construction.schemas import TaskFormat


def _check(task_format: TaskFormat, response: str) -> list[str]:
    return get_pipeline(task_format).rubric_check(response)


class TestRubrics:
    def test_copywriting_pipeline_registered(self) -> None:
        get_pipeline(TaskFormat.COPYWRITING)  # raises KeyError if unregistered

    def test_copywriting_empty_rubric(self) -> None:
        from draper.construction.formats.copywriting import rubric as cw_rubric

        assert cw_rubric.REQUIRED_SECTIONS == []
        assert _check(TaskFormat.COPYWRITING, "anything goes here") == []
