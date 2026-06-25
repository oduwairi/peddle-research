"""Contract test: STATIC_SYSTEM_PROMPT matches the stored target.

The frontend's writer system prompt must be byte-identical to this
constant. Phase 4 will mirror this test on the TS side.
"""

from __future__ import annotations

import json
from pathlib import Path

from draper.construction_v2.schemas.brief import STATIC_SYSTEM_PROMPT

FIXTURE_PATH = Path(__file__).parent / "brief_serialization.json"


def test_static_system_prompt_matches_fixture() -> None:
    """The shipped prompt must equal the fixture target verbatim."""
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    expected = fixture["static_system_prompt"]
    assert expected == STATIC_SYSTEM_PROMPT, (
        "STATIC_SYSTEM_PROMPT drift:\n"
        f"  expected: {expected!r}\n"
        f"  actual:   {STATIC_SYSTEM_PROMPT!r}"
    )
