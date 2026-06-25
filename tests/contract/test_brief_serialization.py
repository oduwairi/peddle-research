"""Contract test: Python canonical_json matches the stored byte target.

When Phase 4 lands, the TS-side ``serializeBriefForDraper`` will run
against this same fixture and must produce byte-identical output. For
now this test locks the Python side: a regression in
``canonical_json`` (e.g., losing ``sort_keys`` or eliding ``null``s)
breaks here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from draper.construction_v2.schemas.brief import Brief, canonical_json

FIXTURE_PATH = Path(__file__).parent / "brief_serialization.json"


def _load_fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("entry", _load_fixture()["briefs"])
def test_canonical_json_matches_fixture(entry: dict[str, object]) -> None:
    """canonical_json(brief) is byte-equal to the fixture target."""
    brief = Brief.model_validate(entry["brief"])
    expected = entry["canonical"]
    actual = canonical_json(brief)
    assert actual == expected, (
        f"canonical_json drift for {entry['name']!r}:\n"
        f"  expected: {expected!r}\n"
        f"  actual:   {actual!r}"
    )


def test_canonical_json_is_deterministic() -> None:
    """Running canonical_json twice on the same Brief yields identical bytes."""
    fixture = _load_fixture()
    for entry in fixture["briefs"]:
        brief = Brief.model_validate(entry["brief"])
        assert canonical_json(brief) == canonical_json(brief)


def test_canonical_json_preserves_null_fields() -> None:
    """Optional fields set to None must appear in the output as ``null``."""
    fixture = _load_fixture()
    minimal = next(e for e in fixture["briefs"] if e["name"] == "minimal")
    canonical = canonical_json(Brief.model_validate(minimal["brief"]))
    # product nullables
    assert '"category":null' in canonical
    assert '"offer":null' in canonical
    assert '"description":null' in canonical
    assert '"name":null' in canonical
    # bridge nullables (newly optional under the grounding contract)
    assert '"positioning":null' in canonical
    assert '"target_audience":null' in canonical


def test_canonical_json_keys_sorted() -> None:
    """Top-level and nested keys must appear in alphabetical order."""
    fixture = _load_fixture()
    for entry in fixture["briefs"]:
        canonical = canonical_json(Brief.model_validate(entry["brief"]))
        # Top-level: bridge, platform, product
        assert canonical.startswith('{"bridge":'), entry["name"]
        assert '"platform":' in canonical and '"product":' in canonical
        assert canonical.index('"bridge":') < canonical.index('"platform":')
        assert canonical.index('"platform":') < canonical.index('"product":')
