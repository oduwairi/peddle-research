"""Smoke test the LLM ad-copy extractor on three known-pathological cases.

Pulls the raw assistant_text from real inference files + GOLD content,
runs extract_ad_copy, and prints raw vs cleaned side-by-side so we can
visually confirm the extractor:

  - Removes Draper-r16's <think> + Hook/Structure/Word-choice rationale.
  - Reduces GOLD's pedagogical breakdown to just the ad copy.
  - Flags B's emoji-spam hallucination as <EXTRACTION_FAILED>.

Usage:
    python scripts/explore/smoke_normalize.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dotenv import load_dotenv

load_dotenv()

from draper.evaluation.briefs import load_test_briefs  # noqa: E402
from draper.evaluation.judge.normalize import extract_ad_copy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _show(label: str, raw: str, cleaned: str) -> None:
    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)
    print(f"\n--- RAW ({len(raw)} chars) ---")
    print(raw[:2000] + ("…[truncated]" if len(raw) > 2000 else ""))
    print(f"\n--- CLEANED ({len(cleaned)} chars) ---")
    print(cleaned)


async def main() -> None:
    cases: list[tuple[str, str, str, str]] = []

    # 1) Config C, draper-r16, <think> + Hook/Structure rationale
    p = ROOT / "data/eval/inferences/C/001953752d4d.json"
    data = json.loads(p.read_text())
    cases.append((
        "CASE 1: Config C (draper-r16) — Halo Collar",
        data["assistant_text"], "facebook", data.get("system", ""),
    ))

    # 2) Config B, qwen base, emoji-spam case
    p = ROOT / "data/eval/inferences/B/40fbf1346b57.json"
    data = json.loads(p.read_text())
    cases.append((
        "CASE 2: Config B (qwen base) — events ticketing (long-form, possible emoji spam)",
        data["assistant_text"], "facebook", data.get("system", ""),
    ))

    # 3) Config A, gpt-5.5, baseline well-formed
    p = ROOT / "data/eval/inferences/A/001953752d4d.json"
    data = json.loads(p.read_text())
    cases.append((
        "CASE 3: Config A (gpt-5.5) — Halo Collar",
        data["assistant_text"], "facebook", data.get("system", ""),
    ))

    # 4) GOLD pedagogical — short reddit ad with long breakdown
    briefs = load_test_briefs(ROOT / "data/final/test")
    by_id = {b.example_id: b for b in briefs}
    target_id = "b417cb44c1c2"
    if target_id not in by_id:
        print(f"WARN: brief {target_id} not in test split — skipping GOLD case")
    else:
        brief = by_id[target_id]
        cases.append((
            f"CASE 4: GOLD ({target_id}) — reddit, pedagogical breakdown",
            brief.reference_assistant, brief.platform, brief.system,
        ))

    # 5) GOLD Halo Collar (matching cases 1+3 for direct comparison)
    if "001953752d4d" in by_id:
        brief = by_id["001953752d4d"]
        cases.append((
            "CASE 5: GOLD (001953752d4d) — Halo Collar reference",
            brief.reference_assistant, brief.platform, brief.system,
        ))

    for label, raw, platform, _system in cases:
        cleaned = await extract_ad_copy(raw, platform=platform)
        _show(label, raw, cleaned)


if __name__ == "__main__":
    asyncio.run(main())
