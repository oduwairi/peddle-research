"""Ad-hoc: quantify caption/artifact leakage in image-brief <think> blocks."""

from __future__ import annotations

import re
import sys

from draper.construction_v2.ingest.response_parser import ParsedResponse, parse_response
from draper.utils.io import read_jsonl

path = sys.argv[1] if len(sys.argv) > 1 else (
    "data/constructed_v2_image/runs/run100img/image_brief/responses_raw.jsonl"
)

CAPTION = re.compile(r"\bcaption", re.I)
# Broader: think referencing the existing creative as a seen artifact.
ARTIFACT = re.compile(
    r"\bcaption|the (?:real |existing |source |original )?creative\b|"
    r"the (?:real|original) ad\b|the image (?:shows|depicts|features)|"
    r"as (?:shown|described|depicted|seen)\b|the visual (?:shows|depicts)",
    re.I,
)

rows = [r for r in read_jsonl(path) if isinstance(r, dict)]
by_model: dict[str, dict[str, int]] = {}
caption_hits: list[tuple[str, str]] = []

for r in rows:
    model = str(r.get("model") or "?")
    aid = str(r.get("ad_id") or "?")
    parsed = parse_response(str(r.get("content") or ""))
    if not isinstance(parsed, ParsedResponse):
        continue
    think = parsed.think
    d = by_model.setdefault(model, {"total": 0, "caption": 0, "artifact": 0})
    d["total"] += 1
    if CAPTION.search(think):
        d["caption"] += 1
        caption_hits.append((model, aid))
    if ARTIFACT.search(think):
        d["artifact"] += 1

print("=== caption / artifact leak in <think>, by model ===")
for m, d in sorted(by_model.items()):
    t = d["total"]
    print(
        f"{m:28s} n={t:3d}  caption={d['caption']:3d} ({100 * d['caption'] / t:.0f}%)"
        f"  artifact-leak={d['artifact']:3d} ({100 * d['artifact'] / t:.0f}%)"
    )

print("\n=== ads leaking the literal word 'caption' ===")
for m, aid in caption_hits:
    print(f"{m:28s} ad {aid}")
