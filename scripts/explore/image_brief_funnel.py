"""Funnel report: 55k v3-scored corpus → image-brief training data.

Mirrors the gates in ``source_selector.select_source_ads()`` so the
counts are exact (not approximate). The captioning step is a
**construction step** that runs *after* select on the chosen ad_ids —
not a sourcing step over the raw corpus.

The pipeline:

    scored_ads.jsonl
      └─ select (all quality + image_capable filters)
           └─ selection.parquet (eligible image-brief ads)
                └─ caption-creatives (over the selected set)
                     └─ submit / collect / ingest

Each drop is shown twice: as % of total (headline impact) and as % of
the pool *entering* that stage (real drop rate at that gate). The
second column is what tells you "where are ads being dropped" — a 1%
gate near the top of the chain wastes more ads than a 30% gate near
the bottom.

Run with: ``uv run python scripts/explore/image_brief_funnel.py``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from draper.construction.religious_scripture import is_religious_scripture_text
from draper.construction_v2.config import ConstructionV2Config
from draper.construction_v2.dataset.source_selector import (
    _has_structural_artifact,
    _row_to_source_ad,
)
from draper.construction_v2.platform_labels import (
    PlatformLabelGroup,
    platform_group_for,
    render_labeled_ad,
)

CONFIG_PATH = Path("configs/construction_v2_image.yaml")


def _iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _row(label: str, dropped: int, before: int, total: int) -> str:
    pct_total = 100.0 * dropped / total if total else 0.0
    pct_stage = 100.0 * dropped / before if before else 0.0
    return (
        f"    - {label:30s} drop {dropped:>6,}  "
        f"({pct_total:5.2f}% of total | {pct_stage:5.2f}% of stage)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--min-composite",
        type=float,
        default=None,
        help="Override min_composite from the config (e.g. 0.65 to widen the pool).",
    )
    ap.add_argument(
        "--min-training-quality",
        type=int,
        default=None,
        help="Override min_training_quality from the config (0 to disable).",
    )
    args = ap.parse_args()

    config = ConstructionV2Config.from_yaml(CONFIG_PATH)
    sel = config.selection
    if args.min_composite is not None:
        sel = sel.model_copy(update={"min_composite": args.min_composite})
    if args.min_training_quality is not None:
        sel = sel.model_copy(update={"min_training_quality": args.min_training_quality})
    scored_path = Path(sel.scored_ads_path)
    if not scored_path.exists():
        msg = f"Scored ads file not found: {scored_path}"
        raise FileNotFoundError(msg)

    unsafe_set = set(sel.unsafe_labels_to_drop)
    drop_verticals = set(sel.drop_verticals)

    # Walk every gate. For each, track:
    #   (gate_name, dropped_count, pool_size_entering_gate)
    # so we can report both "% of total" and "% of stage".
    gate_order: list[str] = []
    gate_drops: Counter[str] = Counter()
    gate_pool_entering: dict[str, int] = {}

    survivors: list[Any] = []  # rows that pass everything

    # Counters for the per-gate stratified-by-reason buckets
    # (content_safety:*, vertical:*, scripture:*, structural:*, training_quality:*).
    bucket_drops: dict[str, Counter[str]] = {
        "content_safety": Counter(),
        "vertical": Counter(),
        "scripture": Counter(),
        "structural": Counter(),
        "training_quality": Counter(),
        "image_capable_format": Counter(),
    }

    raw_image_capable = 0  # context only

    total = 0
    # Streaming pass for memory efficiency: accumulate decisions, never
    # store rows we drop.
    pool = 0
    rows_iter = _iter_rows(scored_path)

    def gate(name: str, dropped: bool, pool_size: int) -> None:
        if name not in gate_pool_entering:
            gate_order.append(name)
            gate_pool_entering[name] = pool_size
        if dropped:
            gate_drops[name] += 1

    # We need to know "pool entering gate" — easiest with two passes.
    # First pass: count drops per gate. Pool-entering = total - (sum of
    # earlier drops). We bookkeep that as we go.
    cumulative_dropped = 0

    # Track per-row decisions inline.
    for row in rows_iter:
        total += 1
        pool = total - cumulative_dropped

        ad = _row_to_source_ad(row)
        if ad is None:
            gate("malformed_row", True, pool)
            cumulative_dropped += 1
            continue
        gate("malformed_row", False, pool)
        raw_ad = row.get("ad")
        ad_dict: dict[str, Any] = raw_ad if isinstance(raw_ad, dict) else {}

        # Context-only: raw image-capable subset (image|carousel + url).
        cf_raw = ad_dict.get("creative_format", "")
        cu_raw = ad_dict.get("creative_url", "")
        if (
            isinstance(cf_raw, str)
            and cf_raw in {"image", "carousel"}
            and isinstance(cu_raw, str)
            and cu_raw.strip()
        ):
            raw_image_capable += 1

        pool = total - cumulative_dropped
        if ad.composite_score < sel.min_composite:
            gate("composite_floor", True, pool)
            cumulative_dropped += 1
            continue
        gate("composite_floor", False, pool)

        copy_chars = (
            len(ad.headline.strip())
            + len(ad.body.strip())
            + len(ad.description.strip())
            + len(ad.cta.strip())
        )
        pool = total - cumulative_dropped
        if copy_chars < sel.min_copy_chars:
            gate("min_copy_chars", True, pool)
            cumulative_dropped += 1
            continue
        gate("min_copy_chars", False, pool)

        pool = total - cumulative_dropped
        if sel.english_only:
            lang = ad_dict.get("language", "")
            if not isinstance(lang, str):
                lang = ""
            if lang and lang != "en":
                gate("non_english", True, pool)
                cumulative_dropped += 1
                continue
        gate("non_english", False, pool)

        pool = total - cumulative_dropped
        if sel.drop_unsafe:
            cs_label = ad_dict.get("content_safety_label", "")
            cs_conf = ad_dict.get("content_safety_confidence", 0.0)
            if not isinstance(cs_label, str):
                cs_label = ""
            if not isinstance(cs_conf, (int, float)):
                cs_conf = 0.0
            if (
                cs_label in unsafe_set
                and float(cs_conf) >= sel.content_safety_min_confidence
            ):
                gate("content_safety", True, pool)
                bucket_drops["content_safety"][cs_label] += 1
                cumulative_dropped += 1
                continue
        gate("content_safety", False, pool)

        pool = total - cumulative_dropped
        if sel.business_vertical_min_confidence > 0:
            bv_conf = ad_dict.get("business_vertical_confidence", 0.0)
            if not isinstance(bv_conf, (int, float)):
                bv_conf = 0.0
            if float(bv_conf) < sel.business_vertical_min_confidence:
                gate("business_vertical_low_confidence", True, pool)
                cumulative_dropped += 1
                continue
        gate("business_vertical_low_confidence", False, pool)

        pool = total - cumulative_dropped
        if drop_verticals:
            bv = ad_dict.get("business_vertical", "")
            if isinstance(bv, str) and bv in drop_verticals:
                gate("vertical_drop", True, pool)
                bucket_drops["vertical"][bv] += 1
                cumulative_dropped += 1
                continue
        gate("vertical_drop", False, pool)

        pool = total - cumulative_dropped
        if sel.drop_religious_scripture:
            flagged, reason = is_religious_scripture_text(ad.ad_copy_text)
            if flagged:
                gate("scripture", True, pool)
                bucket_drops["scripture"][reason] += 1
                cumulative_dropped += 1
                continue
        gate("scripture", False, pool)

        pool = total - cumulative_dropped
        if sel.drop_structural_artifacts:
            artifact = _has_structural_artifact(ad.headline, ad.body, ad.description)
            if artifact is not None:
                gate("structural", True, pool)
                bucket_drops["structural"][artifact] += 1
                cumulative_dropped += 1
                continue
        gate("structural", False, pool)

        pool = total - cumulative_dropped
        if sel.min_training_quality > 0:
            tq = ad_dict.get("training_quality", 0)
            if not isinstance(tq, int):
                tq = 0
            if tq != 0 and tq < sel.min_training_quality:
                gate("training_quality", True, pool)
                bucket_drops["training_quality"][str(tq)] += 1
                cumulative_dropped += 1
                continue
        gate("training_quality", False, pool)

        pool = total - cumulative_dropped
        if platform_group_for(ad.platform) is not PlatformLabelGroup.OTHER:
            labeled = render_labeled_ad(ad)
            if not labeled.strip():
                gate("empty_labeled_render", True, pool)
                cumulative_dropped += 1
                continue
        gate("empty_labeled_render", False, pool)

        # Image-capable.
        creative_format = ad_dict.get("creative_format", "")
        creative_url = ad_dict.get("creative_url", "")
        if not isinstance(creative_format, str):
            creative_format = ""
        if not isinstance(creative_url, str):
            creative_url = ""
        pool = total - cumulative_dropped
        if creative_format not in {"image", "carousel"}:
            gate("image_capable_format", True, pool)
            bucket_drops["image_capable_format"][creative_format or "missing"] += 1
            cumulative_dropped += 1
            continue
        gate("image_capable_format", False, pool)

        pool = total - cumulative_dropped
        if not creative_url.strip():
            gate("image_capable_url", True, pool)
            cumulative_dropped += 1
            continue
        gate("image_capable_url", False, pool)

        survivors.append(ad.ad_id)

    # ----- Report -----
    print(f"Source corpus: {scored_path}")
    print(f"Total rows:    {total:,}")
    print(
        f"Config:        {CONFIG_PATH}  (skill={config.skill}, "
        f"min_composite={sel.min_composite}, "
        f"min_training_quality={sel.min_training_quality}, "
        f"target_count={sel.target_count})"
    )
    print()
    print(
        f"Context — raw image-capable subset (image|carousel + url, no quality "
        f"gates): {raw_image_capable:,} ({100*raw_image_capable/total:.2f}% of total)"
    )
    print()
    print("─" * 84)
    print("CONSTRUCTION FUNNEL (image-brief skill, all gates in order)")
    print("─" * 84)
    print(
        f"{'  gate':32s}{'drops':>14s}{'% total':>14s}{'% of stage':>14s}{'survivors':>14s}"
    )
    print("  " + "─" * 82)
    running = total
    print(f"  {'start':30s}{'':>14s}{'':>14s}{'':>14s}{running:>14,}")
    for name in gate_order:
        d = gate_drops[name]
        if d == 0:
            continue
        pool_in = running  # rows that reached this gate
        running -= d
        pct_total = 100.0 * d / total
        pct_stage = 100.0 * d / pool_in if pool_in else 0.0
        print(
            f"  {name:30s}{d:>14,}{pct_total:>13.2f}%{pct_stage:>13.2f}%{running:>14,}"
        )
    print()
    print(f"  selection.parquet (eligible)    {running:,}")
    print(
        f"  config target_count={sel.target_count:,} → "
        f"{min(sel.target_count, running):,} chosen after stratified sample"
    )
    print()

    # Bucket detail for the multi-reason gates.
    print("─" * 84)
    print("DROP REASONS (per-gate breakdown)")
    print("─" * 84)
    for bucket_name, c in bucket_drops.items():
        if not c:
            continue
        print(f"  {bucket_name}:")
        for reason, n in c.most_common():
            print(f"    - {reason:40s} {n:>5,}")
    print()
    print(
        f"  → caption-creatives runs over the {running:,} eligible ad_ids "
        f"(estimated cost @ $0.000275/ad batch tier: ${running * 0.000275:.2f})"
    )


if __name__ == "__main__":
    main()
