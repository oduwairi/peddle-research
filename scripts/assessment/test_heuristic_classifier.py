"""Diagnostic run of the vertical classifier on the English-filtered corpus.

For each ad, tracks which classifier level fired:
  L1 — domain substring match
  L2 — advertiser name regex
  L3 — ad copy regex
  L4 — semantic embedding fallback
  L0 — unknown (nothing matched)

Reports:
  - Overall level distribution
  - Vertical distribution per level
  - Spot-check samples per vertical
  - Unknown-bucket sample (what's left unclassified?)

Run:
    python scripts/assessment/test_heuristic_classifier.py [--sample N] [--semantic]
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from scripts.assessment.heuristic_classifier import (
    _COPY_PATTERNS,
    _DOMAIN_RULES,
    _NAME_PATTERNS,
    _VERTICAL_ANCHORS,
    PRIORITY_ORDER,
    _match_domain,
)

console = Console()
app = typer.Typer()


def classify_with_level(
    advertiser_name: str,
    landing_page_url: str,
    headline: str,
    body: str,
) -> tuple[str, str]:
    """Like classify_ad() but returns (vertical, level) where level is
    'L1_domain', 'L2_name', 'L3_copy', or 'L0_unknown' (pre-semantic).
    """
    # L1
    for v in PRIORITY_ORDER:
        fragments = _DOMAIN_RULES.get(v, [])
        if fragments and _match_domain(landing_page_url, fragments):
            return v, "L1_domain"
    # L2
    name_lower = advertiser_name.lower()
    for v in PRIORITY_ORDER:
        for pat in _NAME_PATTERNS.get(v, []):
            if pat.search(name_lower):
                return v, "L2_name"
    # L3
    copy_lower = (headline + " " + body).lower()
    for v in PRIORITY_ORDER:
        for pat in _COPY_PATTERNS.get(v, []):
            if pat.search(copy_lower):
                return v, "L3_copy"
    return "unknown", "L0_unknown"


def batch_semantic_classify(
    texts: list[str], threshold: float = 0.28
) -> list[tuple[str, float]]:
    """Semantic classify a batch of texts. Returns [(vertical, score)]."""
    from sentence_transformers import SentenceTransformer  # type: ignore[import]

    console.print("[yellow]Loading multilingual sentence-transformer model…[/yellow]")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    anchors = list(_VERTICAL_ANCHORS.items())
    anchor_verticals = [v for v, _ in anchors]
    anchor_texts = [t for _, t in anchors]
    anchor_mat: np.ndarray = model.encode(
        anchor_texts, normalize_embeddings=True, show_progress_bar=False
    )

    console.print(f"[yellow]Embedding {len(texts):,} unknown ads…[/yellow]")
    ad_mat: np.ndarray = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64
    )

    sims: np.ndarray = ad_mat @ anchor_mat.T  # (n_ads, n_verticals)
    best_idx = np.argmax(sims, axis=1)
    best_scores = sims[np.arange(len(texts)), best_idx]

    results: list[tuple[str, float]] = []
    for i, score in enumerate(best_scores):
        if score >= threshold:
            results.append((anchor_verticals[int(best_idx[i])], float(score)))
        else:
            results.append(("unknown", float(score)))
    return results


@app.command()
def run(
    scored_path: str = typer.Option(
        "data/scored/v3/scored_ads.jsonl",
        "--scored",
        help="Path to scored_ads.jsonl.",
    ),
    sample: int = typer.Option(
        0, "--sample", help="If >0, use a random sample of N ads instead of all."
    ),
    run_semantic: bool = typer.Option(
        True, "--semantic/--no-semantic", help="Run the L4 semantic fallback."
    ),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    rng = random.Random(seed)
    path = Path(scored_path)
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    # ── Load ────────────────────────────────────────────────────────────────
    console.print(f"Loading {path}…")
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    console.print(f"Total records: {len(records):,}")

    # English filter
    records = [r for r in records if r.get("ad", {}).get("language", "") in {"en", ""}]
    console.print(f"After English filter: {len(records):,}")

    if sample > 0 and sample < len(records):
        records = rng.sample(records, sample)
        console.print(f"Sampled: {len(records):,}")

    # ── L1-L3 classification ────────────────────────────────────────────────
    level_counts: Counter[str] = Counter()
    vertical_counts: Counter[str] = Counter()
    per_level_vertical: dict[str, Counter[str]] = defaultdict(Counter)
    samples_per_vertical: dict[str, list[dict]] = defaultdict(list)
    unknown_records: list[dict] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running L1-L3 classifier…", total=len(records))
        for rec in records:
            ad = rec.get("ad", rec)
            adv = ad.get("advertiser_name", "") or ""
            url = ad.get("landing_page_url", "") or ""
            h = ad.get("ad_copy", {}).get("headline", "") or ""
            b = ad.get("ad_copy", {}).get("body", "") or ""
            vertical, level = classify_with_level(adv, url, h, b)

            level_counts[level] += 1
            vertical_counts[vertical] += 1
            per_level_vertical[level][vertical] += 1
            if len(samples_per_vertical[vertical]) < 5:
                samples_per_vertical[vertical].append(
                    {"adv": adv, "url": url, "h": h[:70], "level": level}
                )
            if level == "L0_unknown":
                unknown_records.append(
                    {
                        "ad_id": ad.get("ad_id", ""),
                        "adv": adv,
                        "url": url,
                        "h": h,
                        "b": b,
                    }
                )
            progress.advance(task)

    # ── L4 semantic on the unknowns ─────────────────────────────────────────
    semantic_results: list[tuple[str, float]] = []
    if run_semantic and unknown_records:
        texts = [
            f"{u['adv']} {u['h']} {u['b']}".strip()[:400] for u in unknown_records
        ]
        semantic_results = batch_semantic_classify(texts)
        sem_vertical_counts: Counter[str] = Counter()
        sem_score_buckets: Counter[str] = Counter()
        for (v, score), u in zip(semantic_results, unknown_records, strict=False):
            sem_vertical_counts[v] += 1
            if v == "unknown":
                sem_score_buckets["below_threshold"] += 1
            elif score >= 0.50:
                sem_score_buckets["strong_≥0.50"] += 1
            elif score >= 0.40:
                sem_score_buckets["medium_0.40-0.50"] += 1
            else:
                sem_score_buckets["weak_0.28-0.40"] += 1
            if v != "unknown":
                level_counts["L4_semantic"] += 1
                vertical_counts[v] += 1
                vertical_counts["unknown"] -= 1
                per_level_vertical["L4_semantic"][v] += 1
                if len(samples_per_vertical[v]) < 7:
                    samples_per_vertical[v].append(
                        {
                            "adv": u["adv"],
                            "url": u["url"],
                            "h": u["h"][:70],
                            "level": f"L4 ({score:.2f})",
                        }
                    )

        # rewrite L0 bucket: the still-unknown after semantic
        still_unknown = sum(1 for (v, _) in semantic_results if v == "unknown")
        level_counts["L0_unknown"] = still_unknown

    # ── Report ──────────────────────────────────────────────────────────────
    total = sum(level_counts.values())

    console.rule("[bold]Level distribution[/bold]")
    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Level")
    t.add_column("Count", justify="right")
    t.add_column("%", justify="right")
    t.add_column("Description")
    level_desc = {
        "L1_domain": "URL domain substring match",
        "L2_name": "advertiser-name regex",
        "L3_copy": "ad-copy regex",
        "L4_semantic": "multilingual embedding ≥0.28",
        "L0_unknown": "nothing matched",
    }
    for lvl in ["L1_domain", "L2_name", "L3_copy", "L4_semantic", "L0_unknown"]:
        c = level_counts.get(lvl, 0)
        t.add_row(lvl, f"{c:,}", f"{c/total:.1%}", level_desc[lvl])
    console.print(t)

    if run_semantic and semantic_results:
        console.rule("[bold]L4 semantic score buckets (of the L0→L4 handoff)[/bold]")
        t2 = Table(show_header=True, header_style="bold cyan")
        t2.add_column("Bucket")
        t2.add_column("Count", justify="right")
        for k, v in sem_score_buckets.most_common():
            t2.add_row(k, f"{v:,}")
        console.print(t2)

    console.rule("[bold]Vertical distribution (final)[/bold]")
    t3 = Table(show_header=True, header_style="bold cyan")
    t3.add_column("Vertical")
    t3.add_column("Count", justify="right")
    t3.add_column("%", justify="right")
    for v, c in vertical_counts.most_common():
        t3.add_row(v, f"{c:,}", f"{c/total:.1%}")
    console.print(t3)

    console.rule("[bold]Per-level vertical distribution[/bold]")
    for lvl in ["L1_domain", "L2_name", "L3_copy", "L4_semantic"]:
        per_lvl = per_level_vertical.get(lvl, Counter())
        if not per_lvl:
            continue
        top5 = per_lvl.most_common(5)
        line = "  ".join(f"{v}={c}" for v, c in top5)
        console.print(f"  [bold]{lvl}[/bold]: {line}")

    console.rule("[bold]Spot-check samples per vertical[/bold]")
    shown_verticals = sorted(
        [v for v in samples_per_vertical if v != "unknown"],
        key=lambda v: -vertical_counts[v],
    )
    for v in shown_verticals:
        console.print(f"\n[bold cyan]▸ {v}[/bold cyan] (n={vertical_counts[v]:,})")
        for s in samples_per_vertical[v][:5]:
            console.print(
                f"    [{s['level']:<14}] {s['adv'][:35]:<35} | "
                f"{s['url'][:30]:<30} | {s['h']}"
            )

    # Unknown bucket samples (after semantic)
    console.rule("[bold]Unknown bucket (after semantic)[/bold]")
    if run_semantic and semantic_results:
        still_unk = [
            u
            for u, (v, _) in zip(unknown_records, semantic_results, strict=False)
            if v == "unknown"
        ]
    else:
        still_unk = unknown_records
    console.print(f"Still unknown: {len(still_unk):,}")
    for u in rng.sample(still_unk, min(15, len(still_unk))):
        console.print(
            f"    {u['adv'][:35]:<35} | {u['url'][:30]:<30} | {u['h'][:70]}"
        )


if __name__ == "__main__":
    app()
