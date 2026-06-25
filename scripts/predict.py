"""Scoring-predictor CLI — train + offline-eval + ad-hoc scoring.

Subcommands:

* ``train`` — train one model on one split. Materializes splits if missing.
* ``eval-offline`` — run the metrics suite against a trained checkpoint.
* ``predict`` — score one ad from the command line (smoke test).
* ``splits`` — materialize all three splits without training.

Mirrors ``scripts/score.py`` Typer style. Plan reference:
``~/.claude/plans/ok-heres-important-point-indexed-minsky.md`` (Phase 1).
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from draper.scoring_predictor.config import PredictorConfig
from draper.scoring_predictor.data import iter_examples, load_corpus
from draper.scoring_predictor.eval_offline import evaluate_split, write_report
from draper.scoring_predictor.splits import (
    SPLIT_NAMES,
    make_heldout_platform_split,
    make_heldout_vertical_split,
    make_random_split,
)
from draper.scoring_predictor.train import train_predictor
from draper.utils.logging import setup_logging

app = typer.Typer(help="Train and evaluate the v3 ad-performance predictor")
console = Console()
log = setup_logging()

DEFAULT_CONFIG = "configs/scoring_predictor.yaml"


@app.command()
def train(
    config_path: str = typer.Option(DEFAULT_CONFIG, "--config", help="Predictor config YAML"),
    split: str = typer.Option(
        "random", "--split", help=f"One of {SPLIT_NAMES}; default = random"
    ),
) -> None:
    """Train one model on one split."""
    cfg = PredictorConfig.from_yaml(config_path)
    if split not in SPLIT_NAMES:
        console.print(f"[red]Unknown split: {split}. Expected one of {SPLIT_NAMES}.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Training[/bold] split={split} model={cfg.model_name}")
    out = train_predictor(cfg, split_name=split)  # type: ignore[arg-type]
    console.print(f"[green]Trained checkpoint at {out}[/green]")


@app.command(name="eval-offline")
def eval_offline(
    config_path: str = typer.Option(DEFAULT_CONFIG, "--config", help="Predictor config YAML"),
    split: str = typer.Option(
        "random", "--split", help=f"One of {SPLIT_NAMES}; default = random"
    ),
    checkpoint: str | None = typer.Option(
        None,
        "--checkpoint",
        help="Override the checkpoint path. Defaults to {checkpoint_dir}/{split}/best.",
    ),
    skip_calibration: bool = typer.Option(
        False, "--skip-calibration", help="Don't fit isotonic calibrators if missing"
    ),
    batch_size: int = typer.Option(64, "--batch-size", help="Eval batch size"),
    output: str | None = typer.Option(
        None, "--output", help="Where to write the JSON report (default: report next to checkpoint)"
    ),
) -> None:
    """Run the metrics suite against a trained checkpoint."""
    cfg = PredictorConfig.from_yaml(config_path)
    if split not in SPLIT_NAMES:
        console.print(f"[red]Unknown split: {split}.[/red]")
        raise typer.Exit(1)
    ckpt = Path(checkpoint) if checkpoint else None
    report = evaluate_split(
        config=cfg,
        split_name=split,  # type: ignore[arg-type]
        checkpoint_dir=ckpt,
        fit_calibration=not skip_calibration,
        batch_size=batch_size,
    )

    out_path = Path(output) if output else Path(report["checkpoint"]) / "eval_report.json"
    write_report(report, out_path)
    _render_report(report)
    console.print(f"\n[dim]Wrote {out_path}[/dim]")


@app.command()
def splits(
    config_path: str = typer.Option(DEFAULT_CONFIG, "--config", help="Predictor config YAML"),
    force: bool = typer.Option(False, "--force", help="Re-materialize even if splits exist"),
) -> None:
    """Materialize the three splits without training."""
    cfg = PredictorConfig.from_yaml(config_path)
    df = load_corpus(cfg.corpus_path)
    examples = list(iter_examples(df))
    console.print(f"Loaded {len(examples)} usable examples (post-filter, post-quality-1-drop)")

    from draper.scoring_predictor.data import examples_to_polars

    materialized = examples_to_polars(examples)
    cfg.splits_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "random": make_random_split,
        "heldout-platform": make_heldout_platform_split,
        "heldout-vertical": make_heldout_vertical_split,
    }
    for name, builder in builders.items():
        out = cfg.splits_dir / name
        if out.exists() and not force and (out / "train.parquet").exists():
            console.print(f"  [dim]{name}: already present[/dim]")
            continue
        split = builder(materialized)
        split.write(cfg.splits_dir)
        console.print(
            f"  [green]{name}[/green]: train={len(split.train):,} "
            f"val={len(split.val):,} test={len(split.test):,}"
        )


@app.command()
def predict(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to {checkpoint_dir}/{split}/best"
    ),
    platform: str = typer.Option(
        ..., "--platform", help="Platform (e.g. facebook, tiktok)"
    ),
    vertical: str = typer.Option("unknown", "--vertical", help="Vertical bucket"),
    headline: str = typer.Option("", "--headline"),
    body: str = typer.Option("", "--body"),
    description: str = typer.Option("", "--description"),
) -> None:
    """One-shot CLI scoring — useful for smoke testing a checkpoint."""
    from draper.scoring_predictor.inference import load_predictor

    predictor = load_predictor(checkpoint)
    result = predictor.score_text(
        platform=platform,
        vertical=vertical,
        headline=headline or None,
        body=body or None,
        description=description or None,
    )
    console.print(json.dumps(result, indent=2))


def _render_report(report: dict[str, object]) -> None:
    metrics = report.get("metrics", {})
    table = Table(title=f"Eval — split={report.get('split')}  n={report.get('n_test')}")
    table.add_column("Head")
    table.add_column("Spearman", justify="right")
    table.add_column("Pearson", justify="right")
    table.add_column("MAE", justify="right")
    for head in ("composite", "survivability", "engagement_volume", "engagement_velocity"):
        if isinstance(metrics, dict):
            sp = metrics.get(f"spearman_{head}", float("nan"))
            pe = metrics.get(f"pearson_{head}", float("nan"))
            ma = metrics.get(f"mae_{head}", float("nan"))
            table.add_row(
                head,
                _fmt(sp),
                _fmt(pe),
                _fmt(ma),
            )
    console.print(table)

    console.print(
        f"composite ECE = {_fmt(report.get('composite_ece'))}   "
        f"AUC top-tier (≥{0.80}) = {_fmt(report.get('composite_auc_top_tier'))}   "
        f"AUC bottom-tier (≤{0.30}) = {_fmt(report.get('composite_auc_bottom_tier'))}"
    )

    pp = report.get("per_platform", {})
    if isinstance(pp, dict) and pp:
        ptable = Table(title="Per-platform composite Spearman")
        ptable.add_column("Platform")
        ptable.add_column("n", justify="right")
        ptable.add_column("Spearman", justify="right")
        ptable.add_column("MAE", justify="right")
        for plat, stats in sorted(pp.items()):
            if isinstance(stats, dict):
                ptable.add_row(
                    plat,
                    str(stats.get("n", 0)),
                    _fmt(stats.get("spearman_composite")),
                    _fmt(stats.get("mae_composite")),
                )
        console.print(ptable)


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    app()
