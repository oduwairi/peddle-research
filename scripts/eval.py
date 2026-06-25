"""Eval pipeline CLI — drives inference, judging, aggregation, reporting.

Usage:
    python scripts/eval.py briefs sample --n 5
    python scripts/eval.py infer --configs A,B,C --split test
    python scripts/eval.py scenarios run --configs A_pipe,C_pipe

    # Normalize — extract clean ad copy BEFORE judging (required).
    # Strips rationale, <think> blocks, emoji-spam, and model-collapse artifacts.
    # Caches under data/eval/inferences_clean/. ~$2 with haiku-4-5 (default),
    # cheaper with --extractor gemini-2.5-flash.
    python scripts/eval.py normalize --configs A,B,C,GOLD
    python scripts/eval.py normalize --configs A,B,C,GOLD --extractor gemini-2.5-flash

    # Live judging (sync, full price):
    python scripts/eval.py judge \
        --pair A,C --pair B,C --pair A,B \
        --pair C,GOLD --pair A,GOLD --pair B,GOLD \
        --judge claude-sonnet-4-6 --judge gemini-2.5-flash

    # Batch judging (50% off, 24h SLA — OpenAI + Anthropic only):
    python scripts/eval.py judge-batch submit \
        --pair C,GOLD --judge claude-sonnet-4-6 --run-id may-smoke
    python scripts/eval.py judge-batch status \
        --run-id may-smoke --pair C,GOLD --judge claude-sonnet-4-6
    python scripts/eval.py judge-batch collect \
        --run-id may-smoke --pair C,GOLD --judge claude-sonnet-4-6

    # Aggregate (reads live or batch judgments — same on-disk shape).
    # run_id is YYYY-MM-DD-<slug>; writes runs/<run_id>/aggregates/.
    python scripts/eval.py aggregate --run-id 2026-05-15-smoke \
        --groupby platform,vertical,source_tier --similarity

    # Learned-scorer absolute arm (engagement predictions for held-out test set):
    python scripts/eval.py score --configs A,B,C,GOLD
    python scripts/eval.py score-summary --configs A,B,C,GOLD --run-id 2026-05-15-smoke

    # MAUVE distribution-matching arm (requires `uv pip install -e ".[mauve]"`):
    python scripts/eval.py mauve --configs A,B,C,GOLD
    python scripts/eval.py mauve-summary --run-id 2026-05-15-mauve-v1

    # Reference-overlap arm — BLEU/chrF/ROUGE-L/METEOR/BERTScore vs the GOLD ad
    # + a nearest-neighbor multi-ref pool (requires `uv pip install -e ".[refmetrics]"`):
    python scripts/eval.py reference-metrics --configs A,B,C,GOLD --no-bertscore
    python scripts/eval.py reference-summary --configs A,B,C,GOLD \
        --run-id 2026-06-03-refmetrics-v2 --groupby platform
    # Grounding: do the metrics predict real Upworthy A/B CTR winners?
    python scripts/eval.py reference-validate --metrics bleu,chrf,rouge_l,meteor --limit 200

    python scripts/eval.py report --run-id 2026-05-15-smoke

    # Compare a candidate run vs an existing baseline:
    python scripts/eval.py compare \
        --base 2026-05-14-clean-pipe --candidate 2026-05-15-hook-v2

    # Validate judges against Upworthy A/B ground truth (run before trusting verdicts):
    python scripts/eval.py validate \
        --judge claude-sonnet-4-6 --judge gemini-2.5-flash --judge gpt-5.4-mini \
        --stream upworthy --limit 200
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Load .env so judge/runner clients pick up OPENAI_API_KEY, GEMINI_API_KEY,
# OPENROUTER_API_KEY, VLLM_API_KEY, etc. — same pattern as collect.py / scrape.py.
load_dotenv()

# Make `draper.*` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from draper.evaluation.briefs import (  # noqa: E402
    load_test_briefs,
    load_url_scenarios,
)
from draper.evaluation.config import EvalConfig  # noqa: E402
from draper.evaluation.driver import (  # noqa: E402
    load_inferences_for_config,
    load_pair_results,
    run_inference_for_briefs,
    run_inference_for_scenarios,
    run_judge_pass,
    runners_from_config,
)
from draper.evaluation.gold import (  # noqa: E402
    gold_inferences_from_briefs,
    is_gold,
)
from draper.evaluation.judge.aggregation import (  # noqa: E402
    SEGMENT_COLUMNS,
    bootstrap_win_rate_ci,
    cohen_kappa,
    elo_ratings,
    pair_results_to_dataframe,
    win_rates_table,
)
from draper.evaluation.judge.batch import (  # noqa: E402
    BatchManifest,
    build_anthropic_batch_requests,
    build_openai_batch_jsonl,
    collect_anthropic_batch,
    collect_openai_batch,
    parsed_to_judgments,
    provider_for_model,
    status_anthropic_batch,
    status_openai_batch,
    submit_anthropic_batch,
    submit_openai_batch,
)
from draper.evaluation.judge.normalize import (  # noqa: E402
    DEFAULT_EXTRACTOR_MODEL,
    EXTRACTION_FAILED,
    campaign_published_copy,
    extract_and_cache,
    load_clean,
)
from draper.evaluation.judge.similarity import similarity_to_gold  # noqa: E402
from draper.evaluation.judge.validation import (  # noqa: E402
    validate_judge_on_upworthy_pairs,
)
from draper.evaluation.learned_scorer import SCORE_COLUMNS as LEARNED_HEAD_COLUMNS  # noqa: E402
from draper.evaluation.learned_scorer import score_configs as learned_score_configs  # noqa: E402
from draper.evaluation.learned_scorer import summarize as learned_summarize  # noqa: E402
from draper.evaluation.mauve_scorer import summarize as mauve_summarize  # noqa: E402
from draper.evaluation.reference_metrics import summarize as reference_summarize  # noqa: E402
from draper.evaluation.schemas import Brief, Inference, UrlScenario  # noqa: E402
from draper.evaluation.upworthy_loader import UpworthyLoader  # noqa: E402
from draper.utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

app = typer.Typer(help="Eval pipeline for the fine-tuned 7B copywriter.")
briefs_app = typer.Typer(help="Inspect / sample the held-out test briefs.")
scenarios_app = typer.Typer(help="Run Arm 2 URL-anchored scenarios.")
batch_app = typer.Typer(help="Batch-API judge runs — 50% off, 24h SLA. OpenAI + Anthropic only.")
app.add_typer(briefs_app, name="briefs")
app.add_typer(scenarios_app, name="scenarios")
app.add_typer(batch_app, name="judge-batch")
console = Console()


def _load_cfg(path: Path) -> EvalConfig:
    if not path.exists():
        console.print(f"[red]Config not found: {path}[/red]")
        raise typer.Exit(code=1)
    return EvalConfig.from_yaml(path)


# ---- briefs --------------------------------------------------------------


@briefs_app.command("sample")
def briefs_sample(
    config: Path = typer.Option(  # noqa: B008
        Path("configs/eval.yaml"), help="Eval config YAML."
    ),
    n: int = typer.Option(5, help="Number of briefs to print."),
) -> None:
    """Phase 0 sanity: print the first N briefs in the test split."""
    setup_logging(level="INFO")
    cfg = _load_cfg(config)
    briefs = load_test_briefs(cfg.test_split_dir)
    console.print(f"[green]Loaded {len(briefs)} briefs from {cfg.test_split_dir}[/green]")
    for b in briefs[:n]:
        console.rule(f"{b.example_id} — {b.vertical} / {b.platform}")
        console.print(f"[dim]system:[/dim] {b.system}")
        console.print(f"[bold]user:[/bold] {b.user}")
        console.print(f"[dim]reference assistant (held back):[/dim]\n{b.reference_assistant}")


# ---- infer (Arm 1) -------------------------------------------------------


@app.command("infer")
def infer(
    configs: str = typer.Option(..., help="Comma-separated config names (e.g. 'A,B,C')."),
    split: str = typer.Option("test", help="Dataset split (currently only 'test')."),
    config_path: Path = typer.Option(  # noqa: B008
        Path("configs/eval.yaml"), help="Eval config YAML."
    ),
    limit: int | None = typer.Option(None, help="Limit to first N briefs (smoke / debug)."),
    concurrency: int = typer.Option(8, help="Max in-flight requests per runner."),
    force: bool = typer.Option(False, help="Re-run even if inference JSON exists."),
) -> None:
    """Run Arm 1 single-shot inference for one or more configs."""
    setup_logging(level="INFO")
    if split != "test":
        console.print(f"[red]Only 'test' split supported, got {split!r}[/red]")
        raise typer.Exit(code=1)
    cfg = _load_cfg(config_path)
    config_names = [c.strip() for c in configs.split(",") if c.strip()]
    runners = runners_from_config(cfg, config_names)

    briefs = load_test_briefs(cfg.test_split_dir)
    if limit is not None:
        briefs = briefs[:limit]
    console.print(f"[green]Running {len(config_names)} configs × {len(briefs)} briefs[/green]")

    out_root = cfg.inferences_dir

    async def _go() -> None:
        for name, runner in runners.items():
            written, skipped, errored = await run_inference_for_briefs(
                runner=runner,
                briefs=briefs,
                out_root=out_root,
                force=force,
                max_concurrency=concurrency,
            )
            console.print(
                f"[bold]{name}[/bold]: wrote {written}, skipped {skipped}, errored {errored}"
            )

    asyncio.run(_go())


# ---- normalize (LLM ad-copy extraction, runs between infer and judge) ----


@app.command("normalize")
def normalize(
    configs: str = typer.Option(
        ...,
        help=(
            "Comma-separated config names to normalize (e.g. 'A,B,C,GOLD'). "
            "GOLD reads from briefs[reference_assistant] instead of inferences/."
        ),
    ),
    extractor: str = typer.Option(
        DEFAULT_EXTRACTOR_MODEL,
        help="LLM used to extract ad copy from each candidate response.",
    ),
    config_path: Path = typer.Option(  # noqa: B008
        Path("configs/eval.yaml"), help="Eval config YAML."
    ),
    concurrency: int = typer.Option(16, help="Max in-flight extractor calls (per config)."),
    limit: int | None = typer.Option(
        None, help="Limit to first N briefs per config (smoke / debug)."
    ),
    force: bool = typer.Option(
        False,
        help=(
            "Re-extract even if a cached cleaned record exists with a matching source-text SHA256."
        ),
    ),
) -> None:
    """Pre-extract uniform ad copy for each config via an LLM normalizer.

    Writes per-(config, example_id) JSON files to ``inferences_clean/``
    that the judge consumes in place of ``clean_copy(assistant_text)``.
    Required to defeat the regex-extractor asymmetry that contaminated
    the ``may-ft-r16`` baseline judging — see normalize.py module docs.
    """
    setup_logging(level="INFO")
    cfg = _load_cfg(config_path)
    config_names = [c.strip() for c in configs.split(",") if c.strip()]
    clean_root = cfg.inferences_clean_dir
    clean_root.mkdir(parents=True, exist_ok=True)

    briefs_by_id: dict[str, Brief] = {b.example_id: b for b in load_test_briefs(cfg.test_split_dir)}

    async def _normalize_one(
        config_name: str,
        example_id: str,
        raw_text: str,
        platform: str | None,
    ) -> tuple[str, bool, bool]:
        """Returns (example_id, was_cached, extraction_failed)."""
        existing = load_clean(clean_root, config_name, example_id) if not force else None
        was_cached = existing is not None
        record = await extract_and_cache(
            raw=raw_text,
            config=config_name,
            example_id=example_id,
            clean_root=clean_root,
            platform=platform,
            model=extractor,
            force=force,
        )
        failed = record.assistant_text_clean == EXTRACTION_FAILED
        return example_id, was_cached, failed

    async def _run_config(config_name: str) -> tuple[int, int, int, int]:
        """Returns (extracted, cached, failed, errored)."""
        if is_gold(config_name):
            items = [
                (b.example_id, b.reference_assistant, b.platform) for b in briefs_by_id.values()
            ]
        else:
            inf_dir = cfg.inferences_dir / config_name
            if not inf_dir.exists():
                console.print(f"[yellow]No inferences dir for {config_name}: {inf_dir}[/yellow]")
                return 0, 0, 0, 0
            items = []
            for p in sorted(inf_dir.glob("*.json")):
                with p.open("r", encoding="utf-8") as f:
                    inf = Inference.model_validate(json.load(f))
                if inf.error:
                    continue
                platform = (
                    briefs_by_id[inf.example_id].platform
                    if inf.example_id in briefs_by_id
                    else None
                )
                # Arm-2 (agent-pipeline) runs carry the canonical published copy
                # in the emitted ``campaign`` object; ``assistant_text`` is often
                # just a one-line summary ("Drafted a PINTEREST campaign: ...").
                # Prefer the campaign copy so the extractor sees real ad text.
                raw_text = inf.assistant_text
                if inf.campaign:
                    camp_copy = campaign_published_copy(inf.campaign)
                    if camp_copy.strip():
                        raw_text = camp_copy
                items.append((inf.example_id, raw_text, platform))

        if limit is not None:
            items = items[:limit]

        sem = asyncio.Semaphore(concurrency)

        async def _wrap(
            ex_id: str, raw: str, platform: str | None
        ) -> tuple[str, bool, bool] | BaseException:
            async with sem:
                try:
                    return await _normalize_one(config_name, ex_id, raw, platform)
                except Exception as exc:  # extractor LLM error — log + continue
                    return exc

        results = await asyncio.gather(*[_wrap(*item) for item in items], return_exceptions=False)
        extracted = cached = failed = errored = 0
        for r in results:
            if isinstance(r, BaseException):
                errored += 1
                continue
            _, was_cached, did_fail = r
            if was_cached:
                cached += 1
            else:
                extracted += 1
            if did_fail:
                failed += 1
        return extracted, cached, failed, errored

    async def _go() -> None:
        for name in config_names:
            console.print(f"\n[bold]Normalizing {name}[/bold] → {clean_root / name}/")
            extracted, cached, failed, errored = await _run_config(name)
            tag = "[red]" if errored else "[green]"
            console.print(
                f"{tag}{name}: extracted {extracted}, cached {cached}, "
                f"<EXTRACTION_FAILED> {failed}, errored {errored}[/]"
            )

    asyncio.run(_go())


# ---- scenarios run (Arm 2) -----------------------------------------------


@scenarios_app.command("run")
def scenarios_run(
    configs: str = typer.Option(..., help="Comma-separated config names (e.g. 'A_pipe,C_pipe')."),
    config_path: Path = typer.Option(  # noqa: B008
        Path("configs/eval.yaml"), help="Eval config YAML."
    ),
    file: Path | None = typer.Option(  # noqa: B008
        None, help="Override URL scenarios JSONL path."
    ),
    concurrency: int = typer.Option(2, help="Max in-flight pipeline runs (lower for tools)."),
    force: bool = typer.Option(False, help="Re-run even if inference JSON exists."),
) -> None:
    """Run Arm 2 inference: URL-anchored scenarios via the frontend pipeline."""
    setup_logging(level="INFO")
    cfg = _load_cfg(config_path)
    config_names = [c.strip() for c in configs.split(",") if c.strip()]
    runners = runners_from_config(cfg, config_names)
    path = file or cfg.url_scenarios_path
    scenarios = load_url_scenarios(path)
    if not scenarios:
        console.print(f"[red]No scenarios at {path}[/red]")
        raise typer.Exit(code=1)
    console.print(
        f"[green]Running {len(config_names)} configs × {len(scenarios)} scenarios[/green]"
    )
    out_root = cfg.inferences_dir

    async def _go() -> None:
        for name, runner in runners.items():
            written, skipped, errored = await run_inference_for_scenarios(
                runner=runner,
                scenarios=scenarios,
                out_root=out_root,
                force=force,
                max_concurrency=concurrency,
            )
            console.print(
                f"[bold]{name}[/bold]: wrote {written}, skipped {skipped}, errored {errored}"
            )

    asyncio.run(_go())


# ---- judge ---------------------------------------------------------------


@app.command("judge")
def judge_cmd(
    pairs: list[str] = typer.Option(  # noqa: B008
        ..., "--pair", help="Pair like 'A,C' (repeatable for multiple pairs)."
    ),
    judge: list[str] = typer.Option(  # noqa: B008
        ["claude-sonnet-4-6"], "--judge", help="Judge model (repeatable)."
    ),
    config_path: Path = typer.Option(  # noqa: B008
        Path("configs/eval.yaml"), help="Eval config YAML."
    ),
    arm: str = typer.Option("arm1", help="Which arm: 'arm1' (briefs) or 'arm2' (scenarios)."),
    swap: bool = typer.Option(True, help="Run both orderings (position swap)."),
    concurrency: int = typer.Option(8, help="Max in-flight judge calls."),
    force: bool = typer.Option(False, help="Re-judge even if judgment JSON exists."),
    limit: int | None = typer.Option(None, help="Limit to first N examples per pair."),
) -> None:
    """Run pairwise judging for one or more pairs across one or more judges."""
    setup_logging(level="INFO")
    cfg = _load_cfg(config_path)

    # Pre-load brief/scenario context so the judge prompt has platform + vertical.
    briefs_by_id = (
        {b.example_id: b for b in load_test_briefs(cfg.test_split_dir)} if arm == "arm1" else {}
    )
    scenarios_by_id = (
        {s.scenario_id: s for s in load_url_scenarios(cfg.url_scenarios_path)}
        if arm == "arm2"
        else {}
    )

    pair_tuples: list[tuple[str, str]] = []
    for raw in pairs:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 2:
            console.print(f"[red]Bad pair {raw!r}; expected 'A,B'[/red]")
            raise typer.Exit(code=1)
        pair_tuples.append((parts[0], parts[1]))

    # GOLD pairs synthesize one side from briefs (arm 1 only — arm 2 URL
    # scenarios have no held-out reference ad). Gate early.
    has_gold_pair = any(is_gold(p[0]) or is_gold(p[1]) for p in pair_tuples)
    if has_gold_pair and arm != "arm1":
        console.print(
            f"[red]GOLD pairs require arm='arm1' (got {arm!r}); "
            "URL scenarios have no reference ad.[/red]"
        )
        raise typer.Exit(code=1)

    def _load_side(config_name: str) -> dict[str, Inference]:
        if is_gold(config_name):
            return gold_inferences_from_briefs(briefs_by_id.values(), config_name)
        return load_inferences_for_config(cfg.inferences_dir, config_name)

    async def _go() -> None:
        for jm in judge:
            for pair in pair_tuples:
                infs_a = _load_side(pair[0])
                infs_b = _load_side(pair[1])
                if limit is not None:
                    common = sorted(set(infs_a) & set(infs_b))[:limit]
                    infs_a = {k: infs_a[k] for k in common}
                    infs_b = {k: infs_b[k] for k in common}
                written, skipped = await run_judge_pass(
                    pair=pair,
                    inferences_a=infs_a,
                    inferences_b=infs_b,
                    briefs_by_id=briefs_by_id or None,
                    scenarios_by_id=scenarios_by_id or None,
                    judge_model=jm,
                    out_root=cfg.judgments_dir,
                    swap=swap,
                    max_concurrency=concurrency,
                    force=force,
                    clean_root=cfg.inferences_clean_dir,
                )
                console.print(
                    f"[bold]{jm}[/bold] {pair[0]} vs {pair[1]}: wrote {written}, skipped {skipped}"
                )

    asyncio.run(_go())


# ---- aggregate -----------------------------------------------------------


@app.command("aggregate")
def aggregate(
    run_id: str = typer.Option(..., help="Label for this aggregate run (becomes a subdir)."),
    config_path: Path = typer.Option(  # noqa: B008
        Path("configs/eval.yaml"), help="Eval config YAML."
    ),
    arm: str = typer.Option("arm1", help="'arm1' or 'arm2' — picks which pair set to load."),
    groupby: str = typer.Option(
        "",
        help=(
            "Comma-separated segment columns to break win-rates out by "
            f"(any of: {','.join(SEGMENT_COLUMNS)}). Arm-1 only."
        ),
    ),
    similarity: bool = typer.Option(
        False,
        help=(
            "Emit per-example similarity-to-gold diagnostics (rouge_l_f1 + "
            "cosine_to_gold). GOLD pairs only; arm-1 only. Loads embedder."
        ),
    ),
    force: bool = typer.Option(False, help="Overwrite an existing aggregates dir for this run_id."),
) -> None:
    """Aggregate judgments → win-rate table, Elo, bootstrap CIs, manifest."""
    setup_logging(level="INFO")
    cfg = _load_cfg(config_path)
    pairs = cfg.arm1_pairs if arm == "arm1" else cfg.arm2_pairs
    try:
        cfg.paths.assert_aggregates_free(run_id, force=force)
    except FileExistsError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    out_dir = cfg.paths.aggregates_dir(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Brief metadata is needed to enrich pair-result rows with platform /
    # vertical / source_tier for --groupby and to power similarity-to-gold
    # diagnostics. Arm 2 has no held-out reference, so we only load briefs
    # for arm 1.
    briefs_by_id: dict[str, Any] = {}
    if arm == "arm1":
        briefs_by_id = {b.example_id: b for b in load_test_briefs(cfg.test_split_dir)}

    groupby_cols: list[str] = [c.strip() for c in groupby.split(",") if c.strip()]
    if groupby_cols:
        if arm != "arm1":
            console.print(
                f"[red]--groupby requires arm='arm1' (got {arm!r}); "
                "URL scenarios don't carry segment metadata.[/red]"
            )
            raise typer.Exit(code=1)
        for col in groupby_cols:
            if col not in SEGMENT_COLUMNS:
                console.print(
                    f"[red]Unknown groupby column {col!r}; "
                    f"valid options: {', '.join(SEGMENT_COLUMNS)}[/red]"
                )
                raise typer.Exit(code=1)

    judges_to_aggregate = [cfg.judges.primary, cfg.judges.secondary]
    all_results = []
    per_judge_results: dict[str, list] = {jm: [] for jm in judges_to_aggregate}
    for jm in judges_to_aggregate:
        for pair in pairs:
            results = load_pair_results(root=cfg.judgments_dir, judge_model=jm, pair=pair)
            all_results.extend(results)
            per_judge_results[jm].extend(results)

    if not all_results:
        console.print("[red]No judgments found — run `judge` first.[/red]")
        raise typer.Exit(code=1)

    df = pair_results_to_dataframe(all_results, briefs_by_id=briefs_by_id or None)
    summary = win_rates_table(df)
    summary_path = out_dir / "summary.parquet"
    summary.write_parquet(summary_path)
    console.print(f"[green]Wrote {summary_path}[/green]")

    # Per-segment breakdown — written alongside summary, one parquet per
    # segment column so callers can read them independently.
    for col in groupby_cols:
        seg = win_rates_table(df, groupby=[col])
        seg_path = out_dir / f"summary_by_{col}.parquet"
        seg.write_parquet(seg_path)
        console.print(f"[green]Wrote {seg_path}[/green]")

    # Per-example similarity diagnostics for GOLD pairs.
    if similarity and arm == "arm1":
        sim_path = _emit_similarity_diagnostics(
            cfg=cfg,
            pairs=pairs,
            briefs_by_id=briefs_by_id,
            out_dir=out_dir,
        )
        if sim_path:
            console.print(f"[green]Wrote {sim_path}[/green]")

    # Bootstrap CIs.
    cis = bootstrap_win_rate_ci(all_results, n_bootstrap=cfg.bootstrap_n, seed=cfg.bootstrap_seed)
    ci_rows = [
        {
            "config_a": k[0],
            "config_b": k[1],
            "judge_model": k[2],
            "ci_low": v[0],
            "ci_high": v[1],
        }
        for k, v in cis.items()
    ]
    ci_df = pl.DataFrame(ci_rows) if ci_rows else pl.DataFrame()
    ci_path = out_dir / "ci.parquet"
    ci_df.write_parquet(ci_path)

    # Elo per judge.
    elos = {
        jm: elo_ratings(per_judge_results[jm], k=cfg.elo_k, seed=cfg.elo_seed)
        for jm in judges_to_aggregate
        if per_judge_results[jm]
    }
    elo_path = out_dir / "elo.json"
    with elo_path.open("w") as f:
        json.dump(elos, f, indent=2, sort_keys=True)

    # Cross-judge agreement (kappa) on overlapping examples.
    primary_results = per_judge_results.get(cfg.judges.primary, [])
    secondary_results = per_judge_results.get(cfg.judges.secondary, [])
    kappa = (
        cohen_kappa(primary_results, secondary_results)
        if (primary_results and secondary_results)
        else None
    )

    manifest = {
        "run_id": run_id,
        "arm": arm,
        "created_at": datetime.now(UTC).isoformat(),
        "pairs": [list(p) for p in pairs],
        "judges": judges_to_aggregate,
        "n_pair_results": len(all_results),
        "kappa_primary_vs_secondary": kappa,
        "elo": elos,
    }
    manifest_path = cfg.paths.manifest_path(run_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, default=str)

    _print_summary(summary)
    if kappa is not None:
        console.print(f"[dim]Cohen's kappa (primary vs secondary judges): {kappa:.3f}[/dim]")


# ---- score (learned-scorer absolute arm) ---------------------------------


@app.command("score")
def score_cmd(
    configs: str = typer.Option(
        ..., help="Comma-separated config names to score (e.g. 'A,B,C,GOLD')."
    ),
    checkpoint: Path = typer.Option(  # noqa: B008
        Path("data/scoring_predictor/checkpoints/random/best"),
        help="Path to a trained scoring-predictor checkpoint dir.",
    ),
    out: Path | None = typer.Option(  # noqa: B008
        None,
        help="Output dir for per-config parquets (default: cfg.paths.learned_scores_root).",
    ),
    config_path: Path = typer.Option(  # noqa: B008
        Path("configs/eval.yaml"), help="Eval config YAML."
    ),
    batch_size: int = typer.Option(64, help="Predictor batch size."),
    device: str = typer.Option("auto", help="Torch device: 'auto' | 'cpu' | 'cuda'."),
) -> None:
    """Score one or more configs with the trained scoring-predictor.

    Produces one Parquet per config under ``learned_scores/`` with
    calibrated 4-head scores plus brief metadata (platform, vertical, source
    tier). Reads from the existing ``inferences_clean/`` cache (preferred) or
    falls back to raw ``inferences/``. Engagement heads are nulled for
    Reddit / "other" rows (the predictor was trained with masked loss there).
    """
    setup_logging(level="INFO")
    cfg = _load_cfg(config_path)
    config_names = [c.strip() for c in configs.split(",") if c.strip()]
    if not config_names:
        console.print("[red]--configs is empty.[/red]")
        raise typer.Exit(code=1)
    if not checkpoint.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint}[/red]")
        raise typer.Exit(code=1)

    out_dir = out if out is not None else cfg.paths.learned_scores_root

    briefs_by_id = {b.example_id: b for b in load_test_briefs(cfg.test_split_dir)}
    if not briefs_by_id:
        console.print(f"[red]No briefs loaded from {cfg.test_split_dir}[/red]")
        raise typer.Exit(code=1)

    # Lazy import — predictor pulls in torch/transformers/safetensors which we
    # don't want to load for unrelated eval CLI calls.
    from draper.scoring_predictor import load_predictor

    device_arg: str | None = None if device == "auto" else device
    console.print(
        f"[green]Loading predictor from {checkpoint} (device={device_arg or 'auto'})…[/green]"
    )
    predictor = load_predictor(checkpoint, device=device_arg)

    written = learned_score_configs(
        predictor=predictor,
        briefs_by_id=briefs_by_id,
        configs=config_names,
        inferences_clean_dir=cfg.inferences_clean_dir,
        inferences_raw_dir=cfg.inferences_dir,
        out_dir=out_dir,
        batch_size=batch_size,
    )

    if not written:
        console.print("[red]No configs produced any rows.[/red]")
        raise typer.Exit(code=1)

    table = Table(title="Learned scores — per-config means")
    table.add_column("config")
    table.add_column("n")
    for head in LEARNED_HEAD_COLUMNS:
        table.add_column(f"{head}_mean")
    for cfg_name, parquet_path in written.items():
        df = pl.read_parquet(parquet_path)
        row_vals: list[str] = [cfg_name, str(df.height)]
        for head in LEARNED_HEAD_COLUMNS:
            mean_val = df[head].mean()
            row_vals.append("—" if mean_val is None else f"{mean_val:.3f}")
        table.add_row(*row_vals)
    console.print(table)


@app.command("score-summary")
def score_summary(
    configs: str = typer.Option(
        ..., help="Comma-separated config names to summarize (e.g. 'A,B,C,GOLD')."
    ),
    run_id: str = typer.Option(
        ..., help="Label for this summary run (becomes a subdir under aggregates)."
    ),
    out: Path | None = typer.Option(  # noqa: B008
        None,
        help="Input dir for per-config parquets (default: cfg.paths.learned_scores_root).",
    ),
    groupby: str = typer.Option(
        "",
        help=(
            "Comma-separated segment columns to break learned-scores out by "
            f"(any of: {','.join(SEGMENT_COLUMNS)})."
        ),
    ),
    config_path: Path = typer.Option(  # noqa: B008
        Path("configs/eval.yaml"), help="Eval config YAML."
    ),
    force: bool = typer.Option(False, help="Overwrite an existing aggregates dir for this run_id."),
) -> None:
    """Aggregate learned scores into a per-config (and optionally per-segment) summary.

    Reads ``{out}/{config}.parquet`` (default: ``learned_scores/``)
    and writes per-config and per-segment summaries to
    ``runs/{run_id}/aggregates/learned_scores_summary.parquet`` and
    ``…/learned_scores_by_{segment}.parquet``.
    """
    setup_logging(level="INFO")
    cfg = _load_cfg(config_path)
    config_names = [c.strip() for c in configs.split(",") if c.strip()]
    if not config_names:
        console.print("[red]--configs is empty.[/red]")
        raise typer.Exit(code=1)

    in_dir = out if out is not None else cfg.paths.learned_scores_root
    try:
        cfg.paths.assert_aggregates_free(run_id, force=force)
    except (FileExistsError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    out_dir = cfg.paths.aggregates_dir(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    groupby_cols: list[str] = [c.strip() for c in groupby.split(",") if c.strip()]
    for col in groupby_cols:
        # Allow segments that exist on the per-row schema.
        if col not in {"platform", "vertical", "source_tier_first"}:
            console.print(
                f"[red]Unknown groupby column {col!r}; "
                "valid options: platform, vertical, source_tier_first[/red]"
            )
            raise typer.Exit(code=1)

    summary = learned_summarize(out_dir=in_dir, configs=config_names, by=None)
    summary_path = out_dir / "learned_scores_summary.parquet"
    summary.write_parquet(summary_path)
    console.print(f"[green]Wrote {summary_path}[/green]")
    _print_learned_summary(summary)

    for col in groupby_cols:
        seg = learned_summarize(out_dir=in_dir, configs=config_names, by=[col])
        seg_path = out_dir / f"learned_scores_by_{col}.parquet"
        seg.write_parquet(seg_path)
        console.print(f"[green]Wrote {seg_path}[/green]")


def _print_learned_summary(df: pl.DataFrame) -> None:
    """Render the per-config summary table for ``score-summary``."""
    if df.is_empty():
        console.print("[yellow](no rows in summary)[/yellow]")
        return
    table = Table(title="Learned-scores summary")
    table.add_column("config")
    table.add_column("n")
    for head in LEARNED_HEAD_COLUMNS:
        table.add_column(f"{head}_mean")
        table.add_column(f"{head}_median")
    for row in df.iter_rows(named=True):
        cells: list[str] = [str(row["config"]), str(row["n"])]
        for head in LEARNED_HEAD_COLUMNS:
            for stat in ("mean", "median"):
                val = row.get(f"{head}_{stat}")
                cells.append("—" if val is None else f"{val:.3f}")
        table.add_row(*cells)
    console.print(table)


# ---- mauve (distribution-matching arm) ----------------------------------


@app.command("mauve")
def mauve_cmd(
    configs: str = typer.Option(..., help="Comma-separated configs (e.g. 'A,B,C,GOLD')."),
    out: Path | None = typer.Option(  # noqa: B008
        None,
        help="Output dir for per-config parquets (default: cfg.paths.mauve_scores_root).",
    ),
    config_path: Path = typer.Option(  # noqa: B008
        Path("configs/eval.yaml"), help="Eval config YAML."
    ),
    reference_tier: str | None = typer.Option(
        None, help="Override cfg.mauve.reference_tier (e.g. 'high')."
    ),
    bootstrap_n: int | None = typer.Option(
        None, help="Override cfg.mauve.bootstrap_n (use 0 to disable CIs)."
    ),
    embedding_model: str | None = typer.Option(
        None, help="Override cfg.mauve.embedding_model (default 'gpt2-large')."
    ),
    batch_size: int | None = typer.Option(
        None, help="Override cfg.mauve.batch_size for GPT-2 featurization."
    ),
    device: str = typer.Option(
        "auto", help="'auto' | 'cpu' | 'cuda' (or 'cuda:N'). Maps to MAUVE device_id."
    ),
    v3_parquet: Path | None = typer.Option(  # noqa: B008
        None, help="Override cfg.mauve.v3_parquet."
    ),
    refresh_reference: bool = typer.Option(
        False, help="Force-rebuild the reference cache under data/eval/mauve_ref/."
    ),
) -> None:
    """Score configs with MAUVE distribution-matching against the v3 high-tier pool.

    Writes one Parquet per config under ``mauve_scores/`` with one row per
    (config, platform) plus an ``"ALL"`` row. Generations come from
    ``inferences_clean/<config>/`` (or raw fallback); ``GOLD`` is built from
    ``Brief.reference_assistant``. Aborts on text-hash overlap between
    held-out test refs and the v3 reference pool.
    """
    setup_logging(level="INFO")
    cfg = _load_cfg(config_path)
    config_names = [c.strip() for c in configs.split(",") if c.strip()]
    if not config_names:
        console.print("[red]--configs is empty.[/red]")
        raise typer.Exit(code=1)

    m = cfg.mauve
    ref_tier = reference_tier if reference_tier is not None else m.reference_tier
    boot_n = bootstrap_n if bootstrap_n is not None else m.bootstrap_n
    embed_model = embedding_model if embedding_model is not None else m.embedding_model
    bsz = batch_size if batch_size is not None else m.batch_size
    v3_pq = v3_parquet if v3_parquet is not None else m.v3_parquet
    out_dir = out if out is not None else cfg.paths.mauve_scores_root

    device_id = _device_to_id(device)

    briefs = load_test_briefs(cfg.test_split_dir)
    briefs_by_id = {b.example_id: b for b in briefs}
    if not briefs_by_id:
        console.print(f"[red]No briefs loaded from {cfg.test_split_dir}[/red]")
        raise typer.Exit(code=1)

    # Lazy import — pulls torch / GPT-2 weights, only wanted at score time.
    from draper.evaluation.mauve_reference import (
        ContaminationError,
        load_reference_corpus,
    )
    from draper.evaluation.mauve_scorer import score_configs as mauve_score_configs

    console.print(f"[green]Loading reference corpus (tier={ref_tier}, v3={v3_pq})…[/green]")
    # Contamination check must compare ad-copy vs ad-copy. ``Brief.reference_assistant``
    # is the raw teacher output (copy + rationale) and would never overlap with
    # the v3 corpus by hash — passing trivially. Use the cleaned GOLD inferences
    # (rationale-stripped) when available; fall back to reference_assistant
    # only if normalize hasn't been run yet.
    from draper.evaluation.judge.normalize import load_clean

    held_out_texts: list[str] = []
    for brief in briefs:
        rec = load_clean(cfg.inferences_clean_dir, "GOLD", brief.example_id)
        if rec is not None and rec.assistant_text_clean:
            held_out_texts.append(rec.assistant_text_clean)
        elif brief.reference_assistant:
            held_out_texts.append(brief.reference_assistant)
    try:
        reference = load_reference_corpus(
            parquet_path=v3_pq,
            tier=ref_tier,
            platforms=list(m.platforms),
            held_out_texts=held_out_texts,
            cache_dir=cfg.paths.mauve_ref_root,
            force_rebuild=refresh_reference,
        )
    except ContaminationError as exc:
        console.print(f"[red]Contamination check failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]Scoring {len(config_names)} configs "
        f"(embed={embed_model}, bootstrap={boot_n}, device_id={device_id})…[/green]"
    )
    written = mauve_score_configs(
        briefs_by_id=briefs_by_id,
        reference_corpus_by_platform=reference,
        configs=config_names,
        inferences_clean_dir=cfg.inferences_clean_dir,
        inferences_raw_dir=cfg.inferences_dir,
        out_dir=out_dir,
        platforms=list(m.platforms),
        embedding_model=embed_model,
        bootstrap_n=boot_n,
        seed=m.random_seed,
        device_id=device_id,
        max_text_length=m.max_text_length,
        batch_size=bsz,
    )
    if not written:
        console.print("[red]No configs produced any MAUVE rows.[/red]")
        raise typer.Exit(code=1)

    table = Table(title="MAUVE — per-(config, platform)")
    for col in ("config", "platform", "mauve", "ci_low", "ci_high", "n_gen", "n_ref"):
        table.add_column(col)
    for _cfg_name, parquet_path in written.items():
        df = pl.read_parquet(parquet_path)
        for row in df.iter_rows(named=True):
            table.add_row(
                str(row["config"]),
                str(row["platform"]),
                f"{row['mauve']:.4f}" if row["mauve"] is not None else "—",
                f"{row['ci_low']:.4f}" if row["ci_low"] is not None else "—",
                f"{row['ci_high']:.4f}" if row["ci_high"] is not None else "—",
                str(row["n_gen"]),
                str(row["n_ref"]),
            )
    console.print(table)


@app.command("mauve-summary")
def mauve_summary(
    configs: str = typer.Option(
        ..., help="Comma-separated configs to summarize (e.g. 'A,B,C,GOLD')."
    ),
    run_id: str = typer.Option(
        ..., help="Label for this summary (becomes a subdir under aggregates)."
    ),
    out: Path | None = typer.Option(  # noqa: B008
        None, help="Input dir for per-config parquets (default: mauve_scores_root)."
    ),
    groupby: str = typer.Option(
        "",
        help="Comma-separated segment columns (only 'platform' is meaningful here).",
    ),
    config_path: Path = typer.Option(  # noqa: B008
        Path("configs/eval.yaml"), help="Eval config YAML."
    ),
    force: bool = typer.Option(False, help="Overwrite an existing aggregates dir for this run_id."),
) -> None:
    """Aggregate MAUVE scores into per-config (optionally per-platform) summary.

    Reads ``{out}/{config}.parquet`` (default: ``mauve_scores/``) and writes
    ``runs/{run_id}/aggregates/mauve_scores_summary.parquet`` plus optional
    ``mauve_scores_by_{segment}.parquet``.
    """
    setup_logging(level="INFO")
    cfg = _load_cfg(config_path)
    config_names = [c.strip() for c in configs.split(",") if c.strip()]
    if not config_names:
        console.print("[red]--configs is empty.[/red]")
        raise typer.Exit(code=1)

    in_dir = out if out is not None else cfg.paths.mauve_scores_root
    try:
        cfg.paths.assert_aggregates_free(run_id, force=force)
    except (FileExistsError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    out_dir = cfg.paths.aggregates_dir(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    groupby_cols: list[str] = [c.strip() for c in groupby.split(",") if c.strip()]
    for col in groupby_cols:
        if col != "platform":
            console.print(
                f"[red]Unknown groupby column {col!r}; only 'platform' is "
                "meaningful for MAUVE rows.[/red]"
            )
            raise typer.Exit(code=1)

    summary = mauve_summarize(out_dir=in_dir, configs=config_names, by=None)
    summary_path = out_dir / "mauve_scores_summary.parquet"
    summary.write_parquet(summary_path)
    console.print(f"[green]Wrote {summary_path}[/green]")
    _print_mauve_summary(summary)

    for col in groupby_cols:
        seg = mauve_summarize(out_dir=in_dir, configs=config_names, by=[col])
        seg_path = out_dir / f"mauve_scores_by_{col}.parquet"
        seg.write_parquet(seg_path)
        console.print(f"[green]Wrote {seg_path}[/green]")


def _device_to_id(device: str) -> int:
    """Map 'auto' / 'cpu' / 'cuda[:N]' to MAUVE's ``device_id`` int convention."""
    s = device.strip().lower()
    if s in ("cpu", "-1"):
        return -1
    if s in ("auto", "cuda"):
        try:
            import torch  # type: ignore[import-not-found]

            return 0 if torch.cuda.is_available() else -1
        except ImportError:
            return -1
    if s.startswith("cuda:"):
        try:
            return int(s.split(":", 1)[1])
        except (ValueError, IndexError) as exc:
            logger.warning(
                "Could not parse device string %r: %s. Falling back to CPU (-1).",
                device,
                exc,
            )
            return -1
    if s.isdigit():
        return int(s)
    logger.warning(
        "Unrecognized device string %r. Expected 'auto' / 'cpu' / 'cuda' / 'cuda:N'. "
        "Falling back to CPU (-1).",
        device,
    )
    return -1


def _print_mauve_summary(df: pl.DataFrame) -> None:
    """Render the per-config MAUVE summary table."""
    if df.is_empty():
        console.print("[yellow](no rows in summary)[/yellow]")
        return
    table = Table(title="MAUVE summary")
    for col in (
        "config",
        "n",
        "mauve_mean",
        "mauve_median",
        "mauve_min",
        "mauve_max",
        "n_gen_total",
    ):
        table.add_column(col)
    for row in df.iter_rows(named=True):
        cells = [str(row["config"]), str(row["n"])]
        for stat in ("mauve_mean", "mauve_median", "mauve_min", "mauve_max"):
            val = row.get(stat)
            cells.append("—" if val is None else f"{val:.4f}")
        cells.append(str(row.get("n_gen_total", "—")))
        table.add_row(*cells)
    console.print(table)


# ---- reference-metrics (BLEU / chrF / ROUGE-L / METEOR / BERTScore) -------


def _infer_gold_config(config_names: list[str], test_split_dir: Path) -> str:
    """Resolve the GOLD config name for the active split.

    Prefers a ``GOLD*`` entry in ``--configs``; otherwise returns ``"GOLD"``.
    """
    for name in config_names:
        if is_gold(name):
            return name
    return "GOLD"


@app.command("reference-metrics")
def reference_metrics_cmd(
    configs: str = typer.Option(..., help="Comma-separated configs (e.g. 'A,B,C')."),
    out: Path | None = typer.Option(  # noqa: B008
        None, help="Output dir for per-config parquets (default: cfg.paths.reference_scores_root)."
    ),
    config_path: Path = typer.Option(  # noqa: B008
        Path("configs/eval.yaml"), help="Eval config YAML."
    ),
    reference_tier: str | None = typer.Option(
        None, help="Override cfg.reference_metrics.reference_tier (e.g. 'high')."
    ),
    k_multi: int | None = typer.Option(
        None, help="Override cfg.reference_metrics.k_multi (nearest refs per brief)."
    ),
    bertscore: bool = typer.Option(
        True, "--bertscore/--no-bertscore", help="Compute BERTScore (pulls roberta-large)."
    ),
    v3_parquet: Path | None = typer.Option(  # noqa: B008
        None, help="Override cfg.reference_metrics.v3_parquet."
    ),
    refresh_reference: bool = typer.Option(
        False, help="Force-rebuild the reference cache under data/eval/mauve_ref/."
    ),
) -> None:
    """Score configs by overlap vs the GOLD ad + a nearest-neighbor multi-ref pool.

    Writes one Parquet per config under ``reference_scores/`` with one row per
    (config, example_id). Generations come from ``inferences_clean/<config>/``
    (raw fallback); the GOLD reference is the cleaned winning ad. Reuses the
    MAUVE v3 reference pool + contamination filter + on-disk cache.
    """
    setup_logging(level="INFO")
    cfg = _load_cfg(config_path)
    config_names = [c.strip() for c in configs.split(",") if c.strip()]
    if not config_names:
        console.print("[red]--configs is empty.[/red]")
        raise typer.Exit(code=1)

    rm = cfg.reference_metrics
    ref_tier = reference_tier if reference_tier is not None else rm.reference_tier
    k = k_multi if k_multi is not None else rm.k_multi
    v3_pq = v3_parquet if v3_parquet is not None else rm.v3_parquet
    out_dir = out if out is not None else cfg.paths.reference_scores_root

    briefs = load_test_briefs(cfg.test_split_dir)
    briefs_by_id = {b.example_id: b for b in briefs}
    if not briefs_by_id:
        console.print(f"[red]No briefs loaded from {cfg.test_split_dir}[/red]")
        raise typer.Exit(code=1)

    from draper.evaluation.mauve_reference import (
        ContaminationError,
        load_reference_corpus,
    )
    from draper.evaluation.reference_metrics import score_configs as reference_score_configs

    # GOLD copy is both the per-brief reference and the held-out contamination
    # filter. Resolve the GOLD config name from the split (not hardcoded).
    gold_cfg = _infer_gold_config(config_names, cfg.test_split_dir)
    gold_texts_by_id: dict[str, str] = {}
    held_out_texts: list[str] = []
    for brief in briefs:
        rec = load_clean(cfg.inferences_clean_dir, gold_cfg, brief.example_id)
        clean = rec.assistant_text_clean if rec is not None else ""
        if clean and clean != EXTRACTION_FAILED:
            txt = clean
        elif brief.reference_assistant:
            txt = brief.reference_assistant
        else:
            txt = ""
        if txt:
            gold_texts_by_id[brief.example_id] = txt
            held_out_texts.append(txt)
    if not gold_texts_by_id:
        console.print(
            f"[red]No GOLD references found (looked under inferences_clean/{gold_cfg}/ and "
            "Brief.reference_assistant). Run `normalize` for the GOLD config first.[/red]"
        )
        raise typer.Exit(code=1)

    console.print(
        f"[green]Loading reference corpus (tier={ref_tier}, v3={v3_pq}); "
        f"GOLD config={gold_cfg}, {len(gold_texts_by_id)} references…[/green]"
    )
    try:
        reference = load_reference_corpus(
            parquet_path=v3_pq,
            tier=ref_tier,
            platforms=list(rm.platforms),
            held_out_texts=held_out_texts,
            cache_dir=cfg.paths.mauve_ref_root,
            force_rebuild=refresh_reference,
        )
    except ContaminationError as exc:
        console.print(f"[red]Contamination check failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]Scoring {len(config_names)} configs "
        f"(k_multi={k}, bertscore={bertscore})…[/green]"
    )
    written = reference_score_configs(
        briefs_by_id=briefs_by_id,
        reference_corpus_by_platform=reference,
        gold_texts_by_id=gold_texts_by_id,
        configs=config_names,
        inferences_clean_dir=cfg.inferences_clean_dir,
        inferences_raw_dir=cfg.inferences_dir,
        out_dir=out_dir,
        platforms=list(rm.platforms),
        k_multi=k,
        enable_bertscore=bertscore,
        bertscore_model=rm.bertscore_model,
        seed=rm.random_seed,
    )
    if not written:
        console.print("[red]No configs produced any reference-metric rows.[/red]")
        raise typer.Exit(code=1)

    summary = reference_summarize(out_dir=out_dir, configs=list(written), by=None)
    _print_reference_summary(summary)


@app.command("reference-summary")
def reference_summary(
    configs: str = typer.Option(
        ..., help="Comma-separated configs to summarize (e.g. 'A,B,C,GOLD')."
    ),
    run_id: str = typer.Option(
        ..., help="Label for this summary (becomes a subdir under aggregates)."
    ),
    out: Path | None = typer.Option(  # noqa: B008
        None, help="Input dir for per-config parquets (default: reference_scores_root)."
    ),
    groupby: str = typer.Option(
        "",
        help="Comma-separated segment columns (only 'platform' is meaningful here).",
    ),
    config_path: Path = typer.Option(  # noqa: B008
        Path("configs/eval.yaml"), help="Eval config YAML."
    ),
    force: bool = typer.Option(False, help="Overwrite an existing aggregates dir for this run_id."),
) -> None:
    """Aggregate reference-metric scores into per-config (optionally per-platform) means.

    Reads ``{out}/{config}.parquet`` (default: ``reference_scores/``) and writes
    ``runs/{run_id}/aggregates/reference_scores_summary.parquet`` plus optional
    ``reference_scores_by_{segment}.parquet``.
    """
    setup_logging(level="INFO")
    cfg = _load_cfg(config_path)
    config_names = [c.strip() for c in configs.split(",") if c.strip()]
    if not config_names:
        console.print("[red]--configs is empty.[/red]")
        raise typer.Exit(code=1)

    in_dir = out if out is not None else cfg.paths.reference_scores_root
    try:
        cfg.paths.assert_aggregates_free(run_id, force=force)
    except (FileExistsError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    out_dir = cfg.paths.aggregates_dir(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    groupby_cols: list[str] = [c.strip() for c in groupby.split(",") if c.strip()]
    for col in groupby_cols:
        if col != "platform":
            console.print(
                f"[red]Unknown groupby column {col!r}; only 'platform' is "
                "meaningful for reference-metric rows.[/red]"
            )
            raise typer.Exit(code=1)

    summary = reference_summarize(out_dir=in_dir, configs=config_names, by=None)
    summary_path = out_dir / "reference_scores_summary.parquet"
    summary.write_parquet(summary_path)
    console.print(f"[green]Wrote {summary_path}[/green]")
    _print_reference_summary(summary)

    for col in groupby_cols:
        seg = reference_summarize(out_dir=in_dir, configs=config_names, by=[col])
        seg_path = out_dir / f"reference_scores_by_{col}.parquet"
        seg.write_parquet(seg_path)
        console.print(f"[green]Wrote {seg_path}[/green]")


@app.command("reference-validate")
def reference_validate(
    upworthy_path: Path = typer.Option(  # noqa: B008
        Path("data/validation/upworthy/confirmatory.csv"),
        help="Upworthy CSV (confirmatory or exploratory).",
    ),
    only_significant: bool = typer.Option(
        True, help="Restrict to A/B tests with chi-squared significant winners."
    ),
    limit: int | None = typer.Option(None, help="Limit to first N pairs (for speed)."),
    metrics: str = typer.Option(
        "bleu,chrf,rouge_l,meteor", help="Comma-separated metrics to validate."
    ),
    bertscore: bool = typer.Option(
        False, "--bertscore/--no-bertscore", help="Include BERTScore (slow — pair with --limit)."
    ),
    out: Path | None = typer.Option(  # noqa: B008
        None, help="Output dir for per-metric JSON (default: cfg.paths.validation_root)."
    ),
    config_path: Path = typer.Option(  # noqa: B008
        Path("configs/eval.yaml"), help="Eval config YAML."
    ),
) -> None:
    """Ground the reference metrics: do they predict real Upworthy A/B winners?

    Scores each variant by similarity to a held-out pool of *other* tests'
    winners (leave-one-pair-out) and reports per-metric Precision@1 vs the real
    CTR winner — the same evidentiary bar the LLM-judge arm is held to. A CI
    excluding 0.5 means "closer to known winners" carries signal.
    """
    setup_logging(level="INFO")
    cfg = _load_cfg(config_path)
    if not upworthy_path.exists():
        console.print(f"[red]Upworthy CSV not found: {upworthy_path}[/red]")
        raise typer.Exit(code=1)

    loader = UpworthyLoader()
    tests = loader.load(upworthy_path)
    pairs = loader.to_pairs(tests, only_significant=only_significant)
    if not pairs:
        console.print("[red]No pairs after filtering — try --only-significant=false.[/red]")
        raise typer.Exit(code=1)
    if limit is not None:
        pairs = pairs[:limit]
    metric_list = [m.strip() for m in metrics.split(",") if m.strip()]
    console.print(
        f"[green]Validating {len(metric_list)} reference metrics on {len(pairs)} "
        f"Upworthy pairs (bertscore={bertscore})…[/green]"
    )

    from draper.evaluation.reference_metrics import validate_on_upworthy

    results = validate_on_upworthy(
        pairs=pairs,
        metrics=metric_list,
        enable_bertscore=bertscore,
        bertscore_model=cfg.reference_metrics.bertscore_model,
    )

    out_dir = out if out is not None else cfg.paths.validation_root
    out_dir.mkdir(parents=True, exist_ok=True)

    table = Table(title="Reference-metric validation on Upworthy A/B winners")
    for col in ("metric", "n_pairs", "n_ties", "accuracy", "ci_95", "p_value"):
        table.add_column(col)
    for metric, res in results.items():
        s = res.summary()
        out_path = (
            cfg.paths.validation_path(f"refmetrics_{metric}", "upworthy")
            if out is None
            else (out / f"refmetrics_{metric}_upworthy.json")
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(s, f, indent=2, sort_keys=True)
        table.add_row(
            metric,
            str(s["n_pairs"]),
            str(s["n_ties"]),
            f"{s['accuracy']:.3f}",
            f"[{s['accuracy_ci'][0]:.3f}, {s['accuracy_ci'][1]:.3f}]",
            f"{s['binomial_p_value']:.3g}",
        )
    console.print(table)


def _print_reference_summary(df: pl.DataFrame) -> None:
    """Render a compact per-config reference-metrics summary table."""
    if df.is_empty():
        console.print("[yellow](no rows in summary)[/yellow]")
        return
    # Show the gold + multi means for each metric that has a mean column.
    headline_cols = [
        "rouge_l_gold_mean",
        "rouge_l_multi_mean",
        "bleu_gold_mean",
        "meteor_gold_mean",
        "chrf_gold_mean",
        "bertscore_gold_mean",
        "gold_overlap_excess_mean",
    ]
    present = [c for c in headline_cols if c in df.columns]
    table = Table(title="Reference-metrics summary (means)")
    table.add_column("config")
    table.add_column("n")
    for c in present:
        table.add_column(c.removesuffix("_mean"))
    for row in df.iter_rows(named=True):
        cells = [str(row["config"]), str(row["n"])]
        for c in present:
            val = row.get(c)
            cells.append("—" if val is None else f"{val:.4f}")
        table.add_row(*cells)
    console.print(table)


# ---- judge-batch (submit / status / collect) -----------------------------


@batch_app.command("submit")
def batch_submit(
    pair: str = typer.Option(..., help="Pair like 'C,GOLD' (single pair per submit)."),
    judge: str = typer.Option(..., help="Judge model. Routed to provider by prefix."),
    arm: str = typer.Option("arm1", help="'arm1' or 'arm2'."),
    run_id: str = typer.Option(..., help="Label for this batch (becomes a subdir)."),
    config_path: Path = typer.Option(  # noqa: B008
        Path("configs/eval.yaml"), help="Eval config YAML."
    ),
    swap: bool = typer.Option(True, help="Include swapped ordering (position bias)."),
    limit: int | None = typer.Option(None, help="Limit to first N examples."),
) -> None:
    """Submit a batch judge run. Provider chosen by judge model prefix.

    OpenAI: gpt-*, o*, etc. Anthropic: claude-*. Gemini is not supported
    (no flat batch discount); use the live judge path instead.

    Writes manifest.json + (for OpenAI) input.jsonl + batch_id.txt
    under data/eval/runs/<run_id>/batches/<judge>/<pair>/.
    """
    setup_logging(level="INFO")
    cfg = _load_cfg(config_path)
    parts = [p.strip() for p in pair.split(",")]
    if len(parts) != 2:
        console.print(f"[red]Bad --pair {pair!r}; expected 'A,B'[/red]")
        raise typer.Exit(code=1)
    pair_t: tuple[str, str] = (parts[0], parts[1])

    provider = provider_for_model(judge)

    briefs_by_id: dict[str, Brief] = (
        {b.example_id: b for b in load_test_briefs(cfg.test_split_dir)} if arm == "arm1" else {}
    )
    scenarios_by_id: dict[str, UrlScenario] = (
        {s.scenario_id: s for s in load_url_scenarios(cfg.url_scenarios_path)}
        if arm == "arm2"
        else {}
    )

    def _load_side(name: str) -> dict[str, Inference]:
        if is_gold(name):
            if arm != "arm1":
                console.print("[red]GOLD requires arm='arm1'.[/red]")
                raise typer.Exit(code=1)
            return gold_inferences_from_briefs(briefs_by_id.values(), name)
        return load_inferences_for_config(cfg.inferences_dir, name)

    infs_a = _load_side(pair_t[0])
    infs_b = _load_side(pair_t[1])
    if limit is not None:
        common = sorted(set(infs_a) & set(infs_b))[:limit]
        infs_a = {k: infs_a[k] for k in common}
        infs_b = {k: infs_b[k] for k in common}

    out_dir = cfg.paths.batches_dir(run_id, judge, pair_t)
    out_dir.mkdir(parents=True, exist_ok=True)

    if provider == "openai":
        lines, manifest = build_openai_batch_jsonl(
            pair=pair_t,
            judge_model=judge,
            inferences_a=infs_a,
            inferences_b=infs_b,
            briefs_by_id=briefs_by_id or None,
            scenarios_by_id=scenarios_by_id or None,
            swap=swap,
            clean_root=cfg.inferences_clean_dir,
        )
        jsonl_path = out_dir / "input.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        batch_id = submit_openai_batch(
            jsonl_path=jsonl_path, description=f"{run_id}/{judge}/{pair}"
        )
    else:
        requests, manifest = build_anthropic_batch_requests(
            pair=pair_t,
            judge_model=judge,
            inferences_a=infs_a,
            inferences_b=infs_b,
            briefs_by_id=briefs_by_id or None,
            scenarios_by_id=scenarios_by_id or None,
            swap=swap,
            clean_root=cfg.inferences_clean_dir,
        )
        batch_id = submit_anthropic_batch(requests=requests)

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest.to_json(), f, indent=2)
    (out_dir / "batch_id.txt").write_text(batch_id)
    console.print(
        f"[green]Submitted {provider} batch {batch_id} "
        f"({len(manifest.keys)} requests) → {out_dir}[/green]"
    )


@batch_app.command("status")
def batch_status(
    run_id: str = typer.Option(
        ..., help="Batch run-id (resolves to data/eval/runs/<run_id>/batches/)."
    ),
    pair: str = typer.Option(..., help="Pair like 'C,GOLD'."),
    judge: str = typer.Option(..., help="Judge model."),
    config_path: Path = typer.Option(  # noqa: B008
        Path("configs/eval.yaml"), help="Eval config YAML."
    ),
) -> None:
    """Poll batch status. Won't crash before completion — just prints state."""
    setup_logging(level="INFO")
    cfg = _load_cfg(config_path)
    parts = [p.strip() for p in pair.split(",")]
    out_dir = cfg.paths.batches_dir(run_id, judge, (parts[0], parts[1]))

    batch_id_path = out_dir / "batch_id.txt"
    if not batch_id_path.exists():
        console.print(
            f"[red]No batch_id.txt found at {batch_id_path}\n"
            "[yellow]Did you run: "
            "python scripts/eval.py judge-batch submit --run-id {run_id} "
            "--judge {judge} --pair {pair}[/yellow]"
        )
        raise typer.Exit(code=1)

    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        console.print(
            f"[red]No manifest.json found at {manifest_path}\n"
            "[yellow]Batch directory may be corrupted; re-run submit.[/yellow]"
        )
        raise typer.Exit(code=1)

    try:
        batch_id = batch_id_path.read_text().strip()
        manifest = BatchManifest.from_json(json.loads(manifest_path.read_text()))
    except (ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]Failed to parse batch metadata: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    if manifest.provider == "openai":
        st = status_openai_batch(batch_id)
    else:
        st = status_anthropic_batch(batch_id)
    console.print_json(data=st)


@batch_app.command("collect")
def batch_collect(
    run_id: str = typer.Option(..., help="Batch run-id."),
    pair: str = typer.Option(..., help="Pair like 'C,GOLD'."),
    judge: str = typer.Option(..., help="Judge model."),
    config_path: Path = typer.Option(  # noqa: B008
        Path("configs/eval.yaml"), help="Eval config YAML."
    ),
) -> None:
    """Pull a completed batch and write per-example judgment JSONs.

    Output shape matches the live judge path: ``data/eval/judgments/
    {judge}/{pair}/{example_id}.json`` containing a list of Judgment
    objects (forward + swapped orderings). The aggregate command reads
    these without knowing whether they came from batch or live.
    """
    setup_logging(level="INFO")
    cfg = _load_cfg(config_path)
    parts = [p.strip() for p in pair.split(",")]
    pair_t: tuple[str, str] = (parts[0], parts[1])
    out_dir = cfg.paths.batches_dir(run_id, judge, pair_t)

    # Validate that submit was run first
    batch_id_path = out_dir / "batch_id.txt"
    manifest_path = out_dir / "manifest.json"
    if not batch_id_path.exists() or not manifest_path.exists():
        console.print(
            f"[red]Batch metadata missing from {out_dir}\n"
            "[yellow]Run: python scripts/eval.py judge-batch submit ... first[/yellow]"
        )
        raise typer.Exit(code=1)

    try:
        batch_id = batch_id_path.read_text().strip()
        manifest = BatchManifest.from_json(json.loads(manifest_path.read_text()))
    except (ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]Failed to parse batch metadata: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    if manifest.provider == "openai":
        parsed = collect_openai_batch(batch_id)
    else:
        parsed = collect_anthropic_batch(batch_id)

    judgments, missing = parsed_to_judgments(parsed=parsed, manifest=manifest)
    if not judgments:
        console.print("[yellow]No judgments parsed — check error file in provider UI.[/yellow]")
        if missing:
            console.print(
                f"[yellow]All {len(missing)} requests failed or were not returned.[/yellow]"
            )
        return

    by_example: dict[str, list[Any]] = {}
    for j in judgments:
        by_example.setdefault(j.example_id, []).append(j)

    judgments_root = cfg.judgments_dir / judge / f"{pair_t[0]}_vs_{pair_t[1]}"
    judgments_root.mkdir(parents=True, exist_ok=True)
    written = 0
    for example_id, js in by_example.items():
        target = judgments_root / f"{example_id}.json"
        with target.open("w", encoding="utf-8") as f:
            json.dump([j.model_dump(mode="json") for j in js], f, indent=2, default=str)
        written += 1

    msg = f"[green]Wrote {written} judgment files to {judgments_root} "
    msg += f"(from {len(judgments)} judgments / {len(manifest.keys)} requests)"
    if missing:
        msg += f" — {len(missing)} missing[/green]"
        if len(missing) <= 10:
            msg_detail = ", ".join(missing)
            console.print(f"[yellow]Missing: {msg_detail}[/yellow]")
    else:
        msg += "[/green]"
    console.print(msg)


def _emit_similarity_diagnostics(
    *,
    cfg: EvalConfig,
    pairs: list[tuple[str, str]],
    briefs_by_id: dict[str, Any],
    out_dir: Path,
) -> Path | None:
    """Emit a per-example similarity-to-gold parquet for GOLD pairs only.

    Looks at every (X, GOLD) pair in ``pairs`` and pulls each model's
    inference text (from disk) plus the gold text (from briefs). Computes
    rouge_l_f1 + cosine_to_gold per (config, example_id). Writes a single
    long-format parquet with columns ``[config, example_id, platform,
    vertical, source_tier, rouge_l_f1, cosine_to_gold]``.

    Returns the output path, or None if no GOLD pairs were configured.
    """
    gold_configs: list[str] = []
    for pair in pairs:
        if is_gold(pair[0]) and not is_gold(pair[1]):
            gold_configs.append(pair[1])
        elif is_gold(pair[1]) and not is_gold(pair[0]):
            gold_configs.append(pair[0])
    gold_configs = sorted(set(gold_configs))
    if not gold_configs:
        return None

    rows: list[dict[str, Any]] = []
    for cfg_name in gold_configs:
        infs = load_inferences_for_config(cfg.inferences_dir, cfg_name)
        for example_id, inf in infs.items():
            brief = briefs_by_id.get(example_id)
            if brief is None:
                continue
            sim = similarity_to_gold(inf.assistant_text, brief.reference_assistant)
            rows.append(
                {
                    "config": cfg_name,
                    "example_id": example_id,
                    "platform": brief.platform,
                    "vertical": brief.vertical,
                    "source_tier": (brief.source_tiers[0] if brief.source_tiers else None),
                    "rouge_l_f1": sim["rouge_l_f1"],
                    "cosine_to_gold": sim["cosine_to_gold"],
                }
            )
    if not rows:
        return None
    sim_path = out_dir / "similarity_to_gold.parquet"
    pl.DataFrame(rows).write_parquet(sim_path)
    return sim_path


def _print_summary(df: pl.DataFrame) -> None:
    if df.is_empty():
        console.print("[yellow](no rows in summary)[/yellow]")
        return
    table = Table(title="Win rates")
    for col in (
        "judge_model",
        "config_a",
        "config_b",
        "n",
        "wins_a",
        "wins_b",
        "ties",
        "order_dep",
        "win_rate_a",
        "tie_rate",
    ):
        table.add_column(col)
    for row in df.iter_rows(named=True):
        table.add_row(
            str(row["judge_model"]),
            str(row["config_a"]),
            str(row["config_b"]),
            str(row["n"]),
            str(row["wins_a"]),
            str(row["wins_b"]),
            str(row["ties"]),
            str(row["order_dep"]),
            f"{row['win_rate_a']:.3f}",
            f"{row['tie_rate']:.3f}",
        )
    console.print(table)


# ---- report --------------------------------------------------------------


@app.command("validate")
def validate_judge(
    judge: list[str] = typer.Option(  # noqa: B008
        ..., "--judge", help="Judge model to validate (repeatable)."
    ),
    stream: str = typer.Option(
        "upworthy",
        help="Ground-truth stream. Currently only 'upworthy' is wired.",
    ),
    upworthy_path: Path = typer.Option(  # noqa: B008
        Path("data/validation/upworthy/confirmatory.csv"),
        help="Upworthy CSV (confirmatory or exploratory).",
    ),
    only_significant: bool = typer.Option(
        True,
        help="Restrict to A/B tests with chi-squared significant winners.",
    ),
    limit: int | None = typer.Option(
        None, help="Limit to first N pairs (for fast smoke validation)."
    ),
    concurrency: int = typer.Option(8, help="Max in-flight judge calls."),
    out: Path | None = typer.Option(  # noqa: B008
        None,
        help=(
            "Output dir for per-judge validation summary JSON (default: cfg.paths.validation_root)."
        ),
    ),
    config_path: Path = typer.Option(  # noqa: B008
        Path("configs/eval.yaml"), help="Eval config YAML."
    ),
) -> None:
    """Validate judge methodology against an external A/B-test ground truth.

    Asks each judge model to predict the winner of an Upworthy A/B test
    (winner = highest CTR, optionally chi-squared significant) and reports
    accuracy + binomial p-value + Wilson 95% CI.

    Use this BEFORE interpreting model-comparison verdicts. A judge that
    can't beat ~chance on real A/B winners is not a trustworthy signal.
    """
    setup_logging(level="INFO")
    cfg = _load_cfg(config_path)
    if stream != "upworthy":
        console.print(f"[red]Stream {stream!r} not supported (only 'upworthy').[/red]")
        raise typer.Exit(code=1)
    if not upworthy_path.exists():
        console.print(f"[red]Upworthy CSV not found: {upworthy_path}[/red]")
        raise typer.Exit(code=1)

    loader = UpworthyLoader()
    tests = loader.load(upworthy_path)
    pairs = loader.to_pairs(tests, only_significant=only_significant)
    if not pairs:
        console.print("[red]No pairs after filtering — try --only-significant=false.[/red]")
        raise typer.Exit(code=1)
    if limit is not None:
        pairs = pairs[:limit]
    console.print(f"[green]Validating {len(judge)} judges on {len(pairs)} Upworthy pairs[/green]")

    out_dir = out if out is not None else cfg.paths.validation_root
    out_dir.mkdir(parents=True, exist_ok=True)

    async def _go() -> dict[str, dict[str, Any]]:
        summaries: dict[str, dict[str, Any]] = {}
        for jm in judge:
            result = await validate_judge_on_upworthy_pairs(
                judge_model=jm,
                pairs=pairs,
                max_concurrency=concurrency,
                source=stream,
            )
            summary = result.summary()
            summaries[jm] = summary
            out_path = (
                cfg.paths.validation_path(stream, jm)
                if out is None
                else (out / f"{stream}_{jm.replace('/', '_')}.json")
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, sort_keys=True)
            console.print(f"[green]Wrote {out_path}[/green]")
        return summaries

    summaries = asyncio.run(_go())

    table = Table(title=f"Judge validation on {stream}")
    table.add_column("judge")
    table.add_column("n_pairs")
    table.add_column("accuracy")
    table.add_column("ci_95")
    table.add_column("p_value")
    table.add_column("ord_dep")
    for jm, s in summaries.items():
        table.add_row(
            str(jm),
            str(s["n_pairs"]),
            f"{s['accuracy']:.3f}",
            f"[{s['accuracy_ci'][0]:.3f}, {s['accuracy_ci'][1]:.3f}]",
            f"{s['binomial_p_value']:.3g}",
            str(s["n_order_dependent"]),
        )
    console.print(table)


@app.command("report")
def report(
    run_id: str = typer.Option(..., help="Aggregate run-id to report on."),
    config_path: Path = typer.Option(  # noqa: B008
        Path("configs/eval.yaml"), help="Eval config YAML."
    ),
) -> None:
    """Pretty-print an existing aggregate run."""
    setup_logging(level="INFO")
    cfg = _load_cfg(config_path)
    aggregates_dir = cfg.paths.aggregates_dir(run_id)
    summary_path = aggregates_dir / "summary.parquet"
    if not summary_path.exists():
        console.print(f"[red]No summary at {summary_path}[/red]")
        raise typer.Exit(code=1)
    df = pl.read_parquet(summary_path)
    _print_summary(df)
    elo_path = aggregates_dir / "elo.json"
    if elo_path.exists():
        with elo_path.open() as f:
            console.print_json(data=json.load(f))


# ---- compare (two run_ids side-by-side) ----------------------------------


def _read_optional_parquet(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pl.read_parquet(path)
    except Exception:  # noqa: BLE001 — defensive read, parquet may be partial
        return None


def _compare_win_rates(base: pl.DataFrame, cand: pl.DataFrame) -> pl.DataFrame:
    """Inner-join two summary.parquets on (config_a, config_b, judge_model)."""
    key = ["config_a", "config_b", "judge_model"]
    return (
        base.select(
            *key,
            pl.col("n").alias("n_base"),
            pl.col("win_rate_a").alias("win_rate_a_base"),
            pl.col("tie_rate").alias("tie_rate_base"),
        )
        .join(
            cand.select(
                *key,
                pl.col("n").alias("n_cand"),
                pl.col("win_rate_a").alias("win_rate_a_cand"),
                pl.col("tie_rate").alias("tie_rate_cand"),
            ),
            on=key,
            how="inner",
        )
        .with_columns(
            (pl.col("win_rate_a_cand") - pl.col("win_rate_a_base")).alias("delta_win_rate_a"),
            (pl.col("tie_rate_cand") - pl.col("tie_rate_base")).alias("delta_tie_rate"),
        )
        .sort(key)
    )


def _compare_learned(base: pl.DataFrame, cand: pl.DataFrame) -> pl.DataFrame:
    """Join two learned_scores_summary.parquets on ``config``."""
    base_cols = [c for c in base.columns if c == "config" or c.endswith("_mean")]
    cand_cols = [c for c in cand.columns if c == "config" or c.endswith("_mean")]
    base_renamed = base.select(base_cols).rename(
        {c: f"{c}_base" if c != "config" else c for c in base_cols}
    )
    cand_renamed = cand.select(cand_cols).rename(
        {c: f"{c}_cand" if c != "config" else c for c in cand_cols}
    )
    joined = base_renamed.join(cand_renamed, on="config", how="inner")
    head_means = [c for c in base_cols if c.endswith("_mean")]
    deltas = [(pl.col(f"{h}_cand") - pl.col(f"{h}_base")).alias(f"delta_{h}") for h in head_means]
    return joined.with_columns(deltas).sort("config")


def _compare_elo(base: dict[str, Any], cand: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Per-judge per-config Elo delta."""
    out: dict[str, dict[str, float]] = {}
    for judge in sorted(set(base) | set(cand)):
        b = base.get(judge, {})
        c = cand.get(judge, {})
        delta: dict[str, float] = {}
        for cfg_name in sorted(set(b) | set(c)):
            bv = b.get(cfg_name)
            cv = c.get(cfg_name)
            if bv is None or cv is None:
                continue
            delta[cfg_name] = float(cv) - float(bv)
        out[judge] = delta
    return out


def _render_compare_markdown(
    *,
    base_id: str,
    candidate_id: str,
    win_rate_df: pl.DataFrame,
    learned_df: pl.DataFrame | None,
    elo_delta: dict[str, dict[str, float]],
) -> str:
    lines: list[str] = []
    lines.append(f"# Compare: `{candidate_id}` vs `{base_id}`\n")
    lines.append("Positive delta = candidate beats base.\n")

    lines.append("## Win-rate delta (config_a vs config_b, per judge)\n")
    if win_rate_df.is_empty():
        lines.append("_No overlapping (pair × judge) rows between the two runs._\n")
    else:
        lines.append("| judge | a | b | n_base | n_cand | wr_a_base | wr_a_cand | Δ wr_a | Δ tie |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in win_rate_df.iter_rows(named=True):
            lines.append(
                f"| {row['judge_model']} | {row['config_a']} | {row['config_b']} | "
                f"{row['n_base']} | {row['n_cand']} | "
                f"{row['win_rate_a_base']:.3f} | {row['win_rate_a_cand']:.3f} | "
                f"{row['delta_win_rate_a']:+.3f} | {row['delta_tie_rate']:+.3f} |"
            )
        lines.append("")

    if learned_df is not None and not learned_df.is_empty():
        lines.append("## Learned-score delta (per config)\n")
        head_means = [
            c[: -len("_mean_base")] for c in learned_df.columns if c.endswith("_mean_base")
        ]
        header = ["config"] + [f"Δ {h}_mean" for h in head_means]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in learned_df.iter_rows(named=True):
            cells = [str(row["config"])]
            for h in head_means:
                v = row.get(f"delta_{h}_mean")
                cells.append("—" if v is None else f"{v:+.3f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    if elo_delta:
        lines.append("## Elo delta (per judge × config)\n")
        for judge, deltas in elo_delta.items():
            if not deltas:
                continue
            lines.append(f"### {judge}")
            lines.append("| config | Δ Elo |")
            lines.append("| --- | --- |")
            for cfg_name, dv in sorted(deltas.items()):
                lines.append(f"| {cfg_name} | {dv:+.1f} |")
            lines.append("")
    return "\n".join(lines) + "\n"


@app.command("compare")
def compare(
    base: str = typer.Option(..., "--base", help="Baseline run_id."),
    candidate: str = typer.Option(..., "--candidate", help="Candidate run_id."),
    config_path: Path = typer.Option(  # noqa: B008
        Path("configs/eval.yaml"), help="Eval config YAML."
    ),
) -> None:
    """Diff two aggregate runs into a side-by-side parquet + markdown report.

    Reads ``runs/<base>/aggregates/`` and ``runs/<candidate>/aggregates/``,
    joins on (config_a, config_b, judge_model), and writes:

      - ``runs/<candidate>/aggregates/compare_vs_<base>.parquet`` — paired
        rows with win-rate delta + tie-rate delta per (pair × judge).
      - ``runs/<candidate>/aggregates/compare_vs_<base>.md`` — terminal-
        friendly markdown with win-rate Δ, learned-score Δ, and Elo Δ.

    Use this to quickly check whether a new agent-architecture variant
    (or any candidate run) outperforms the canonical baseline. Run a
    fresh ``aggregate``/``score-summary`` for both ``base`` and
    ``candidate`` first.
    """
    setup_logging(level="INFO")
    cfg = _load_cfg(config_path)
    try:
        base_dir = cfg.paths.aggregates_dir(base)
        cand_dir = cfg.paths.aggregates_dir(candidate)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    base_summary = _read_optional_parquet(base_dir / "summary.parquet")
    cand_summary = _read_optional_parquet(cand_dir / "summary.parquet")
    win_rate_df = (
        _compare_win_rates(base_summary, cand_summary)
        if (base_summary is not None and cand_summary is not None)
        else pl.DataFrame()
    )

    base_learned = _read_optional_parquet(base_dir / "learned_scores_summary.parquet")
    cand_learned = _read_optional_parquet(cand_dir / "learned_scores_summary.parquet")
    learned_df = (
        _compare_learned(base_learned, cand_learned)
        if (base_learned is not None and cand_learned is not None)
        else None
    )

    if win_rate_df.is_empty() and learned_df is None:
        console.print(
            f"[red]Neither run has comparable artifacts.\n"
            f"  base: {base_dir}\n  cand: {cand_dir}\n"
            "Need at least one of summary.parquet or "
            "learned_scores_summary.parquet on both sides.[/red]"
        )
        raise typer.Exit(code=1)

    base_elo: dict[str, Any] = {}
    cand_elo: dict[str, Any] = {}
    base_elo_path = base_dir / "elo.json"
    cand_elo_path = cand_dir / "elo.json"
    if base_elo_path.exists():
        with base_elo_path.open() as f:
            base_elo = json.load(f)
    if cand_elo_path.exists():
        with cand_elo_path.open() as f:
            cand_elo = json.load(f)
    elo_delta = _compare_elo(base_elo, cand_elo)

    parquet_out = cand_dir / f"compare_vs_{base}.parquet"
    md_out = cand_dir / f"compare_vs_{base}.md"
    cand_dir.mkdir(parents=True, exist_ok=True)
    win_rate_df.write_parquet(parquet_out)
    md_out.write_text(
        _render_compare_markdown(
            base_id=base,
            candidate_id=candidate,
            win_rate_df=win_rate_df,
            learned_df=learned_df,
            elo_delta=elo_delta,
        )
    )
    console.print(f"[green]Wrote {parquet_out}[/green]")
    console.print(f"[green]Wrote {md_out}[/green]")

    if not win_rate_df.is_empty():
        _print_summary(
            win_rate_df.select(
                pl.col("judge_model"),
                pl.col("config_a"),
                pl.col("config_b"),
                pl.col("n_cand").alias("n"),
                pl.lit(0).alias("wins_a"),
                pl.lit(0).alias("wins_b"),
                pl.lit(0).alias("ties"),
                pl.lit(0).alias("order_dep"),
                pl.col("delta_win_rate_a").alias("win_rate_a"),
                pl.col("delta_tie_rate").alias("tie_rate"),
            )
        )


if __name__ == "__main__":
    app()
