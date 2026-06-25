"""Orchestration helpers for inference + judge runs (CLI calls into here)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import EvalConfig
from .inference import build_runner
from .inference.base import InferenceRunner
from .judge.pairwise import judge_pair, reconcile_pair
from .schemas import Brief, Inference, Judgment, PairResult, UrlScenario

# ---- Inference orchestration ---------------------------------------------


def _inference_path(root: Path, config_name: str, example_id: str) -> Path:
    return root / config_name / f"{example_id}.json"


def _load_inference(path: Path) -> Inference:
    with path.open("r", encoding="utf-8") as f:
        return Inference.model_validate(json.load(f))


def _save_inference(inf: Inference, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(inf.model_dump_json(indent=2))


async def _bounded_gather(
    coros: Sequence[asyncio.Future[Any] | Any],
    max_concurrency: int,
) -> list[Any]:
    sem = asyncio.Semaphore(max_concurrency)

    async def _wrap(coro: Any) -> Any:
        async with sem:
            return await coro

    return await asyncio.gather(*[_wrap(c) for c in coros])


async def run_inference_for_briefs(
    *,
    runner: InferenceRunner,
    briefs: list[Brief],
    out_root: Path,
    force: bool,
    max_concurrency: int,
) -> tuple[int, int, int]:
    """Run a runner across briefs, writing per-example JSONs under out_root.

    Returns ``(written, skipped, errored)``.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    todo: list[Brief] = []
    skipped = 0
    for b in briefs:
        path = _inference_path(out_root, runner.config_name, b.example_id)
        if path.exists() and not force:
            skipped += 1
            continue
        todo.append(b)

    async def _do(brief: Brief) -> Inference:
        inf = await runner.run_brief(brief)
        # Persist per-brief on completion — losing the process mid-run must
        # not lose completed inferences (we paid for them).
        path = _inference_path(out_root, runner.config_name, inf.example_id)
        _save_inference(inf, path)
        return inf

    written = 0
    errored = 0
    if todo:
        results = await _bounded_gather([_do(b) for b in todo], max_concurrency=max_concurrency)
        for inf in results:
            assert isinstance(inf, Inference)
            if inf.error:
                errored += 1
            else:
                written += 1
    return written, skipped, errored


async def run_inference_for_scenarios(
    *,
    runner: InferenceRunner,
    scenarios: list[UrlScenario],
    out_root: Path,
    force: bool,
    max_concurrency: int,
) -> tuple[int, int, int]:
    """Same shape as run_inference_for_briefs, for Arm 2 URL scenarios."""
    out_root.mkdir(parents=True, exist_ok=True)
    todo: list[UrlScenario] = []
    skipped = 0
    for s in scenarios:
        path = _inference_path(out_root, runner.config_name, s.scenario_id)
        if path.exists() and not force:
            skipped += 1
            continue
        todo.append(s)

    async def _do(scen: UrlScenario) -> Inference:
        inf = await runner.run_scenario(scen)
        # Persist per-scenario on completion — see run_inference_for_briefs.
        path = _inference_path(out_root, runner.config_name, inf.example_id)
        _save_inference(inf, path)
        return inf

    written = 0
    errored = 0
    if todo:
        results = await _bounded_gather([_do(s) for s in todo], max_concurrency=max_concurrency)
        for inf in results:
            assert isinstance(inf, Inference)
            if inf.error:
                errored += 1
            else:
                written += 1
    return written, skipped, errored


def load_inferences_for_config(root: Path, config_name: str) -> dict[str, Inference]:
    """Load all per-example inference JSONs for one config.

    Keyed by ``example_id`` for cheap lookup during pairwise judging.
    """
    out: dict[str, Inference] = {}
    config_dir = root / config_name
    if not config_dir.exists():
        return out
    for p in sorted(config_dir.glob("*.json")):
        inf = _load_inference(p)
        out[inf.example_id] = inf
    return out


# ---- Judge orchestration --------------------------------------------------


def _judgment_path(root: Path, judge_model: str, pair: tuple[str, str], example_id: str) -> Path:
    pair_dir = f"{pair[0]}_vs_{pair[1]}"
    return root / judge_model / pair_dir / f"{example_id}.json"


def _save_judgments(judgments: list[Judgment], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([j.model_dump(mode="json") for j in judgments], f, indent=2, default=str)


def _load_judgments(path: Path) -> list[Judgment]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Judgment.model_validate(j) for j in raw]


async def run_judge_pass(
    *,
    pair: tuple[str, str],
    inferences_a: dict[str, Inference],
    inferences_b: dict[str, Inference],
    briefs_by_id: dict[str, Brief] | None = None,
    scenarios_by_id: dict[str, UrlScenario] | None = None,
    judge_model: str,
    out_root: Path,
    swap: bool,
    max_concurrency: int,
    force: bool,
    clean_root: Path | None = None,
) -> tuple[int, int]:
    """Run pairwise judging across all examples both inferences agree on.

    ``clean_root`` enables the LLM ad-copy extractor cache — see ``judge_pair``.
    """
    common = sorted(set(inferences_a) & set(inferences_b))
    todo: list[str] = []
    skipped = 0
    for ex_id in common:
        path = _judgment_path(out_root, judge_model, pair, ex_id)
        if path.exists() and not force:
            skipped += 1
            continue
        todo.append(ex_id)

    async def _do(example_id: str) -> tuple[str, list[Judgment]]:
        a = inferences_a[example_id]
        b = inferences_b[example_id]
        platform = ""
        vertical = ""
        user_prompt = a.brief
        if briefs_by_id and example_id in briefs_by_id:
            br = briefs_by_id[example_id]
            platform = br.platform
            vertical = br.vertical
            user_prompt = br.user
        elif scenarios_by_id and example_id in scenarios_by_id:
            sc = scenarios_by_id[example_id]
            platform = sc.platform
            vertical = sc.vertical
            user_prompt = sc.user_prompt
        judgments = await judge_pair(
            example_id=example_id,
            platform=platform,
            vertical=vertical,
            user_prompt=user_prompt,
            a=a,
            b=b,
            judge_model=judge_model,
            swap=swap,
            clean_root=clean_root,
        )
        return example_id, judgments

    written = 0
    if todo:
        results = await _bounded_gather([_do(ex) for ex in todo], max_concurrency=max_concurrency)
        for example_id, judgments in results:
            path = _judgment_path(out_root, judge_model, pair, example_id)
            _save_judgments(judgments, path)
            written += 1
    return written, skipped


def load_pair_results(
    *,
    root: Path,
    judge_model: str,
    pair: tuple[str, str],
) -> list[PairResult]:
    """Load all judgments for a (judge, pair) and reconcile into PairResults."""
    pair_dir = root / judge_model / f"{pair[0]}_vs_{pair[1]}"
    if not pair_dir.exists():
        return []
    out: list[PairResult] = []
    for p in sorted(pair_dir.glob("*.json")):
        judgments = _load_judgments(p)
        if not judgments:
            continue
        out.append(
            reconcile_pair(
                example_id=judgments[0].example_id,
                config_a=pair[0],
                config_b=pair[1],
                judgments=judgments,
                judge_model=judge_model,
            )
        )
    return out


def runners_from_config(cfg: EvalConfig, names: list[str]) -> dict[str, InferenceRunner]:
    out: dict[str, InferenceRunner] = {}
    for name in names:
        if name not in cfg.configs:
            raise ValueError(f"Config '{name}' not in eval.yaml (got {list(cfg.configs)}).")
        out[name] = build_runner(name, cfg.configs[name])
    return out
