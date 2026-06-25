"""Side-by-side smoke harness for the new agent flow vs. single-shot baseline.

Reads from the canonical eval caches but writes only under the per-run
diagnostics tree (``data/eval/runs/<run_id>/diagnostics/agent_smoke/``)
so it never collides with the main pipeline.

For N briefs from the held-out test split, fires each through the frontend
agent endpoint (POST /api/eval/run) and pairs it against the matching cached
single-shot inference under ``data/eval/inferences/<base_config>/<id>.json``
(read-only). Both candidates are scored via the local scoring predictor
service so we can eyeball relative quality + composite delta.

Defaults are tuned for the current Qwen-writer setup:
    base_config = "B"   (single-shot qwen/qwen3-8b)
    new agent   = whatever frontend MODEL_ID is (qwen/qwen3-8b expected)

Run:
    SCORING_PREDICTOR_API_KEY=... EVAL_SERVICE_TOKEN=... \\
    uv run python scripts/diagnostics/agent_smoke.py run --n 5

Outputs (canonical layout):
    data/eval/runs/<run_id>/diagnostics/agent_smoke/pairs.jsonl
    data/eval/runs/<run_id>/diagnostics/agent_smoke/report.md
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import typer
from rich.console import Console

from draper.evaluation.briefs import load_test_briefs
from draper.evaluation.judge.normalize import EXTRACTION_FAILED, extract_ad_copy
from draper.evaluation.paths import EvalPaths, validate_config_name, validate_run_id
from draper.evaluation.schemas import Brief

app = typer.Typer(add_completion=False)
console = Console()

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_SPLIT = REPO_ROOT / "data" / "final" / "test"
_PATHS = EvalPaths(root=REPO_ROOT / "data" / "eval")
DEFAULT_INFERENCES = _PATHS.inferences_root
DEFAULT_INFERENCES_CLEAN = _PATHS.inferences_clean_root


def _resolve_run_id(run_id: str | None) -> str:
    """Validate or auto-generate a canonical run_id.

    Auto-format ``YYYY-MM-DD-agent-smoke-HHMMSS`` keeps existing usage
    (no flag passed) one click away from the new convention.
    """
    if run_id:
        return validate_run_id(run_id)
    now = datetime.now(UTC)
    return f"{now:%Y-%m-%d}-agent-smoke-{now:%H%M%S}"


def _diagnostics_out_dir(run_id: str) -> Path:
    out: Path = _PATHS.diagnostics_dir(run_id, "agent_smoke")
    return out


# ---------------------------------------------------------------------------
# Predictor client (talks to local scoring predictor on :8001)
# ---------------------------------------------------------------------------


def _platform_for_predictor(p: str | None) -> str:
    allowed = {"meta", "tiktok", "x", "google", "pinterest", "reddit", "other"}
    return p if p in allowed else "other"


async def _score_one(
    client: httpx.AsyncClient,
    *,
    predictor_url: str,
    api_key: str,
    platform: str,
    vertical: str,
    headline: str | None,
    body: str | None,
    description: str | None,
) -> dict[str, float] | None:
    payload = {
        "items": [
            {
                "platform": _platform_for_predictor(platform),
                "vertical": vertical or "unknown",
                "headline": headline,
                "body": body,
                "description": description,
            }
        ]
    }
    try:
        resp = await client.post(
            f"{predictor_url.rstrip('/')}/score",
            json=payload,
            headers={"X-API-Key": api_key},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        scores = data.get("scores") or []
        if not scores:
            return None
        s = scores[0]
        return {
            "composite": float(s["composite"]),
            "survivability": float(s["survivability"]),
            "engagement_volume": float(s["engagement_volume"]),
            "engagement_velocity": float(s["engagement_velocity"]),
        }
    except Exception as e:
        console.print(f"[yellow]score error: {type(e).__name__}: {e}[/yellow]")
        return None


# ---------------------------------------------------------------------------
# Campaign helpers
# ---------------------------------------------------------------------------


def _campaign_copy_fields(campaign: dict[str, Any]) -> dict[str, str | None]:
    """Pull (headline, body, description) out of a CampaignOutput payload.

    Mirrors `_primary_copy_from_campaign` in the eval frontend runner but
    keeps the fields *separated* so the scorer can see them in their proper
    slots (rather than mashed together).
    """
    head = campaign.get("headline") or campaign.get("hook_line") or campaign.get("tweet_body")
    body_parts: list[str] = []
    for k in ("body", "caption"):
        v = campaign.get(k)
        if isinstance(v, str) and v.strip():
            body_parts.append(v.strip())
    body = "\n\n".join(body_parts) if body_parts else None
    desc = campaign.get("cta")
    return {
        "headline": head.strip() if isinstance(head, str) and head.strip() else None,
        "body": body,
        "description": desc.strip() if isinstance(desc, str) and desc.strip() else None,
    }


def _campaign_display_text(campaign: dict[str, Any]) -> str:
    parts: list[str] = []
    for k in ("headline", "hook_line", "tweet_body", "body", "caption", "cta"):
        v = campaign.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(f"[{k}] {v.strip()}")
    return "\n".join(parts) if parts else "(no copy fields)"


def _tool_summary(traces: Iterable[dict[str, Any]] | None) -> list[str]:
    """Extract tool-call names from agent_trace.toolCalls payloads.

    Drizzle stores each row's ``toolCalls`` as a wrapper object
    ``{_prompts: {system, user}, _toolCalls: [...]}`` (see
    ``scripts/diagnostics/inspect_traces.py::TraceRow.input_artifact``). We
    unwrap to the inner array and pull out ``toolName``.
    """
    if not traces:
        return []
    out: list[str] = []
    for t in traces:
        tc = t.get("toolCalls")
        inner = tc.get("_toolCalls") if isinstance(tc, dict) else tc
        if not isinstance(inner, list):
            continue
        for entry in inner:
            if isinstance(entry, dict):
                name = entry.get("toolName") or entry.get("name")
                if name:
                    out.append(str(name))
    return out


# ---------------------------------------------------------------------------
# Pair execution
# ---------------------------------------------------------------------------


async def _fire_agent(
    client: httpx.AsyncClient,
    *,
    frontend_url: str,
    token: str,
    brief: Brief,
    timeout_s: int,
) -> dict[str, Any]:
    url = f"{frontend_url.rstrip('/')}/api/eval/run"
    payload = {
        "userPrompt": brief.user,
        "platform": brief.platform,
        "exampleId": brief.example_id,
    }
    resp = await client.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


def _load_base_inference(
    inferences_dir: Path, base_config: str, example_id: str
) -> dict[str, Any] | None:
    p = inferences_dir / base_config / f"{example_id}.json"
    if not p.exists():
        return None
    data: dict[str, Any] = json.loads(p.read_text())
    return data


def _load_clean_text(
    inferences_clean_dir: Path, config: str, example_id: str
) -> str | None:
    """Return the rationale-stripped ad copy from inferences_clean/<cfg>/<id>.

    The eval pipeline's `normalize` step writes pre-extracted ad copy here
    (LLM extraction via claude-haiku-4-5 against the raw assistant_text).
    The scorer was trained on ad copy only — feeding raw assistant_text
    (which includes rationale prose, ``<think>`` blocks, and preambles)
    pulls the scorer off-distribution.
    """
    p = inferences_clean_dir / config / f"{example_id}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    text = data.get("assistant_text_clean")
    if not isinstance(text, str):
        return None
    if EXTRACTION_FAILED in text:
        return None
    return text.strip() or None


async def _score_cached_pipe(
    client: httpx.AsyncClient,
    *,
    envelope: dict[str, Any],
    brief: Brief,
    predictor_url: str,
    predictor_key: str,
    inferences_clean_dir: Path,
    pipe_config: str,
) -> tuple[str, dict[str, float] | None]:
    """Score a cached _pipe inference (e.g. C_pipe).

    Trust `campaign.predicted_score` when present (attached at emit time
    from the structured fields — no rationale leakage). For prose-only
    inferences, score the rationale-stripped text from inferences_clean,
    not the raw assistant_text.
    """
    campaign = envelope.get("campaign") if isinstance(envelope, dict) else None
    if isinstance(campaign, dict):
        display = _campaign_display_text(campaign)
        predicted = campaign.get("predicted_score")
        if isinstance(predicted, (int, float)):
            return display, {"composite": float(predicted)}
        fields = _campaign_copy_fields(campaign)
        score = await _score_one(
            client,
            predictor_url=predictor_url,
            api_key=predictor_key,
            platform=brief.platform,
            vertical=brief.vertical,
            **fields,
        )
        return display, score
    clean = _load_clean_text(inferences_clean_dir, pipe_config, brief.example_id)
    display = clean or "(no campaign emitted)"
    if not clean:
        return display, None
    score = await _score_one(
        client,
        predictor_url=predictor_url,
        api_key=predictor_key,
        platform=brief.platform,
        vertical=brief.vertical,
        headline=None,
        body=clean,
        description=None,
    )
    return display, score


async def _process_brief(
    client: httpx.AsyncClient,
    *,
    brief: Brief,
    frontend_url: str,
    token: str,
    predictor_url: str,
    predictor_key: str,
    base_config: str,
    compare_pipe: str,
    inferences_dir: Path,
    inferences_clean_dir: Path,
    timeout_s: int,
) -> dict[str, Any]:
    base_inf = _load_base_inference(inferences_dir, base_config, brief.example_id)
    # Score the rationale-stripped text the eval pipeline already extracted
    # (claude-haiku-4-5 via `eval.py normalize`). Feeding raw assistant_text
    # pulls the scorer off its training distribution — it was trained on ad
    # copy fields, not on rationale prose.
    base_clean = _load_clean_text(inferences_clean_dir, base_config, brief.example_id)
    base_score = await _score_one(
        client,
        predictor_url=predictor_url,
        api_key=predictor_key,
        platform=brief.platform,
        vertical=brief.vertical,
        headline=None,
        body=base_clean or None,
        description=None,
    )

    # Optional cached old-agent column (e.g. C_pipe). Read-only: never re-runs
    # the old agent — just scores whatever the prior eval run produced.
    pipe_inf: dict[str, Any] | None = None
    pipe_display = ""
    pipe_score: dict[str, float] | None = None
    if compare_pipe:
        pipe_inf = _load_base_inference(inferences_dir, compare_pipe, brief.example_id)
        if pipe_inf is not None:
            pipe_display, pipe_score = await _score_cached_pipe(
                client,
                envelope=pipe_inf,
                brief=brief,
                predictor_url=predictor_url,
                predictor_key=predictor_key,
                inferences_clean_dir=inferences_clean_dir,
                pipe_config=compare_pipe,
            )

    t0 = time.perf_counter()
    try:
        envelope = await _fire_agent(
            client,
            frontend_url=frontend_url,
            token=token,
            brief=brief,
            timeout_s=timeout_s,
        )
        agent_error = None
    except Exception as e:
        envelope = {}
        agent_error = f"{type(e).__name__}: {e}"
    agent_latency_ms = int((time.perf_counter() - t0) * 1000)

    campaign = envelope.get("campaign") if isinstance(envelope, dict) else None
    assistant_text = str(envelope.get("assistantText") or "") if isinstance(envelope, dict) else ""
    clean_body: str | None = None  # populated in the prose branch below

    if isinstance(campaign, dict):
        fields = _campaign_copy_fields(campaign)
        agent_display = _campaign_display_text(campaign)
        # Trust the auto-attached predictor score when present (it was scored
        # with the same predictor + the same platform tag at emit time).
        predicted_score = campaign.get("predicted_score")
        if isinstance(predicted_score, (int, float)):
            agent_score: dict[str, float] | None = {"composite": float(predicted_score)}
        else:
            agent_score = await _score_one(
                client,
                predictor_url=predictor_url,
                api_key=predictor_key,
                platform=brief.platform,
                vertical=brief.vertical,
                **fields,
            )
    else:
        agent_display = assistant_text or "(no campaign emitted)"
        # Mirror the eval pipeline: extract creative-only text before scoring.
        # The orchestrator's prose reply typically contains rationale ("Why
        # this works: ...") or variant labels alongside the ad copy; the
        # scorer was trained on ad copy fields, so feeding the raw blob is
        # off-distribution.
        if assistant_text and assistant_text.strip():
            try:
                extracted = await extract_ad_copy(
                    assistant_text, platform=brief.platform
                )
            except Exception as e:
                console.print(
                    f"[yellow]extract_ad_copy failed: {type(e).__name__}: {e}[/yellow]"
                )
                extracted = EXTRACTION_FAILED
        else:
            extracted = EXTRACTION_FAILED
        clean_body = (
            None if extracted == EXTRACTION_FAILED or not extracted.strip() else extracted
        )
        agent_score = await _score_one(
            client,
            predictor_url=predictor_url,
            api_key=predictor_key,
            platform=brief.platform,
            vertical=brief.vertical,
            headline=None,
            body=clean_body,
            description=None,
        )

    tools = _tool_summary(envelope.get("traces") if isinstance(envelope, dict) else None)

    return {
        "example_id": brief.example_id,
        "platform": brief.platform,
        "vertical": brief.vertical,
        "brief_user": brief.user,
        "reference_assistant": brief.reference_assistant,
        "base": {
            "config": base_config,
            "model_id": (base_inf or {}).get("model_id"),
            "text": (base_inf or {}).get("assistant_text") or "",
            "text_clean": base_clean,
            "score": base_score,
            "latency_ms": (base_inf or {}).get("latency_ms"),
            "missing": base_inf is None,
        },
        "pipe": {
            "config": compare_pipe or None,
            "model_id": (pipe_inf or {}).get("model_id"),
            "campaign": pipe_inf.get("campaign") if pipe_inf else None,
            "assistant_text": (pipe_inf or {}).get("assistant_text") or "",
            "display": pipe_display,
            "score": pipe_score,
            "latency_ms": (pipe_inf or {}).get("latency_ms"),
            "missing": bool(compare_pipe) and pipe_inf is None,
        },
        "agent": {
            "label": "agent v2 (new flow)",
            "model_id": envelope.get("modelId") if isinstance(envelope, dict) else None,
            "campaign": campaign if isinstance(campaign, dict) else None,
            "assistant_text": assistant_text,
            "assistant_text_clean": clean_body,
            "display": agent_display,
            "score": agent_score,
            "tool_calls": tools,
            "latency_ms": agent_latency_ms,
            "input_tokens": (envelope or {}).get("inputTokens"),
            "output_tokens": (envelope or {}).get("outputTokens"),
            "conversation_id": (envelope or {}).get("conversationId"),
            "error": agent_error,
        },
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _fmt_score(s: dict[str, float] | None) -> str:
    if not s:
        return "n/a"
    c = s.get("composite")
    return f"{c:.3f}" if isinstance(c, (int, float)) else "n/a"


def _delta(a: dict[str, float] | None, b: dict[str, float] | None) -> str:
    if not a or not b:
        return "n/a"
    ac, bc = a.get("composite"), b.get("composite")
    if not isinstance(ac, (int, float)) or not isinstance(bc, (int, float)):
        return "n/a"
    d = ac - bc
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def _write_report(out_dir: Path, pairs: list[dict[str, Any]], meta: dict[str, Any]) -> Path:
    has_pipe = bool(meta.get("compare_pipe"))
    pipe_label = meta.get("compare_pipe") or ""
    lines: list[str] = []
    lines.append(f"# Agent smoke report — {meta['run_id']}\n")
    lines.append(f"- Generated: {meta['generated_at']}")
    lines.append(f"- Frontend URL: `{meta['frontend_url']}`")
    lines.append(f"- Base config (single-shot): `{meta['base_config']}`")
    if has_pipe:
        lines.append(f"- Compare pipe (cached old agent): `{pipe_label}`")
    agent_model = meta.get("agent_model") or "(unknown)"
    lines.append(f"- Agent writer (frontend `MODEL_ID`): `{agent_model}`")
    lines.append(f"- N briefs: {len(pairs)}\n")

    lines.append("## Composite-score summary\n")
    if has_pipe:
        lines.append(
            f"| # | example_id | platform | base ({meta['base_config']}) | "
            f"old ({pipe_label}) | agent v2 | Δ (v2-old) | Δ (v2-base) | tools |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for i, p in enumerate(pairs, 1):
            bs, ps, ascr = p["base"]["score"], p["pipe"]["score"], p["agent"]["score"]
            tools = ", ".join(p["agent"]["tool_calls"]) or "-"
            lines.append(
                f"| {i} | `{p['example_id']}` | {p['platform']} | {_fmt_score(bs)} | "
                f"{_fmt_score(ps)} | {_fmt_score(ascr)} | "
                f"{_delta(ascr, ps)} | {_delta(ascr, bs)} | {tools} |"
            )
    else:
        lines.append("| # | example_id | platform | base | agent | Δ (agent-base) | tools |")
        lines.append("|---|---|---|---|---|---|---|")
        for i, p in enumerate(pairs, 1):
            bs, ascr = p["base"]["score"], p["agent"]["score"]
            tools = ", ".join(p["agent"]["tool_calls"]) or "-"
            lines.append(
                f"| {i} | `{p['example_id']}` | {p['platform']} | {_fmt_score(bs)} | "
                f"{_fmt_score(ascr)} | {_delta(ascr, bs)} | {tools} |"
            )
    lines.append("")

    # Aggregate
    base_cs = [p["base"]["score"]["composite"] for p in pairs if p["base"]["score"]]
    ag_cs = [p["agent"]["score"]["composite"] for p in pairs if p["agent"]["score"]]
    pipe_cs = (
        [p["pipe"]["score"]["composite"] for p in pairs if p["pipe"]["score"]]
        if has_pipe
        else []
    )
    if base_cs and ag_cs:
        bmean = sum(base_cs) / len(base_cs)
        amean = sum(ag_cs) / len(ag_cs)
        parts = [
            f"base ({meta['base_config']}, n={len(base_cs)}): {bmean:.3f}",
        ]
        if pipe_cs:
            pmean = sum(pipe_cs) / len(pipe_cs)
            parts.append(f"old ({pipe_label}, n={len(pipe_cs)}): {pmean:.3f}")
            parts.append(f"agent v2 (n={len(ag_cs)}): {amean:.3f}")
            parts.append(f"Δ(v2-old): {amean - pmean:+.3f}")
            parts.append(f"Δ(v2-base): {amean - bmean:+.3f}")
        else:
            parts.append(f"agent v2 (n={len(ag_cs)}): {amean:.3f}")
            parts.append(f"Δ: {amean - bmean:+.3f}")
        lines.append("**Mean composite** — " + " | ".join(parts) + "\n")

    for i, p in enumerate(pairs, 1):
        lines.append(f"---\n\n## {i}. `{p['example_id']}` — {p['platform']} / {p['vertical']}\n")
        lines.append("### Brief\n")
        lines.append("```\n" + p["brief_user"].strip() + "\n```\n")

        base_label = meta["base_config"]
        base_score_str = _fmt_score(p["base"]["score"])
        lines.append(f"### Base — `{base_label}` (composite: {base_score_str})\n")
        if p["base"]["missing"]:
            lines.append("_(no cached base inference for this brief)_\n")
        else:
            lines.append("```\n" + (p["base"]["text"] or "").strip() + "\n```\n")

        if has_pipe:
            pipe_score_str = _fmt_score(p["pipe"]["score"])
            lines.append(
                f"### Old agent — `{pipe_label}` (composite: {pipe_score_str})\n"
            )
            if p["pipe"]["missing"]:
                lines.append("_(no cached pipe inference for this brief)_\n")
            elif not p["pipe"]["campaign"]:
                lines.append("- **no campaign emitted** (prose only)\n")
                lines.append(
                    "```\n" + (p["pipe"]["assistant_text"] or "").strip() + "\n```\n"
                )
            else:
                lines.append("```\n" + (p["pipe"]["display"] or "").strip() + "\n```\n")

        lines.append(f"### Agent v2 — new flow (composite: {_fmt_score(p['agent']['score'])})\n")
        lines.append(
            f"- model_id: `{p['agent']['model_id']}` · latency: {p['agent']['latency_ms']}ms · "
            f"tokens: in={p['agent']['input_tokens']} out={p['agent']['output_tokens']}"
        )
        conv_id = p["agent"]["conversation_id"]
        lines.append(f"- conversation: `{conv_id}`")
        if conv_id:
            lines.append(
                f"- drill in: `uv run scripts/diagnostics/inspect_traces.py {conv_id}`"
            )
        if p["agent"]["tool_calls"]:
            lines.append(f"- tool calls: {', '.join(p['agent']['tool_calls'])}")
        if not p["agent"]["campaign"]:
            lines.append(
                "- **no campaign emitted** (prose only — `emit_campaign` did not fire)"
            )
        if p["agent"]["error"]:
            lines.append(f"- **error:** `{p['agent']['error']}`")
        lines.append("")
        lines.append("```\n" + p["agent"]["display"].strip() + "\n```\n")

        if p.get("reference_assistant"):
            lines.append("<details><summary>reference (gold winning ad — held out)</summary>\n")
            lines.append("\n```\n" + p["reference_assistant"].strip() + "\n```\n\n</details>\n")

    out = out_dir / "report.md"
    out.write_text("\n".join(lines))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command("run")
def run(
    n: int = typer.Option(5, help="Number of briefs to compare."),
    seed: int = typer.Option(42, help="RNG seed for brief selection."),
    base_config: str = typer.Option(
        "B", help="Cached single-shot baseline config under data/eval/inferences/ (read-only)."
    ),
    compare_pipe: str = typer.Option(
        "",
        help=(
            "Cached old-agent pipe config to score as a third column "
            "(e.g. 'C_pipe'). Read-only: loaded from data/eval/inferences/<cfg>/, "
            "never re-runs the old agent. Empty = pair-only (legacy mode)."
        ),
    ),
    frontend_url: str = typer.Option(
        "http://localhost:3000", envvar="EVAL_FRONTEND_B_PIPE_URL"
    ),
    predictor_url: str = typer.Option(
        "http://localhost:8001", envvar="SCORING_PREDICTOR_URL"
    ),
    timeout_s: int = typer.Option(180, help="Per-brief timeout for the agent call."),
    test_split_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_TEST_SPLIT, help="HF test split directory."
    ),
    inferences_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_INFERENCES, help="Root of cached eval inferences (read-only)."
    ),
    inferences_clean_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_INFERENCES_CLEAN,
        help=(
            "Root of rationale-stripped eval inferences (read-only). "
            "Produced by `eval.py normalize`. The scorer is trained on ad "
            "copy only, so we score these rather than raw assistant_text."
        ),
    ),
    run_id: str = typer.Option(
        "",
        help=(
            "Canonical run_id (YYYY-MM-DD-<slug>). "
            "Empty = auto-generated as YYYY-MM-DD-agent-smoke-HHMMSS."
        ),
    ),
    only_with_base: bool = typer.Option(
        True, help="Only pick briefs that already have a cached base inference."
    ),
    require_valid_pipe: str = typer.Option(
        "",
        help=(
            "Only pick briefs where the named cached pipe config (e.g. 'B_pipe') "
            "produced ad copy — i.e. its cached inference called ask_draper or "
            "emit_campaign. Excludes orchestrator-only clarifier responses."
        ),
    ),
    platforms: str = typer.Option(
        "", help="Comma-separated platform filter (e.g. 'meta,tiktok'). Empty = all."
    ),
    only_ids: str = typer.Option(
        "",
        help=(
            "Comma-separated example_ids — when set, runs ONLY these briefs "
            "(bypasses --n / --seed / --platforms / --only-with-base)."
        ),
    ),
) -> None:
    """Run the side-by-side smoke comparison."""
    token = os.environ.get("EVAL_SERVICE_TOKEN", "")
    if not token:
        console.print("[red]EVAL_SERVICE_TOKEN is not set[/red]")
        raise typer.Exit(code=1)
    predictor_key = os.environ.get("SCORING_PREDICTOR_API_KEY", "")
    if not predictor_key:
        console.print("[red]SCORING_PREDICTOR_API_KEY is not set[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Loading test briefs from {test_split_dir}[/green]")
    briefs = load_test_briefs(test_split_dir)
    if only_ids:
        wanted_ids = [x.strip() for x in only_ids.split(",") if x.strip()]
        by_id = {b.example_id: b for b in briefs}
        missing = [x for x in wanted_ids if x not in by_id]
        if missing:
            console.print(f"[red]Unknown example_id(s): {missing}[/red]")
            raise typer.Exit(code=1)
        picks = [by_id[x] for x in wanted_ids]
        console.print(f"[green]Running {len(picks)} explicit brief(s): {wanted_ids}.[/green]")
    else:
        if platforms:
            wanted = {p.strip() for p in platforms.split(",") if p.strip()}
            briefs = [b for b in briefs if b.platform in wanted]
        if only_with_base:
            base_dir = inferences_dir / base_config
            briefs = [b for b in briefs if (base_dir / f"{b.example_id}.json").exists()]
        if require_valid_pipe:
            # Inline mini-classifier — same logic as eval_clean_pipe.classify_config.
            # We can't import that one without dragging Polars into this script's
            # cold path; the rules are short enough to duplicate here.
            writer_tools = {"ask_draper", "draft_campaign", "emit_campaign"}
            pipe_dir = inferences_dir / require_valid_pipe
            valid: set[str] = set()
            for jf in pipe_dir.glob("*.json"):
                data = json.loads(jf.read_text())
                tool_set: set[str] = set()
                for stage in data.get("tool_calls") or []:
                    tc = stage.get("toolCalls") if isinstance(stage, dict) else None
                    inner = tc.get("_toolCalls") if isinstance(tc, dict) else tc
                    if isinstance(inner, list):
                        for entry in inner:
                            if isinstance(entry, dict):
                                tn = entry.get("toolName") or entry.get("name")
                                if isinstance(tn, str):
                                    tool_set.add(tn)
                if tool_set & writer_tools or data.get("campaign"):
                    valid.add(data["example_id"])
            before = len(briefs)
            briefs = [b for b in briefs if b.example_id in valid]
            console.print(
                f"[green]Filter --require-valid-pipe={require_valid_pipe}: "
                f"{before} → {len(briefs)} briefs.[/green]"
            )
        if not briefs:
            console.print("[red]No eligible briefs after filtering.[/red]")
            raise typer.Exit(code=1)

        rng = random.Random(seed)
        picks = rng.sample(briefs, k=min(n, len(briefs)))
        console.print(f"[green]Picked {len(picks)} briefs (seed={seed}).[/green]")

    try:
        run_id = _resolve_run_id(run_id or None)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    out_dir = _diagnostics_out_dir(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Writing run {run_id} → {out_dir}[/green]")

    async def _go() -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        async with httpx.AsyncClient() as client:
            for i, brief in enumerate(picks, 1):
                console.print(
                    f"[cyan][{i}/{len(picks)}][/cyan] {brief.example_id} "
                    f"({brief.platform}/{brief.vertical}) → POST agent…"
                )
                pair = await _process_brief(
                    client,
                    brief=brief,
                    frontend_url=frontend_url,
                    token=token,
                    predictor_url=predictor_url,
                    predictor_key=predictor_key,
                    base_config=base_config,
                    compare_pipe=compare_pipe,
                    inferences_dir=inferences_dir,
                    inferences_clean_dir=inferences_clean_dir,
                    timeout_s=timeout_s,
                )
                results.append(pair)
                bs = _fmt_score(pair["base"]["score"])
                asr = _fmt_score(pair["agent"]["score"])
                err = pair["agent"]["error"]
                marker = "[red]err[/red]" if err else "[green]ok[/green]"
                console.print(
                    f"     {marker}  base={bs}  agent={asr}  "
                    f"Δ={_delta(pair['agent']['score'], pair['base']['score'])}  "
                    f"tools=[{', '.join(pair['agent']['tool_calls']) or '-'}]"
                )
        return results

    pairs = asyncio.run(_go())

    jsonl_path = out_dir / "pairs.jsonl"
    with jsonl_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, default=str) + "\n")

    meta = {
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "frontend_url": frontend_url,
        "base_config": base_config,
        "compare_pipe": compare_pipe or None,
        "agent_model": pairs[0]["agent"]["model_id"] if pairs else None,
    }
    report_path = _write_report(out_dir, pairs, meta)

    console.print(f"\n[bold green]Wrote {report_path}[/bold green]")
    console.print(f"[bold green]Wrote {jsonl_path}[/bold green]")


@app.command("rescore")
def rescore(
    run_id: str = typer.Argument(
        ..., help="Existing canonical run_id (e.g. 2026-05-15-agent-smoke-094008)."
    ),
    predictor_url: str = typer.Option(
        "http://localhost:8001", envvar="SCORING_PREDICTOR_URL"
    ),
    inferences_clean_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_INFERENCES_CLEAN, help="Root of rationale-stripped cached inferences."
    ),
    suffix: str = typer.Option(
        "-rescore", help="Suffix appended to the source run_id for the rescored output."
    ),
) -> None:
    """Re-score an existing run's pairs.jsonl with rationale-stripped text.

    Reuses the original agent prose / campaign outputs — does NOT re-fire
    the agent. For prose-only paths, runs `extract_ad_copy` (claude-haiku-4-5)
    to produce ad-copy-only text before scoring. For successful emits, keeps
    the auto-attached `campaign.predicted_score` (already structured-field-only,
    no rationale). For cached base / pipe columns, reads the pre-extracted
    `assistant_text_clean` from data/eval/inferences_clean/<cfg>/<id>.json.

    Writes a fresh run directory; the source run is untouched.
    """
    predictor_key = os.environ.get("SCORING_PREDICTOR_API_KEY", "")
    if not predictor_key:
        console.print("[red]SCORING_PREDICTOR_API_KEY is not set[/red]")
        raise typer.Exit(code=1)

    try:
        validate_run_id(run_id)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    src_dir = _diagnostics_out_dir(run_id)
    src_jsonl = src_dir / "pairs.jsonl"
    if not src_jsonl.exists():
        console.print(f"[red]No pairs.jsonl at {src_jsonl}[/red]")
        raise typer.Exit(code=1)

    pairs_in: list[dict[str, Any]] = []
    with src_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if line:
                pairs_in.append(json.loads(line))
    console.print(f"[green]Loaded {len(pairs_in)} pairs from {src_jsonl}[/green]")

    base_config = pairs_in[0]["base"]["config"]
    pipe_config = pairs_in[0]["pipe"]["config"] if pairs_in[0].get("pipe") else None

    async def _go() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        async with httpx.AsyncClient() as client:
            for i, p in enumerate(pairs_in, 1):
                ex_id = p["example_id"]
                platform = p["platform"]
                vertical = p["vertical"]

                # Base column — clean text from inferences_clean.
                base_clean = _load_clean_text(inferences_clean_dir, base_config, ex_id)
                base_score = (
                    await _score_one(
                        client,
                        predictor_url=predictor_url,
                        api_key=predictor_key,
                        platform=platform,
                        vertical=vertical,
                        headline=None,
                        body=base_clean,
                        description=None,
                    )
                    if base_clean
                    else None
                )

                # Pipe column — predicted_score on successful emit, else clean text.
                pipe_score: dict[str, float] | None = None
                if pipe_config:
                    pipe_campaign = p["pipe"].get("campaign")
                    if isinstance(pipe_campaign, dict) and isinstance(
                        pipe_campaign.get("predicted_score"), (int, float)
                    ):
                        pipe_score = {
                            "composite": float(pipe_campaign["predicted_score"])
                        }
                    else:
                        pipe_clean = _load_clean_text(
                            inferences_clean_dir, pipe_config, ex_id
                        )
                        if pipe_clean:
                            pipe_score = await _score_one(
                                client,
                                predictor_url=predictor_url,
                                api_key=predictor_key,
                                platform=platform,
                                vertical=vertical,
                                headline=None,
                                body=pipe_clean,
                                description=None,
                            )

                # Agent column — predicted_score on emit, else extract+score prose.
                ag_campaign = p["agent"].get("campaign")
                ag_score: dict[str, float] | None = None
                ag_clean: str | None = None
                if isinstance(ag_campaign, dict) and isinstance(
                    ag_campaign.get("predicted_score"), (int, float)
                ):
                    ag_score = {"composite": float(ag_campaign["predicted_score"])}
                else:
                    raw = p["agent"].get("assistant_text") or ""
                    if raw.strip():
                        try:
                            extracted = await extract_ad_copy(raw, platform=platform)
                            if extracted and extracted != EXTRACTION_FAILED:
                                ag_clean = extracted.strip() or None
                        except Exception as e:
                            console.print(
                                f"[yellow]extract_ad_copy failed for {ex_id}: "
                                f"{type(e).__name__}: {e}[/yellow]"
                            )
                    if ag_clean:
                        ag_score = await _score_one(
                            client,
                            predictor_url=predictor_url,
                            api_key=predictor_key,
                            platform=platform,
                            vertical=vertical,
                            headline=None,
                            body=ag_clean,
                            description=None,
                        )

                new = json.loads(json.dumps(p))  # deep copy
                new["base"]["score"] = base_score
                new["base"]["text_clean"] = base_clean
                if pipe_config and "pipe" in new:
                    new["pipe"]["score"] = pipe_score
                new["agent"]["score"] = ag_score
                new["agent"]["assistant_text_clean"] = ag_clean
                out.append(new)

                console.print(
                    f"[cyan][{i}/{len(pairs_in)}][/cyan] {ex_id} "
                    f"base={_fmt_score(base_score)} "
                    f"pipe={_fmt_score(pipe_score)} "
                    f"agent={_fmt_score(ag_score)} "
                    f"Δ(v2-base)={_delta(ag_score, base_score)}"
                )
        return out

    pairs_out = asyncio.run(_go())

    dst_run = f"{run_id}{suffix}"
    try:
        validate_run_id(dst_run)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    dst_dir = _diagnostics_out_dir(dst_run)
    dst_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = dst_dir / "pairs.jsonl"
    with jsonl_path.open("w") as f:
        for p in pairs_out:
            f.write(json.dumps(p, default=str) + "\n")

    meta = {
        "run_id": dst_run,
        "generated_at": datetime.now(UTC).isoformat(),
        "frontend_url": f"(rescore of {run_id})",
        "base_config": base_config,
        "compare_pipe": pipe_config,
        "agent_model": pairs_out[0]["agent"]["model_id"] if pairs_out else None,
    }
    report_path = _write_report(dst_dir, pairs_out, meta)

    console.print(f"\n[bold green]Wrote {report_path}[/bold green]")
    console.print(f"[bold green]Wrote {jsonl_path}[/bold green]")


# ---- materialize (bridge to eval.py compare) -----------------------------


_LEARNED_HEADS = ("composite", "survivability", "engagement_volume", "engagement_velocity")


def _stats_for_head(values: list[float]) -> dict[str, float | int]:
    """Return ``{n, mean, median, p25, p75, p90}`` for a list of scores.

    Empty input returns zeros — matches the canonical
    ``learned_scores_summary.parquet`` shape so the resulting row can join
    cleanly even when a head has no observations (e.g. ``C_pipe`` where
    only emit-time ``composite`` is recorded). Sorted-list quantile keeps
    the schema strictly Python-typed (no polars-stub coercion noise).
    """
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0}
    s = sorted(values)
    n = len(s)

    def _q(q: float) -> float:
        # Linear interpolation, identical to polars default quantile.
        if n == 1:
            return s[0]
        idx = q * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return s[lo] + (s[hi] - s[lo]) * frac

    return {
        "n": n,
        "mean": sum(s) / n,
        "median": _q(0.5),
        "p25": _q(0.25),
        "p75": _q(0.75),
        "p90": _q(0.90),
    }


def _row_for_config(config: str, scores: list[dict[str, float]]) -> dict[str, Any]:
    """Build one ``learned_scores_summary`` row from a list of score dicts.

    Each score dict may carry any subset of ``_LEARNED_HEADS``; missing
    heads contribute zero observations to that head's column group.
    """
    row: dict[str, Any] = {"config": config, "n": len(scores)}
    for head in _LEARNED_HEADS:
        head_vals = [
            float(s[head]) for s in scores if isinstance(s.get(head), (int, float))
        ]
        stats = _stats_for_head(head_vals)
        row[f"{head}_n"] = stats["n"]
        row[f"{head}_mean"] = stats["mean"]
        row[f"{head}_median"] = stats["median"]
        row[f"{head}_p25"] = stats["p25"]
        row[f"{head}_p75"] = stats["p75"]
        row[f"{head}_p90"] = stats["p90"]
    return row


@app.command("materialize")
def materialize(
    run_id: str = typer.Argument(
        ..., help="Canonical smoke run_id (e.g. 2026-05-15-agent-smoke-094008)."
    ),
    agent_config: str = typer.Option(
        ...,
        "--agent-config",
        help=(
            "Config-name slot the live agent column should occupy "
            "(e.g. 'A_pipe@hook-v2'). Must match the variant naming rules."
        ),
    ),
    force: bool = typer.Option(
        False, help="Overwrite an existing aggregates/learned_scores_summary.parquet."
    ),
) -> None:
    """Convert a smoke ``pairs.jsonl`` into ``learned_scores_summary.parquet``.

    Writes ``runs/<run_id>/aggregates/learned_scores_summary.parquet`` in
    the same shape as ``eval.py score-summary``, so the smoke run becomes a
    candidate for ``eval.py compare --base <eval_run> --candidate <run_id>``.

    Three rows are emitted: the cached base config, the cached pipe config
    (if present), and the live agent under ``--agent-config``. Each row
    carries n / mean / median / p25 / p75 / p90 for the four predictor
    heads, with zero-n where a head wasn't recorded (e.g. ``C_pipe`` only
    has emit-time ``composite``).
    """
    try:
        validate_run_id(run_id)
        validate_config_name(agent_config)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    src_jsonl = _diagnostics_out_dir(run_id) / "pairs.jsonl"
    if not src_jsonl.exists():
        console.print(f"[red]No pairs.jsonl at {src_jsonl}[/red]")
        raise typer.Exit(code=1)

    pairs: list[dict[str, Any]] = []
    with src_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    if not pairs:
        console.print(f"[red]Empty pairs.jsonl at {src_jsonl}[/red]")
        raise typer.Exit(code=1)

    base_config = pairs[0]["base"]["config"]
    pipe_config = pairs[0]["pipe"]["config"] if pairs[0].get("pipe") else None

    base_scores: list[dict[str, float]] = [
        p["base"]["score"] for p in pairs if isinstance(p["base"].get("score"), dict)
    ]
    pipe_scores: list[dict[str, float]] = (
        [
            p["pipe"]["score"]
            for p in pairs
            if p.get("pipe") and isinstance(p["pipe"].get("score"), dict)
        ]
        if pipe_config
        else []
    )
    agent_scores: list[dict[str, float]] = [
        p["agent"]["score"] for p in pairs if isinstance(p["agent"].get("score"), dict)
    ]

    rows: list[dict[str, Any]] = [_row_for_config(base_config, base_scores)]
    if pipe_config:
        rows.append(_row_for_config(pipe_config, pipe_scores))
    rows.append(_row_for_config(agent_config, agent_scores))

    df = pl.DataFrame(rows)
    out_dir = _PATHS.aggregates_dir(run_id)
    out_path = out_dir / "learned_scores_summary.parquet"
    if out_path.exists() and not force:
        console.print(
            f"[red]{out_path} already exists. Pass --force to overwrite.[/red]"
        )
        raise typer.Exit(code=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)

    console.print(f"[green]Wrote {out_path}[/green]")
    console.print(df)
    console.print(
        f"\n[bold]Now diff vs a baseline run:[/bold]\n"
        f"  uv run python scripts/eval.py compare "
        f"--base <baseline_run_id> --candidate {run_id}"
    )


if __name__ == "__main__":
    app()
