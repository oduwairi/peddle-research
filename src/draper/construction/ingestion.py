"""Shared response-to-TrainingExample conversion.

Both ``ingest`` (chat-client workflow) and ``batch-collect`` (API batch
workflow) take a raw tagged response plus a sidecar dict of rolled-dice
provenance and need to turn that into a ``TrainingExample`` saved to
disk. Extracting the logic here keeps the two CLI paths consistent.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from draper.construction.base_constructor import business_category
from draper.construction.bundle import parse_bundle_output
from draper.construction.formats.registry import get_pipeline
from draper.construction.schemas import (
    ChatMessage,
    ExampleMetadata,
    PromptStyle,
    TrainingExample,
)
from draper.scoring.schemas import ScoredAd
from draper.utils.io import read_jsonl

if TYPE_CHECKING:
    from draper.construction.base_constructor import BaseConstructor

logger = logging.getLogger("draper")


@dataclass
class IngestResult:
    """Outcome of attempting to save one response as a training example."""

    saved: int = 0
    error: str = ""
    verbatim_failed: bool = False  # backtranslation fidelity check failed


def load_ads_by_id(scored_ads_path: str) -> dict[str, ScoredAd]:
    """Materialize the scored-ads JSONL into an id → ScoredAd lookup.

    Reading the full file once per batch-collect (or ingest) run is
    cheaper than re-reading it per example.
    """
    ads_by_id: dict[str, ScoredAd] = {}
    for rec in read_jsonl(scored_ads_path):
        ad = ScoredAd(**rec)
        ads_by_id[ad.ad.ad_id] = ad
    return ads_by_id


def _append_bundle_sidecar(
    output_dir: Path,
    task_format: str,
    example_id: str,
    teacher_bundle: str,
) -> None:
    """Append one bundle diagnostic record to the per-format sidecar.

    The bundle is the exact prompt text sent to the teacher provider. It
    lives outside ``examples.jsonl`` so the training payload stays lean
    while the review renderer can still show what each teacher saw.
    Best-effort: a failure here must not abort ingest (bundle sidecars
    are diagnostic-only, not training state).
    """
    path = output_dir / task_format / "bundles.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"example_id": example_id, "teacher_bundle": teacher_bundle}
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Bundle sidecar write failed for %s: %s", example_id, exc)


def _append_rejection_record(
    output_dir: Path,
    task_format: str,
    source_ad_ids: list[str],
    reason: str,
    batch_id: str,
) -> None:
    """Append one rejection record to the per-format rejected-ads sidecar.

    Rejected ads must be excluded from future source selection so the
    same hard ads don't repeatedly burn teacher tokens batch after
    batch. Without this, ``consumed_ad_ids()`` (which reads only saved
    examples) lets failed ads bubble back to the top of the deterministic
    selector ordering on every submission.
    """
    path = output_dir / task_format / "_rejected_ads.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "source_ad_ids": source_ad_ids,
        "reason": reason,
        "batch_id": batch_id,
    }
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Rejection record write failed: %s", exc)


def ingest_response(
    response_text: str,
    sidecar: dict[str, Any],
    constructor: BaseConstructor,
    ads_by_id: dict[str, ScoredAd],
    *,
    construction_model_override: str = "",
    teacher_bundle: str = "",
    batch_id: str = "",
) -> IngestResult:
    """Parse a tagged response, build a ``TrainingExample``, and save it.

    Parameters
    ----------
    response_text:
        Raw text returned by the chat agent or batch API. Must contain
        the required ``<user_prompt>`` / ``<assistant_response>`` tags
        (plus the multi-turn tags when applicable).
    sidecar:
        Dice-roll metadata produced by ``prepare_bundles`` — either the
        chat workflow's ``_last_prepared.json`` entry or the batch
        registry's ``PendingBatchSidecar.__dict__``.
    constructor:
        The task constructor for the target format (used for system-
        prompt lookup and saving).
    ads_by_id:
        Pre-loaded scored-ad lookup (see ``load_ads_by_id``).
    construction_model_override:
        When set, overrides the sidecar's declared provider in the
        saved ``construction_model`` field. Batch mode passes the API
        model ID here (e.g. ``"gpt-4o-mini"``) so provenance records
        the real teacher rather than a provider bucket.
    """
    parsed = parse_bundle_output(response_text)
    if not parsed.user_prompt or not parsed.assistant_response:
        return IngestResult(
            error=(
                "Missing required tags <user_prompt> and/or "
                "<assistant_response>. Response is unusable."
            )
        )

    source_ad_ids = [str(x) for x in sidecar.get("source_ad_ids") or []]
    prompt_style = PromptStyle(str(sidecar.get("prompt_style", "backtranslation")))
    persona_id = str(sidecar.get("persona_id", ""))
    seed_idx = int(sidecar.get("seed_idx", -1) or -1)
    evol_op = str(sidecar.get("evol_op", ""))
    difficulty = str(sidecar.get("difficulty", "standard"))
    declared_provider = str(sidecar.get("provider", ""))

    source_ads: list[ScoredAd] = []
    for ad_id in source_ad_ids:
        resolved = ads_by_id.get(ad_id)
        if resolved is not None:
            source_ads.append(resolved)
    if not source_ads:
        return IngestResult(error="Could not resolve any source ads from sidecar.")

    # Per-format ingestion check (copywriting enforces backtranslation fidelity).
    pipeline = get_pipeline(constructor.task_format)
    check = pipeline.ingestion_check(
        parsed.assistant_response, source_ads, prompt_style
    )
    if not check.passed:
        _append_rejection_record(
            output_dir=constructor.output_dir,
            task_format=constructor.task_format.value,
            source_ad_ids=[a.ad.ad_id for a in source_ads],
            reason=check.reason,
            batch_id=batch_id,
        )
        return IngestResult(verbatim_failed=True, error=check.reason)

    construction_model = construction_model_override or declared_provider

    metadata = ExampleMetadata(
        source_ad_ids=[a.ad.ad_id for a in source_ads],
        source_tiers=[a.tier for a in source_ads],
        source_scores=[a.composite_score for a in source_ads],
        platform=source_ads[0].ad.platform.value if source_ads else "",
        vertical=business_category(source_ads[0]) if source_ads else "",
        construction_model=construction_model,
        prompt_style=prompt_style,
        persona_id=persona_id,
        seed_idx=seed_idx,
        evol_op=evol_op,
        difficulty=difficulty,
        turn_structure="single",
        followup_type="",
        source_ad_shape=str(sidecar.get("source_ad_shape", "")),
        conversation_register=str(
            sidecar.get("conversation_register")
            or sidecar.get("brief_register", "")
        ),
        batch_id=batch_id,
    )

    messages = [
        ChatMessage(role="system", content=constructor.SYSTEM_PROMPT),
        ChatMessage(role="user", content=parsed.user_prompt),
        ChatMessage(role="assistant", content=parsed.assistant_response),
    ]

    example = TrainingExample(
        task_format=constructor.task_format,
        messages=messages,
        metadata=metadata,
    )
    saved = constructor.save_examples([example])
    if saved and teacher_bundle:
        _append_bundle_sidecar(
            output_dir=constructor.output_dir,
            task_format=constructor.task_format.value,
            example_id=example.example_id,
            teacher_bundle=teacher_bundle,
        )
    return IngestResult(saved=saved)
