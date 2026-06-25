"""Construction v2 orchestration — unified single-pass pipeline.

The CLI in ``scripts/construct_v2.py`` is a thin dispatch layer over
these functions. All file I/O, batch submission/collection, and the
ingest loop live here so they can be unit-tested directly and reused
from notebooks or smoke scripts without going through Typer.

Path helpers derive from :class:`ConstructionV2Config` so every caller
agrees on where briefs / raw responses / examples / filtered records /
audit logs land. The optional ``run_id`` argument scopes batch
registries and audit overlays under ``runs/<run_id>/`` for smoke /
exploration runs without polluting the production tree.
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from draper.construction.batch import (
    BatchRegistry,
    PendingBatch,
    PendingBatchSidecar,
    make_batch_client,
    validate_batch_model,
)
from draper.construction.batch.registry import utc_now_iso
from draper.construction_v2.config import (
    BatchConfig,
    ConstructionV2Config,
)
from draper.construction_v2.dataset.source_selector import (
    SourceAd,
    load_source_ads_by_id,
    selection_lineage_hash,
)
from draper.construction_v2.ingest.response_parser import (
    ParsedResponse,
    ParseRejection,
    parse_response,
)
from draper.construction_v2.ingest.skills import get_bundle
from draper.construction_v2.schemas.records import ExampleRecord, RejectionRecord
from draper.utils.io import read_jsonl, write_jsonl

CUSTOM_ID_PREFIX = "teacher-"


def task_format_for(skill: str) -> str:
    """Build the ``task_format`` string registered with batches for ``skill``.

    Kept symmetric with ``BatchRegistry``: ``"<skill>_v2"`` distinguishes
    each skill's batches from other v2 work and from the legacy v1 tree.
    """
    return f"{skill}_v2"


# Legacy module-level constant; tests and the CLI imported it before the
# multi-skill refactor. Kept pointing at the default-skill string so older
# call sites keep working without change.
TASK_FORMAT = task_format_for("copywriting")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SelectionLineageMismatch(RuntimeError):
    """The selection parquet hash no longer matches the current config."""


class PartialFailureThreshold(RuntimeError):
    """A collected batch's provider-error ratio exceeds ``max_partial_error_rate``."""


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def runs_dir(config: ConstructionV2Config, run_id: str) -> Path:
    """Per-run overlay directory under ``output_dir/runs/<run_id>/``."""
    return Path(config.output_dir) / "runs" / run_id


def _scope_root(config: ConstructionV2Config, run_id: str | None) -> Path:
    """Return ``output_dir`` for production, ``runs/<run_id>`` for a run."""
    return runs_dir(config, run_id) if run_id else Path(config.output_dir)


def briefs_path(config: ConstructionV2Config, *, run_id: str | None = None) -> Path:
    """Where parsed briefs live (single-pass writes ``<brief>`` JSON here)."""
    if run_id:
        return runs_dir(config, run_id) / config.skill / "briefs.jsonl"
    return Path(config.single_pass.briefs_cache_path)


def responses_path(config: ConstructionV2Config, *, run_id: str | None = None) -> Path:
    """Where parsed ``<think>+deliverable`` rows live."""
    return _scope_root(config, run_id) / config.skill / "responses_raw.jsonl"


def examples_path(config: ConstructionV2Config, *, run_id: str | None = None) -> Path:
    return _scope_root(config, run_id) / config.skill / "examples.jsonl"


def filtered_path(config: ConstructionV2Config, *, run_id: str | None = None) -> Path:
    return _scope_root(config, run_id) / config.skill / "filtered.jsonl"


def audit_path(config: ConstructionV2Config, name: str, *, run_id: str | None = None) -> Path:
    """Audit log path. Per-run audits land under ``runs/<run_id>/_audit/``."""
    base = runs_dir(config, run_id) / "_audit" if run_id else Path(config.audit_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / name


def selection_audit_path(config: ConstructionV2Config) -> Path:
    """The shared selection.parquet — always at ``audit_dir``, never per-run."""
    return Path(config.audit_dir) / "selection.parquet"


def manifest_path(config: ConstructionV2Config, run_id: str) -> Path:
    return runs_dir(config, run_id) / "manifest.json"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def append_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_briefs(
    config: ConstructionV2Config, *, run_id: str | None = None
) -> dict[str, dict[str, Any]]:
    """Return ``{ad_id: raw_brief_dict}`` from the briefs cache (empty if absent).

    The cached value is the teacher's ``<brief>`` JSON. Validation and any
    skill-specific field injection (e.g. image_brief's verbatim ``copy``)
    happen at ingest via the skill bundle's ``build_brief``, where the
    source ad is in scope — so this loader stays brief-model-agnostic and
    returns raw dicts.
    """
    path = briefs_path(config, run_id=run_id)
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        ad_id = row.get("ad_id")
        brief = row.get("brief")
        if not isinstance(ad_id, str) or not isinstance(brief, dict):
            continue
        out[ad_id] = brief
    return out


def load_selection_ad_ids(config: ConstructionV2Config) -> list[str]:
    """Read the ad_id column from ``_audit/selection.parquet``."""
    audit = selection_audit_path(config)
    if not audit.exists():
        msg = f"Selection audit not found at {audit}. Run `scripts/construct_v2.py select` first."
        raise FileNotFoundError(msg)
    import polars as pl

    df = pl.read_parquet(audit)
    return [str(aid) for aid in df["ad_id"].to_list()]


def _read_selection_lineage(config: ConstructionV2Config) -> str | None:
    """Return the stored selection lineage hash, or ``None`` if absent."""
    audit = selection_audit_path(config)
    if not audit.exists():
        return None
    import polars as pl

    df = pl.read_parquet(audit)
    if "selection_lineage_hash" not in df.columns:
        return None
    series = df["selection_lineage_hash"]
    return None if series.is_empty() else str(series[0])


def load_ads_for_selection(
    config: ConstructionV2Config, *, limit: int | None = None
) -> list[SourceAd]:
    ad_ids = load_selection_ad_ids(config)
    if limit is not None:
        ad_ids = ad_ids[:limit]
    by_id = load_source_ads_by_id(config, ad_ids)
    return [by_id[aid] for aid in ad_ids if aid in by_id]


def registry_for(config: ConstructionV2Config, *, run_id: str | None = None) -> BatchRegistry:
    root = _scope_root(config, run_id)
    return BatchRegistry(root, task_format_for(config.skill))


# ---------------------------------------------------------------------------
# Slice partitioning
# ---------------------------------------------------------------------------


def parse_slice_spec(spec: str) -> tuple[int, int, int]:
    """Parse a slice spec into ``(start, end, N)`` with ``0 <= start <= end < N``.

    Supports two forms:

    - ``"i/N"`` — single contiguous chunk, equivalent to ``(i, i, N)``.
    - ``"i-j/N"`` — union of contiguous chunks ``i`` through ``j``
      inclusive, e.g. ``"0-7/20"`` is the first eight twentieths.
    """
    cleaned = spec.strip()
    if "/" not in cleaned:
        msg = f"slice_spec must be 'i/N' or 'i-j/N' (got {spec!r})"
        raise ValueError(msg)
    numer_part, denom_part = cleaned.split("/", 1)
    try:
        n = int(denom_part)
    except ValueError as exc:
        msg = f"slice_spec denominator must be an integer (got {spec!r})"
        raise ValueError(msg) from exc
    if n <= 0:
        msg = f"slice_spec denominator must be > 0 (got {spec!r})"
        raise ValueError(msg)
    if "-" in numer_part:
        a, b = numer_part.split("-", 1)
        try:
            start = int(a)
            end = int(b)
        except ValueError as exc:
            msg = f"slice_spec range must be 'i-j' integers (got {spec!r})"
            raise ValueError(msg) from exc
        if start > end:
            msg = f"slice_spec range start must be <= end (got {spec!r})"
            raise ValueError(msg)
    else:
        try:
            start = int(numer_part)
        except ValueError as exc:
            msg = f"slice_spec numerator must be an integer (got {spec!r})"
            raise ValueError(msg) from exc
        end = start
    if start < 0 or end >= n:
        msg = f"slice_spec indices must satisfy 0 <= start <= end < N (got {spec!r})"
        raise ValueError(msg)
    return start, end, n


def apply_slice(items: list[str], spec: str) -> list[str]:
    """Take the union of slices ``start..end`` (inclusive) out of ``N``.

    Single-slice form ``"i/N"`` returns the ``i``-th of ``N`` disjoint
    contiguous chunks. Range form ``"i-j/N"`` returns the concatenation
    of slices ``i`` through ``j`` (a single contiguous range, since
    underlying chunks are contiguous).
    """
    start, end, n = parse_slice_spec(spec)
    if not items:
        return []
    chunk = math.ceil(len(items) / n)
    lo = start * chunk
    hi = min((end + 1) * chunk, len(items))
    return items[lo:hi]


# ---------------------------------------------------------------------------
# Lineage verification
# ---------------------------------------------------------------------------


def verify_selection_lineage(
    config: ConstructionV2Config,
    *,
    allow_drift: bool = False,
    run_id: str | None = None,
) -> None:
    """Raise :class:`SelectionLineageMismatch` on stale selection.parquet.

    Missing ``selection_lineage_hash`` column → log warning, proceed
    (compatibility with selection.parquets written by the old selector).
    When the hash differs AND the registry already has reserved ad_ids,
    abort unless ``allow_drift`` is True.
    """
    import logging

    log = logging.getLogger("draper")
    stored = _read_selection_lineage(config)
    if stored is None:
        log.warning(
            "selection.parquet has no selection_lineage_hash column; "
            "skipping lineage check. Re-run `select` to populate it."
        )
        return
    expected = selection_lineage_hash(config.selection, Path(config.selection.scored_ads_path))
    if stored == expected:
        return
    registry = registry_for(config, run_id=run_id)
    if not registry.reserved_ad_ids():
        log.info(
            "selection.parquet lineage drift detected (stored=%s expected=%s) "
            "but no batches reserve ads — proceeding.",
            stored,
            expected,
        )
        return
    if allow_drift:
        log.warning(
            "selection.parquet lineage drift (stored=%s expected=%s) with "
            "reserved ad_ids in registry — proceeding because "
            "allow_drift=True.",
            stored,
            expected,
        )
        return
    msg = (
        f"selection.parquet lineage hash {stored} does not match "
        f"current config ({expected}), and the batch registry has "
        f"reserved ad_ids from the old universe. Either clear the "
        f"registry (cancel in-flight batches) or rerun `select` after "
        f"collecting them. Pass --allow-lineage-drift to override."
    )
    raise SelectionLineageMismatch(msg)


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


class SubmitResult(BaseModel):
    """Outcome of submitting a single-pass batch."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str = ""
    provider: str = ""
    model: str = ""
    request_count: int = 0
    skipped: bool = False
    skipped_reason: str = ""


async def submit_single_pass(
    config: ConstructionV2Config,
    *,
    provider: str,
    slice_spec: str = "0/1",
    limit: int | None = None,
    model_override: str | None = None,
    run_id: str | None = None,
    allow_lineage_drift: bool = False,
) -> SubmitResult:
    """Submit a single-pass batch for one provider's disjoint ad slice.

    The provider's model is resolved from
    ``config.providers[provider]`` unless ``model_override`` is set.
    ``slice_spec`` (``"i/N"``) partitions the shared
    ``selection.parquet`` into N disjoint contiguous chunks; this
    submission takes the ``i``-th chunk. Within that chunk, ad_ids
    already reserved by other in-flight batches in this scope are
    excluded.
    """
    provider_cfg = config.provider_config(provider)
    model = model_override or provider_cfg.model
    validate_batch_model(model)
    verify_selection_lineage(config, allow_drift=allow_lineage_drift, run_id=run_id)

    all_ids = load_selection_ad_ids(config)
    slice_ids = apply_slice(all_ids, slice_spec)
    registry = registry_for(config, run_id=run_id)
    reserved = registry.reserved_ad_ids()
    ad_ids = [aid for aid in slice_ids if aid not in reserved]
    if limit is not None:
        ad_ids = ad_ids[:limit]
    if not ad_ids:
        return SubmitResult(skipped=True, skipped_reason="no_ads_to_submit")

    by_id = load_source_ads_by_id(config, ad_ids)
    ads = [by_id[aid] for aid in ad_ids if aid in by_id]
    if not ads:
        return SubmitResult(skipped=True, skipped_reason="no_source_ads_resolved")

    bundle = get_bundle(config.skill)
    ads, dropped_in_prepare = bundle.prepare_source_ads(ads, config)
    if dropped_in_prepare:
        import logging

        logging.getLogger("draper").warning(
            "%s submit: dropped %d ad(s) in prepare_source_ads (skill=%s); first few: %s",
            config.skill,
            len(dropped_in_prepare),
            config.skill,
            dropped_in_prepare[:5],
        )
    if not ads:
        return SubmitResult(skipped=True, skipped_reason="no_ads_after_prepare")

    requests = [
        bundle.build_request(
            ad,
            model=model,
            max_tokens=provider_cfg.max_tokens,
            temperature=provider_cfg.temperature,
        )
        for ad in ads
    ]
    sidecars = [
        PendingBatchSidecar(custom_id=r.custom_id, prompt_index=i, source_ad_ids=[ad.ad_id])
        for i, (r, ad) in enumerate(zip(requests, ads, strict=True))
    ]
    try:
        info = await make_batch_client(model).submit(requests)
    except Exception as exc:
        # Submission failed before the batch was accepted; nothing to clean up.
        # The error is re-raised immediately — don't create a registry entry.
        raise exc from exc
    try:
        registry.add(
            PendingBatch(
                batch_id=info.batch_id,
                provider=info.provider,
                model=model,
                task_format=task_format_for(config.skill),
                submitted_at=utc_now_iso(),
                status=info.status.value,
                request_count=len(requests),
                sidecars=sidecars,
            )
        )
    except Exception as exc:  # noqa: BLE001 — registry write failure is provider-agnostic
        # Registry add failed after the batch was submitted. Log a loud warning
        # so the operator knows to manually register the batch or cancel it.
        import logging

        log = logging.getLogger("draper")
        log.error(
            "Failed to register batch %s (provider=%s, model=%s). "
            "The batch was accepted upstream but is not in the registry. "
            "Manually add it via registry_for().add() or cancel it.",
            info.batch_id,
            info.provider,
            model,
            exc_info=exc,
        )
        raise exc from exc
    if run_id:
        _record_run_submission(
            config,
            run_id=run_id,
            provider=info.provider,
            model=model,
            batch_id=info.batch_id,
            request_count=len(requests),
            slice_spec=slice_spec,
        )
    return SubmitResult(
        batch_id=info.batch_id,
        provider=info.provider,
        model=model,
        request_count=len(requests),
    )


def _record_run_submission(
    config: ConstructionV2Config,
    *,
    run_id: str,
    provider: str,
    model: str,
    batch_id: str,
    request_count: int,
    slice_spec: str,
) -> None:
    """Append a row to ``runs/<run_id>/submissions.json`` + manifest if new."""
    base = runs_dir(config, run_id)
    base.mkdir(parents=True, exist_ok=True)
    subs_path = base / "submissions.json"
    rows: list[dict[str, Any]] = []
    if subs_path.exists():
        try:
            rows = json.loads(subs_path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                rows = []
        except json.JSONDecodeError:
            rows = []
    rows.append(
        {
            "run_id": run_id,
            "provider": provider,
            "model": model,
            "batch_id": batch_id,
            "request_count": request_count,
            "slice_spec": slice_spec,
            "submitted_at": utc_now_iso(),
        }
    )
    subs_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    mp = manifest_path(config, run_id)
    if not mp.exists():
        mp.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "started_at": utc_now_iso(),
                    "selection_lineage_hash": _read_selection_lineage(config),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


class CollectResult(BaseModel):
    """Outcome of collecting one batch's results."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str
    terminal: bool = False
    status: str = ""
    briefs_written: int = 0
    rationales_written: int = 0
    parse_failures: int = 0
    provider_errors: int = 0
    missing: bool = False
    stuck_cancelled: bool = False


def _parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


async def _check_stuck(
    pending: PendingBatch,
    *,
    status: str,
    batch_cfg: BatchConfig,
) -> bool:
    """Return True if ``pending`` exceeds the stuck timeout."""
    if status in {"completed", "failed", "cancelled", "expired"}:
        return False
    submitted = _parse_iso(pending.submitted_at)
    if submitted is None:
        return False
    if submitted.tzinfo is None:
        submitted = submitted.replace(tzinfo=UTC)
    elapsed = datetime.now(UTC) - submitted
    return elapsed.total_seconds() > batch_cfg.stuck_timeout_minutes * 60


async def collect_batch(
    config: ConstructionV2Config,
    batch_id: str,
    *,
    run_id: str | None = None,
) -> CollectResult:
    """Poll a batch, persist single-pass results, remove on terminal success.

    Single-pass results land in two files:

    - ``briefs.jsonl`` — ``{ad_id, brief}`` rows from each ``<brief>`` region.
    - ``responses_raw.jsonl`` — ``{ad_id, content, model, batch_id}`` rows
      reassembling ``<think>`` + deliverable so :func:`ingest_responses`
      runs unchanged.

    Provider-side errors (non-empty ``r.error``) are counted into
    ``CollectResult.provider_errors``. When their fraction of the
    request count exceeds ``config.batch.max_partial_error_rate`` the
    function raises :class:`PartialFailureThreshold` AFTER persisting
    successful rows, so the operator gets both the data and the alert.
    """
    import logging

    log = logging.getLogger("draper")
    registry = registry_for(config, run_id=run_id)
    pending = registry.get(batch_id)
    if pending is None:
        return CollectResult(batch_id=batch_id, missing=True)
    client = make_batch_client(pending.model)
    info = await client.poll(batch_id)
    registry.update_status(
        batch_id,
        status=info.status.value,
        completed_count=info.completed_count,
        failed_count=info.failed_count,
        request_count=info.request_count,
    )

    if (
        await _check_stuck(pending, status=info.status.value, batch_cfg=config.batch)
        and config.batch.auto_force_cancel
    ):
        log.warning(
            "Batch %s stuck (status=%s, submitted_at=%s); cancelling.",
            batch_id,
            info.status.value,
            pending.submitted_at,
        )
        try:
            await client.cancel(batch_id)
        except Exception as exc:  # noqa: BLE001 — provider-specific failure modes
            log.warning("Cancel failed for stuck batch %s: %s", batch_id, exc)
        registry.update_status(batch_id, status="cancelled_stuck")
        registry.remove(batch_id)
        return CollectResult(
            batch_id=batch_id,
            terminal=True,
            status="cancelled_stuck",
            stuck_cancelled=True,
        )

    if not info.is_terminal:
        return CollectResult(batch_id=batch_id, terminal=False, status=info.status.value)

    results = await client.fetch_results(batch_id)
    bundle = get_bundle(config.skill)
    brief_rows: list[dict[str, Any]] = []
    rationale_rows: list[dict[str, Any]] = []
    parse_failures = 0
    provider_errors = 0
    parse_rejections: list[RejectionRecord] = []
    for r in results:
        if r.error:
            provider_errors += 1
            continue
        if not r.custom_id.startswith(CUSTOM_ID_PREFIX):
            log.warning(
                "Result %s has unexpected custom_id prefix; skipping.",
                r.custom_id,
            )
            continue
        sidecar = pending.sidecar_by_custom_id(r.custom_id)
        if sidecar is None or not sidecar.source_ad_ids:
            log.warning("Result %s has no matching sidecar; skipping.", r.custom_id)
            continue
        ad_id = sidecar.source_ad_ids[0]
        parsed = bundle.parse_response(r.content)
        if parsed.brief is None or parsed.think is None or parsed.deliverable is None:
            parse_failures += 1
            reason = "; ".join(parsed.errors) or "single_pass_parse_failed"
            parse_rejections.append(RejectionRecord(ad_id=ad_id, stage="parse", reason=reason))
            continue
        brief_rows.append({"ad_id": ad_id, "brief": parsed.brief})
        # Reassemble think+deliverable as the canonical content string so
        # `ingest_responses.parse_response` round-trips it identically to
        # a two-stage rationale response.
        content = f"<think>\n{parsed.think}\n</think>\n\n{parsed.deliverable}"
        rationale_rows.append(
            {
                "ad_id": ad_id,
                "content": content,
                "model": r.model,
                "batch_id": batch_id,
            }
        )
    if brief_rows:
        append_jsonl(brief_rows, briefs_path(config, run_id=run_id))
    if rationale_rows:
        append_jsonl(rationale_rows, responses_path(config, run_id=run_id))
    if parse_rejections:
        append_jsonl(
            [r.model_dump(mode="json") for r in parse_rejections],
            audit_path(config, "collect_rejections.jsonl", run_id=run_id),
        )
    registry.remove(batch_id)
    result = CollectResult(
        batch_id=batch_id,
        terminal=True,
        status=info.status.value,
        briefs_written=len(brief_rows),
        rationales_written=len(rationale_rows),
        parse_failures=parse_failures,
        provider_errors=provider_errors,
    )
    request_count = info.request_count or pending.request_count or 1
    error_rate = provider_errors / max(request_count, 1)
    if error_rate > config.batch.max_partial_error_rate:
        msg = (
            f"Batch {batch_id}: provider_errors={provider_errors} / "
            f"{request_count} = {error_rate:.1%} exceeds threshold "
            f"{config.batch.max_partial_error_rate:.1%}. "
            f"Persisted rows: briefs={len(brief_rows)} "
            f"rationales={len(rationale_rows)} parse_failures={parse_failures}."
        )
        raise PartialFailureThreshold(msg)
    return result


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class IngestStats(BaseModel):
    """Counters for one ingest pass."""

    model_config = ConfigDict(extra="forbid")

    total_input: int = 0
    passed: int = 0
    teacher_failed: int = 0
    parse_failed: int = 0
    leak_failed: int = 0
    fidelity_failed: int = 0
    grounding_failed: int = 0
    labels_failed: int = 0
    content_bridge_failed: int = 0
    missing_brief: int = 0
    missing_ad: int = 0


class IngestResult(BaseModel):
    """Outcome of one ingest pass — examples + rejections + counters."""

    model_config = ConfigDict(extra="forbid")

    examples: list[ExampleRecord] = Field(default_factory=list)
    rejections: list[RejectionRecord] = Field(default_factory=list)
    stats: IngestStats = Field(default_factory=IngestStats)


def ingest_responses(
    config: ConstructionV2Config,
    *,
    input_path: Path | None = None,
    run_id: str | None = None,
) -> IngestResult:
    """Parse raw teacher responses + run leak / fidelity / grounding gates.

    Writes ``examples.jsonl`` and appends rejections to the audit log.
    Returns the in-memory result so callers can render their own report.
    """
    in_path = input_path if input_path is not None else responses_path(config, run_id=run_id)
    if not in_path.exists():
        msg = f"no responses to ingest at {in_path}"
        raise FileNotFoundError(msg)
    briefs_by_id = load_briefs(config, run_id=run_id)
    rows = read_jsonl(in_path)
    ad_ids_set: set[str] = set()
    for r in rows:
        aid = r.get("ad_id")
        if isinstance(aid, str):
            ad_ids_set.add(aid)
    ad_ids = sorted(ad_ids_set)
    ads_by_id = load_source_ads_by_id(config, [aid for aid in ad_ids if isinstance(aid, str)])

    result = IngestResult()
    result.stats.total_input = len(rows)

    for row in rows:
        ad_id = row.get("ad_id")
        if not isinstance(ad_id, str):
            continue
        brief_dict = briefs_by_id.get(ad_id)
        ad = ads_by_id.get(ad_id)
        if brief_dict is None:
            result.stats.missing_brief += 1
            result.rejections.append(
                RejectionRecord(ad_id=ad_id, stage="parse", reason="missing_brief")
            )
            continue
        if ad is None:
            result.stats.missing_ad += 1
            result.rejections.append(
                RejectionRecord(ad_id=ad_id, stage="parse", reason="missing_source_ad")
            )
            continue
        content = row.get("content") or ""
        parsed = parse_response(content)
        if isinstance(parsed, ParseRejection):
            if parsed == ParseRejection.TEACHER_FAILED:
                result.stats.teacher_failed += 1
            else:
                result.stats.parse_failed += 1
            result.rejections.append(
                RejectionRecord(ad_id=ad_id, stage="parse", reason=parsed.value)
            )
            continue
        if not isinstance(parsed, ParsedResponse):
            continue

        bundle = get_bundle(config.skill)

        # Validate the cached brief into the skill's model, injecting any
        # field the teacher does not author (image_brief's verbatim copy).
        try:
            brief = bundle.build_brief(brief_dict, ad)
        except (ValueError, TypeError) as e:
            result.stats.missing_brief += 1
            result.rejections.append(
                RejectionRecord(ad_id=ad_id, stage="parse", reason=f"invalid_brief:{e}")
            )
            continue

        # Leak guard — skipped for skills where the copy is a legitimate
        # verbatim brief input (image_brief sets bundle.leak = None).
        if bundle.leak is not None:
            leak = bundle.leak(brief, ad, n=config.single_pass.forbid_ngram_overlap)
            if not leak.passed:
                result.stats.leak_failed += 1
                result.rejections.append(
                    RejectionRecord(
                        ad_id=ad_id,
                        stage="leak",
                        reason=f"{leak.reason}:{leak.offending_field}",
                    )
                )
                continue

        fidelity = bundle.fidelity(parsed.deliverable, ad)
        if not fidelity.passed:
            result.stats.fidelity_failed += 1
            result.rejections.append(
                RejectionRecord(ad_id=ad_id, stage="fidelity", reason=fidelity.reason)
            )
            continue

        grounding = bundle.grounding(parsed.think, brief)
        if not grounding.passed:
            result.stats.grounding_failed += 1
            result.rejections.append(
                RejectionRecord(ad_id=ad_id, stage="grounding", reason=grounding.reason)
            )
            continue

        # Content-bridge gate — verifies the brief's factual content bridge is
        # grounded in the caption and consistent with the deliverable. Skipped
        # for skills with no content bridge (copywriting sets it None).
        if bundle.content_bridge is not None:
            content_bridge = bundle.content_bridge(brief, parsed.deliverable, ad)
            if not content_bridge.passed:
                result.stats.content_bridge_failed += 1
                result.rejections.append(
                    RejectionRecord(
                        ad_id=ad_id,
                        stage="content_bridge",
                        reason=f"{content_bridge.reason}:{content_bridge.detail}",
                    )
                )
                continue

        if bundle.labels is not None:
            labels = bundle.labels(parsed.deliverable, ad)
            if not labels.passed:
                result.stats.labels_failed += 1
                result.rejections.append(
                    RejectionRecord(ad_id=ad_id, stage="labels", reason=labels.reason)
                )
                continue

        result.examples.append(
            ExampleRecord(
                example_id=f"v2-{uuid.uuid4().hex[:12]}",
                ad_id=ad_id,
                platform=brief.platform,
                brief=brief.model_dump(mode="json"),
                think=parsed.think,
                deliverable=parsed.deliverable,
                fidelity_coverage=fidelity.coverage,
                fidelity_signature_passed=fidelity.signature_passed,
                teacher_model=str(row.get("model") or ""),
                batch_id=str(row.get("batch_id") or ""),
            )
        )

    result.stats.passed = len(result.examples)

    write_jsonl(
        [e.model_dump(mode="json") for e in result.examples],
        examples_path(config, run_id=run_id),
    )
    append_jsonl(
        [r.model_dump(mode="json") for r in result.rejections],
        audit_path(config, "ingest_rejections.jsonl", run_id=run_id),
    )
    return result
