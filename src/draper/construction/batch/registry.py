"""Per-format registry of submitted batch jobs.

State needed to survive across CLI invocations:

- For each batch: ``batch_id``, provider, model, task format, submission
  timestamp, last-known status, request count.
- For each request within that batch: a sidecar with the full rolled-dice
  metadata (persona, seed, evol op, difficulty, turn structure, declared
  provider, source ad IDs) keyed by the OpenAI-style ``custom_id``.

Stored at ``data/constructed/<format>/_pending_batches.json``. Completed
batches are flushed out during ``batch-collect`` once their results are
ingested — keeping the file scoped to in-flight work.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from draper.construction.batch.types import BatchStatus


@dataclass
class PendingBatchSidecar:
    """Per-request dice-roll metadata needed to ingest results later."""

    custom_id: str
    prompt_index: int
    source_ad_ids: list[str] = field(default_factory=list)
    prompt_style: str = "data_grounded"
    persona_id: str = ""
    seed_idx: int = -1
    evol_op: str = ""
    difficulty: str = "standard"
    turn_structure: str = "single"
    followup_type: str = ""
    provider_label: str = ""
    # Copywriting-specific context derived from the source ad (no RNG).
    # Empty for other formats.
    source_ad_shape: str = ""
    # Copywriting-specific strongly-enforced conversation register
    # (``conversational`` / ``structured`` / ``imperative``). Governs the
    # register of *both* the user brief and the assistant response. Empty
    # for other formats and for older rows.
    conversation_register: str = ""
    # Exact bundle text sent to the teacher provider. Preserved through the
    # registry so batch-collect can persist it alongside the saved example
    # for prompt-diagnostic review. Empty string on pre-feature rows.
    teacher_bundle: str = ""


@dataclass
class PendingBatch:
    """A single in-flight batch job + its per-request sidecars."""

    batch_id: str
    provider: str
    model: str
    task_format: str
    submitted_at: str
    status: str = BatchStatus.PENDING.value
    request_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    sidecars: list[PendingBatchSidecar] = field(default_factory=list)

    def sidecar_by_custom_id(self, custom_id: str) -> PendingBatchSidecar | None:
        """Look up a single sidecar by the request's custom_id."""
        for s in self.sidecars:
            if s.custom_id == custom_id:
                return s
        return None


class BatchRegistry:
    """File-backed list of pending batches for one task format."""

    def __init__(self, output_dir: str | Path, task_format: str) -> None:
        self._path = Path(output_dir) / task_format / "_pending_batches.json"
        self._task_format = task_format
        self._batches: list[PendingBatch] = []
        if self._path.exists():
            self._batches = self._load()

    @property
    def path(self) -> Path:
        return self._path

    def all(self) -> list[PendingBatch]:
        """Return all tracked batches (in-memory, no disk read)."""
        return list(self._batches)

    def pending(self) -> list[PendingBatch]:
        """Return only batches that haven't reached a terminal state."""
        terminal = {
            BatchStatus.COMPLETED.value,
            BatchStatus.FAILED.value,
            BatchStatus.CANCELLED.value,
            BatchStatus.EXPIRED.value,
        }
        return [b for b in self._batches if b.status not in terminal]

    def get(self, batch_id: str) -> PendingBatch | None:
        for b in self._batches:
            if b.batch_id == batch_id:
                return b
        return None

    def add(self, batch: PendingBatch) -> None:
        """Append a newly submitted batch and persist."""
        self._batches.append(batch)
        self._save()

    def update_status(
        self,
        batch_id: str,
        *,
        status: str,
        completed_count: int | None = None,
        failed_count: int | None = None,
        request_count: int | None = None,
    ) -> None:
        """Patch the status of an existing batch and persist."""
        for b in self._batches:
            if b.batch_id == batch_id:
                b.status = status
                if completed_count is not None:
                    b.completed_count = completed_count
                if failed_count is not None:
                    b.failed_count = failed_count
                if request_count is not None:
                    b.request_count = request_count
                break
        self._save()

    def remove(self, batch_id: str) -> None:
        """Drop a batch entry (used after successful ingest or cancel)."""
        self._batches = [b for b in self._batches if b.batch_id != batch_id]
        self._save()

    def _active(self) -> list[PendingBatch]:
        """Batches that are in-flight OR completed-but-not-yet-ingested.

        ``completed`` batches stay in the registry until ``batch-collect``
        removes them after ingest. They must be treated as active for
        reservation purposes: if ``batch-list`` polls a batch to
        ``completed`` before ``batch-collect`` runs, those ad IDs are not
        yet in the per-format JSONL, so without this guard a concurrent
        ``batch-submit`` could draw the same ads and roll the same dice.

        Failed / cancelled / expired batches are excluded — their work will
        never land in the JSONL, so their ads are free to reuse.
        """
        inactive = {
            BatchStatus.FAILED.value,
            BatchStatus.CANCELLED.value,
            BatchStatus.EXPIRED.value,
        }
        return [b for b in self._batches if b.status not in inactive]

    def reserved_ad_ids(self) -> set[str]:
        """Ad IDs claimed by in-flight or completed-but-not-ingested batches.

        Used by ``batch-submit`` to exclude already-reserved ads from the
        next submission so two overlapping batches never draw the same ads.
        """
        ids: set[str] = set()
        for b in self._active():
            for s in b.sidecars:
                ids.update(s.source_ad_ids)
        return ids

    def reserved_bundle_fingerprints(self) -> frozenset[frozenset[str]]:
        """Bundle-level fingerprints reserved by active batches.

        A "fingerprint" is the ``frozenset`` of source ad IDs for one
        bundle. Passed to ``SourceSelector.select_batches`` so the next
        submission can never emit a bundle with the same ad-set as one
        already in flight — individual-ad reuse is allowed, duplicate
        ad-sets are not.
        """
        fps: set[frozenset[str]] = set()
        for b in self._active():
            for s in b.sidecars:
                if s.source_ad_ids:
                    fps.add(frozenset(s.source_ad_ids))
        return frozenset(fps)

    def pending_request_count(self) -> int:
        """Request count across all in-flight or completed-but-not-ingested batches.

        Used to advance the RNG offset in ``prepare_bundles`` so two
        submissions before any ingest roll different dice.
        """
        return sum(b.request_count for b in self._active())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> list[PendingBatch]:
        with self._path.open() as f:
            data = json.load(f)
        out: list[PendingBatch] = []
        for entry in data:
            sidecars = [
                PendingBatchSidecar(**_migrate_sidecar_dict(s)) for s in entry.get("sidecars", [])
            ]
            out.append(
                PendingBatch(
                    batch_id=entry["batch_id"],
                    provider=entry["provider"],
                    model=entry["model"],
                    task_format=entry["task_format"],
                    submitted_at=entry["submitted_at"],
                    status=entry.get("status", BatchStatus.PENDING.value),
                    request_count=int(entry.get("request_count", 0)),
                    completed_count=int(entry.get("completed_count", 0)),
                    failed_count=int(entry.get("failed_count", 0)),
                    sidecars=sidecars,
                )
            )
        return out

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        serializable: list[dict[str, Any]] = [asdict(b) for b in self._batches]
        with self._path.open("w") as f:
            json.dump(serializable, f, indent=2)


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (for submission timestamps)."""
    return datetime.now(UTC).isoformat()


def _migrate_sidecar_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Translate legacy sidecar JSON keys to the current schema.

    The brief-tone-axes feature originally shipped the field as
    ``brief_register``; it was promoted to ``conversation_register`` after
    the axis was expanded to govern both sides of the turn. Pending
    batches submitted under the old name (``_pending_batches.json`` rows
    written before the rename) carry the legacy key — translate it on
    load so those in-flight batches ingest cleanly. Prefer the new key if
    both are somehow present.
    """
    if "brief_register" not in raw:
        return raw
    migrated = dict(raw)
    legacy = migrated.pop("brief_register")
    migrated.setdefault("conversation_register", legacy)
    return migrated
