"""Tests for BatchRegistry reservation + persistence.

Exercises the three interfaces prepare_bundles relies on for parallel
batch safety:

- ``reserved_ad_ids()`` — ad IDs claimed by in-flight or
  completed-but-not-ingested batches.
- ``reserved_bundle_fingerprints()`` — frozenset of ad-ID sets, so two
  overlapping batches never emit bundles with the same source-ad set.
- ``pending_request_count()`` — total request count for RNG offset.

Also covers persistence round-trip (write → reload → identical state)
and correct handling of terminal-vs-active status for each method.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from draper.construction.batch.registry import (
    BatchRegistry,
    PendingBatch,
    PendingBatchSidecar,
    utc_now_iso,
)
from draper.construction.batch.types import BatchStatus


def _sidecar(custom_id: str, ad_ids: list[str]) -> PendingBatchSidecar:
    return PendingBatchSidecar(
        custom_id=custom_id,
        prompt_index=int(custom_id.rsplit("-", 1)[-1]) if "-" in custom_id else 0,
        source_ad_ids=ad_ids,
        prompt_style="data_grounded",
        persona_id="smb_founder_first_ads",
        seed_idx=0,
        evol_op="",
        difficulty="standard",
        turn_structure="single",
        followup_type="",
        provider_label="gpt",
    )


def _batch(
    batch_id: str,
    task_format: str,
    sidecars: list[PendingBatchSidecar],
    *,
    status: str = BatchStatus.IN_PROGRESS.value,
) -> PendingBatch:
    return PendingBatch(
        batch_id=batch_id,
        provider="openai",
        model="gpt-4o-mini",
        task_format=task_format,
        submitted_at=utc_now_iso(),
        status=status,
        request_count=len(sidecars),
        sidecars=sidecars,
    )


class TestReservedAdIds:
    def test_empty_registry_returns_empty_set(self, tmp_path: Path) -> None:
        reg = BatchRegistry(tmp_path, "positioning")
        assert reg.reserved_ad_ids() == set()

    def test_single_active_batch_reserves_its_ads(self, tmp_path: Path) -> None:
        reg = BatchRegistry(tmp_path, "positioning")
        reg.add(
            _batch(
                "b1",
                "positioning",
                [
                    _sidecar("positioning-00000", ["ad-a", "ad-b", "ad-c"]),
                    _sidecar("positioning-00001", ["ad-d", "ad-e"]),
                ],
            )
        )
        assert reg.reserved_ad_ids() == {"ad-a", "ad-b", "ad-c", "ad-d", "ad-e"}

    def test_completed_batch_still_reserves_until_removed(
        self, tmp_path: Path
    ) -> None:
        reg = BatchRegistry(tmp_path, "positioning")
        reg.add(
            _batch(
                "b1",
                "positioning",
                [_sidecar("positioning-00000", ["ad-a"])],
                status=BatchStatus.COMPLETED.value,
            )
        )
        # Completed but not yet ingested → ads stay reserved so a parallel
        # submit doesn't draw them and collide with the about-to-ingest batch.
        assert reg.reserved_ad_ids() == {"ad-a"}

    @pytest.mark.parametrize(
        "terminal_status",
        [
            BatchStatus.FAILED.value,
            BatchStatus.CANCELLED.value,
            BatchStatus.EXPIRED.value,
        ],
    )
    def test_terminal_failed_batches_release_reservation(
        self, tmp_path: Path, terminal_status: str
    ) -> None:
        reg = BatchRegistry(tmp_path, "positioning")
        reg.add(
            _batch(
                "b1",
                "positioning",
                [_sidecar("positioning-00000", ["ad-a"])],
                status=terminal_status,
            )
        )
        # Work will never land in the JSONL → ad is free to reuse.
        assert reg.reserved_ad_ids() == set()

    def test_multiple_active_batches_union(self, tmp_path: Path) -> None:
        reg = BatchRegistry(tmp_path, "positioning")
        reg.add(
            _batch(
                "b1", "positioning", [_sidecar("positioning-00000", ["ad-a", "ad-b"])]
            )
        )
        reg.add(
            _batch(
                "b2", "positioning", [_sidecar("positioning-00001", ["ad-b", "ad-c"])]
            )
        )
        assert reg.reserved_ad_ids() == {"ad-a", "ad-b", "ad-c"}


class TestReservedBundleFingerprints:
    def test_empty_registry_returns_empty_frozenset(self, tmp_path: Path) -> None:
        reg = BatchRegistry(tmp_path, "positioning")
        assert reg.reserved_bundle_fingerprints() == frozenset()

    def test_bundle_fingerprint_is_ad_id_set(self, tmp_path: Path) -> None:
        reg = BatchRegistry(tmp_path, "positioning")
        reg.add(
            _batch(
                "b1",
                "positioning",
                [_sidecar("positioning-00000", ["ad-a", "ad-b", "ad-c"])],
            )
        )
        fps = reg.reserved_bundle_fingerprints()
        assert len(fps) == 1
        assert frozenset({"ad-a", "ad-b", "ad-c"}) in fps

    def test_identical_ad_sets_collapse_to_one_fingerprint(
        self, tmp_path: Path
    ) -> None:
        # Two bundles with the same ad set across batches produce one fingerprint.
        reg = BatchRegistry(tmp_path, "positioning")
        reg.add(
            _batch(
                "b1", "positioning", [_sidecar("positioning-00000", ["ad-a", "ad-b"])]
            )
        )
        reg.add(
            _batch(
                "b2", "positioning", [_sidecar("positioning-00001", ["ad-b", "ad-a"])]
            )
        )
        assert reg.reserved_bundle_fingerprints() == frozenset(
            {frozenset({"ad-a", "ad-b"})}
        )

    def test_different_ad_sets_produce_distinct_fingerprints(
        self, tmp_path: Path
    ) -> None:
        reg = BatchRegistry(tmp_path, "positioning")
        reg.add(
            _batch(
                "b1",
                "positioning",
                [
                    _sidecar("positioning-00000", ["ad-a", "ad-b"]),
                    _sidecar("positioning-00001", ["ad-c", "ad-d"]),
                ],
            )
        )
        fps = reg.reserved_bundle_fingerprints()
        assert len(fps) == 2
        assert frozenset({"ad-a", "ad-b"}) in fps
        assert frozenset({"ad-c", "ad-d"}) in fps

    def test_empty_ad_list_is_not_a_fingerprint(self, tmp_path: Path) -> None:
        reg = BatchRegistry(tmp_path, "positioning")
        reg.add(_batch("b1", "positioning", [_sidecar("positioning-00000", [])]))
        assert reg.reserved_bundle_fingerprints() == frozenset()


class TestPendingRequestCount:
    def test_empty_registry_returns_zero(self, tmp_path: Path) -> None:
        reg = BatchRegistry(tmp_path, "positioning")
        assert reg.pending_request_count() == 0

    def test_sums_active_request_counts(self, tmp_path: Path) -> None:
        reg = BatchRegistry(tmp_path, "positioning")
        reg.add(
            _batch(
                "b1",
                "positioning",
                [
                    _sidecar("positioning-00000", ["a"]),
                    _sidecar("positioning-00001", ["b"]),
                ],
            )
        )
        reg.add(_batch("b2", "positioning", [_sidecar("positioning-00002", ["c"])]))
        assert reg.pending_request_count() == 3

    def test_excludes_terminal_failed_batches(self, tmp_path: Path) -> None:
        reg = BatchRegistry(tmp_path, "positioning")
        reg.add(_batch("b1", "positioning", [_sidecar("positioning-00000", ["a"])]))
        reg.add(
            _batch(
                "b2",
                "positioning",
                [_sidecar("positioning-00001", ["b"])],
                status=BatchStatus.FAILED.value,
            )
        )
        # Only the active one counts.
        assert reg.pending_request_count() == 1

    def test_includes_completed_but_not_ingested(self, tmp_path: Path) -> None:
        reg = BatchRegistry(tmp_path, "positioning")
        reg.add(
            _batch(
                "b1",
                "positioning",
                [_sidecar("positioning-00000", ["a"])],
                status=BatchStatus.COMPLETED.value,
            )
        )
        # Until batch-collect removes it, the RNG offset must reflect it.
        assert reg.pending_request_count() == 1


class TestPersistenceRoundTrip:
    def test_add_then_reload_preserves_state(self, tmp_path: Path) -> None:
        reg = BatchRegistry(tmp_path, "positioning")
        sidecars = [
            _sidecar("positioning-00000", ["ad-a", "ad-b"]),
            _sidecar("positioning-00001", ["ad-c"]),
        ]
        reg.add(_batch("b1", "positioning", sidecars))

        reloaded = BatchRegistry(tmp_path, "positioning")
        assert len(reloaded.all()) == 1
        batch = reloaded.get("b1")
        assert batch is not None
        assert batch.request_count == 2
        assert batch.task_format == "positioning"
        assert [s.custom_id for s in batch.sidecars] == [
            "positioning-00000",
            "positioning-00001",
        ]
        assert batch.sidecars[0].source_ad_ids == ["ad-a", "ad-b"]

    def test_update_status_persists(self, tmp_path: Path) -> None:
        reg = BatchRegistry(tmp_path, "positioning")
        reg.add(_batch("b1", "positioning", [_sidecar("positioning-00000", ["a"])]))
        reg.update_status(
            "b1",
            status=BatchStatus.COMPLETED.value,
            completed_count=1,
            failed_count=0,
        )

        reloaded = BatchRegistry(tmp_path, "positioning")
        batch = reloaded.get("b1")
        assert batch is not None
        assert batch.status == BatchStatus.COMPLETED.value
        assert batch.completed_count == 1

    def test_remove_persists(self, tmp_path: Path) -> None:
        reg = BatchRegistry(tmp_path, "positioning")
        reg.add(_batch("b1", "positioning", [_sidecar("positioning-00000", ["a"])]))
        reg.remove("b1")

        reloaded = BatchRegistry(tmp_path, "positioning")
        assert reloaded.all() == []
        assert reloaded.reserved_ad_ids() == set()

    def test_per_format_isolation(self, tmp_path: Path) -> None:
        """Registries for different formats are stored at different paths."""
        reg_pos = BatchRegistry(tmp_path, "positioning")
        reg_cw = BatchRegistry(tmp_path, "copywriting")
        reg_pos.add(
            _batch("p1", "positioning", [_sidecar("positioning-00000", ["a"])])
        )
        assert reg_cw.all() == []

        reloaded_cw = BatchRegistry(tmp_path, "copywriting")
        assert reloaded_cw.all() == []
        reloaded_pos = BatchRegistry(tmp_path, "positioning")
        assert len(reloaded_pos.all()) == 1


class TestPendingVsAll:
    def test_all_includes_terminal(self, tmp_path: Path) -> None:
        reg = BatchRegistry(tmp_path, "positioning")
        reg.add(
            _batch(
                "b1",
                "positioning",
                [_sidecar("positioning-00000", ["a"])],
                status=BatchStatus.FAILED.value,
            )
        )
        reg.add(
            _batch(
                "b2",
                "positioning",
                [_sidecar("positioning-00001", ["b"])],
                status=BatchStatus.PENDING.value,
            )
        )
        assert {b.batch_id for b in reg.all()} == {"b1", "b2"}

    def test_pending_excludes_terminal(self, tmp_path: Path) -> None:
        reg = BatchRegistry(tmp_path, "positioning")
        reg.add(
            _batch(
                "b1",
                "positioning",
                [_sidecar("positioning-00000", ["a"])],
                status=BatchStatus.FAILED.value,
            )
        )
        reg.add(
            _batch(
                "b2",
                "positioning",
                [_sidecar("positioning-00001", ["b"])],
                status=BatchStatus.PENDING.value,
            )
        )
        pending_ids = {b.batch_id for b in reg.pending()}
        assert pending_ids == {"b2"}
