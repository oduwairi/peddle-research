"""Poll the single-pass smoke batches until all are terminal.

Usage::

    python scripts/explore/poll_single_pass_smoke.py \\
        --submissions data/constructed_v2/smoke/single_pass/submissions.json \\
        --interval 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Ensure ``src/`` is importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from draper.construction.batch import make_batch_client  # noqa: E402
from draper.construction.batch.types import BatchStatus  # noqa: E402

TERMINAL = {
    BatchStatus.COMPLETED,
    BatchStatus.FAILED,
    BatchStatus.CANCELLED,
    BatchStatus.EXPIRED,
}


async def _poll_once(subs: list[dict[str, str]]) -> list[BatchStatus]:
    statuses: list[BatchStatus] = []
    for s in subs:
        client = make_batch_client(s["model"])
        info = await client.poll(s["batch_id"])
        statuses.append(info.status)
        print(
            f"  {s['label']:10s} ({s['model']:35s}) → "
            f"{info.status.value:15s} "
            f"completed={info.completed_count}/{info.request_count}"
        )
    return statuses


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--submissions",
        default="data/constructed_v2/smoke/single_pass/submissions.json",
    )
    parser.add_argument("--interval", type=int, default=30, help="Poll interval (seconds).")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    subs = json.loads(Path(args.submissions).read_text(encoding="utf-8"))
    start = time.time()
    while True:
        print(f"[t={int(time.time() - start)}s]")
        statuses = await _poll_once(subs)
        if all(s in TERMINAL for s in statuses):
            print(
                "\nAll batches terminal. Final statuses: "
                + ", ".join(s.value for s in statuses)
            )
            return 0 if all(s == BatchStatus.COMPLETED for s in statuses) else 1
        print(f"  -- not all terminal; sleeping {args.interval}s --\n")
        await asyncio.sleep(args.interval)


def main() -> int:
    return asyncio.run(_run(_parse_args(sys.argv[1:])))


if __name__ == "__main__":
    sys.exit(main())
