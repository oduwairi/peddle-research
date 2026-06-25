"""Tail trackio training metrics from its SQLite db.

When ``report_to=trackio``, TRL routes per-step losses to SQLite — they do
NOT appear in stdout/log. This CLI replaces the postmortem's hand-escaped
one-liner with a proper command that works locally and over SSH.

Usage:
    uv run python scripts/tail_train_metrics.py
    uv run python scripts/tail_train_metrics.py --last 50
    uv run python scripts/tail_train_metrics.py --follow
    uv run python scripts/tail_train_metrics.py --run smoke
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_DB = Path.home() / ".cache/huggingface/trackio/huggingface.db"


def _decode_metrics(raw: Any) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return dict(json.loads(raw))


def _latest_run(con: sqlite3.Connection) -> str | None:
    cur = con.execute("SELECT run_name FROM metrics ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    return str(row[0]) if row else None


def _fetch(
    con: sqlite3.Connection, run_substr: str, last: int
) -> list[tuple[int, dict[str, Any]]]:
    rows = list(
        con.execute(
            "SELECT step, metrics FROM metrics WHERE run_name LIKE ? "
            "ORDER BY id DESC LIMIT ?",
            (f"%{run_substr}%", last),
        )
    )
    return [(int(s), _decode_metrics(m)) for s, m in reversed(rows)]


def _fmt(v: Any, prec: int = 4) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.{prec}g}"
    except (TypeError, ValueError):
        return str(v)


def _render(rows: list[tuple[int, dict[str, Any]]], run_name: str) -> Table:
    table = Table(title=f"Trackio metrics — run={run_name}", show_lines=False)
    table.add_column("STEP", justify="right")
    table.add_column("TRAIN_LOSS", justify="right")
    table.add_column("GRAD_NORM", justify="right")
    table.add_column("LR", justify="right")
    table.add_column("EPOCH", justify="right")
    table.add_column("EVAL_LOSS", justify="right")
    for step, m in rows:
        table.add_row(
            str(step),
            _fmt(m.get("train/loss")),
            _fmt(m.get("train/grad_norm")),
            _fmt(m.get("train/learning_rate"), prec=3),
            _fmt(m.get("train/epoch") or m.get("epoch"), prec=3),
            _fmt(m.get("eval/loss")),
        )
    return table


def tail(
    db: Path = typer.Option(  # noqa: B008
        DEFAULT_DB, "--db", help="Trackio SQLite db path."
    ),
    last: int = typer.Option(20, "--last", "-n", help="Show this many recent rows."),
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Poll for new rows every 2s."
    ),
    run: str = typer.Option(
        "", "--run", help="Filter by run_name substring. Default: most recent run."
    ),
) -> None:
    """Print the last N metric rows; with --follow, poll for new ones."""
    if not db.exists():
        console.print(f"[red]Trackio db not found: {db}[/red]")
        raise typer.Exit(1)

    con = sqlite3.connect(str(db))
    run_filter = run
    if not run_filter:
        latest = _latest_run(con)
        if not latest:
            console.print("[yellow]No metric rows in db.[/yellow]")
            raise typer.Exit(0)
        run_filter = latest
        console.print(f"[dim]No --run given; defaulting to latest: {latest}[/dim]")

    console.print(_render(_fetch(con, run_filter, last), run_filter))

    if not follow:
        return

    cur = con.execute(
        "SELECT MAX(id) FROM metrics WHERE run_name LIKE ?", (f"%{run_filter}%",)
    )
    seen_max_id = int(cur.fetchone()[0] or 0)

    try:
        while True:
            time.sleep(2)
            new_rows = list(
                con.execute(
                    "SELECT id, step, metrics FROM metrics "
                    "WHERE id > ? AND run_name LIKE ? ORDER BY id ASC",
                    (seen_max_id, f"%{run_filter}%"),
                )
            )
            if not new_rows:
                continue
            new = [(int(s), _decode_metrics(m)) for _, s, m in new_rows]
            console.print(_render(new, run_filter))
            seen_max_id = max(int(rid) for rid, _, _ in new_rows)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    typer.run(tail)
