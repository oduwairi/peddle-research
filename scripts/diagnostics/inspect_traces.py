#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "psycopg[binary]>=3.2",
#   "textual>=0.85",
#   "rich>=13.7",
# ]
# ///
"""
Step-by-step trace inspector for the copywriting agent pipeline.

Loads agent_trace rows from the frontend's Postgres DB and renders them as a
navigable, fully-expandable tree:

    conversation
    └── message <responseMessageId>
        ├── 1. research_writer  attempt=0 step=0
        │     ├── prompts.system     (full system prompt)
        │     ├── prompts.user       (the modelMessages or rendered user text)
        │     ├── tool_calls         (model's tool invocations / input artifact)
        │     ├── tool_results       (tool outputs / produced artifact)
        │     ├── text               (free-text / reasoning)
        │     └── meta               (model, tokens, latency, finish_reason)
        ├── 2. research_critic ...
        ├── 3. angles_writer  ...
        └── ...

Run:
    uv run scripts/diagnostics/inspect_traces.py                  # picker
    uv run scripts/diagnostics/inspect_traces.py <conversationId> # jump straight in
    uv run scripts/diagnostics/inspect_traces.py --list           # print conversations and exit
    uv run scripts/diagnostics/inspect_traces.py <id> --dump      # plain stdout (no TUI)

DATABASE_URL is read from frontend/.env.local (or the env var if already set).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from rich.console import Console
from rich.json import JSON as RichJSON
from rich.measure import Measurement
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer
from textual.widgets import Footer, Header, Static, Tree
from textual.widgets.tree import TreeNode

# Stage ordering mirrors frontend/components/chat/pipeline-indicator.tsx.
# Anything not in this list sorts to the bottom by raw stage name.
STAGE_ORDER = [
    "research_writer",
    "research_critic",
    "research_fallback",
    "angles_writer",
    "angles_critic",
    "draft",
    "draft_critique",
    "draft_critique:revise",
    "visual_writer",
    "visual_critic",
    "visual_image",
    "emit",
]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = REPO_ROOT / "frontend" / ".env.local"

# Width we pre-render every right-pane panel at. Larger than any reasonable
# terminal pane, so the Static widget ends up wider than its container and the
# horizontal scrollbar actually has something to scroll. Tweak if you run a
# 4-monitor wraparound and somehow exceed this.
PRERENDER_WIDTH = 220


# ───────────────────────────────────────── env / db ─────────────────────────


def load_env_local(path: Path) -> dict[str, str]:
    """
    Parse frontend/.env.local. Bash-style key=value, ignoring comments and
    blank lines. Strips matching surrounding quotes. Not a full shell parser
    — good enough for the values we need.
    """
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in {'"', "'"}:
            val = val[1:-1]
        out[key] = val
    return out


def get_database_url() -> str:
    if "DATABASE_URL" in os.environ:
        return os.environ["DATABASE_URL"]
    parsed = load_env_local(ENV_FILE)
    url = parsed.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            f"DATABASE_URL not set and not found in {ENV_FILE}. "
            "Export DATABASE_URL or fill it in frontend/.env.local."
        )
    return url


# ───────────────────────────────────────── data ─────────────────────────────


@dataclass
class TraceRow:
    id: str
    conversation_id: str
    message_id: str | None
    stage: str | None
    step_index: int
    finish_reason: str | None
    text: str | None
    tool_calls: Any
    tool_results: Any
    input_tokens: int
    output_tokens: int
    model_id: str | None
    latency_ms: int | None
    created_at: datetime

    @property
    def stage_label(self) -> str:
        return self.stage or "(legacy)"

    def stage_sort_key(self) -> tuple[int, str, int]:
        try:
            base = self.stage or ""
            # Trim "error:" prefix for ordering so an error row sorts next
            # to its happy-path sibling.
            if base.startswith("error:"):
                base = base[len("error:") :]
            idx = STAGE_ORDER.index(base)
        except ValueError:
            idx = len(STAGE_ORDER)
        return (idx, self.stage or "", self.step_index)

    def prompts(self) -> tuple[str | None, Any | None]:
        """
        Pull system + user prompts out of toolCalls._prompts (set by
        persistStageTrace / persistStep). Returns (system_text, user_payload).
        """
        tc = self.tool_calls
        if isinstance(tc, dict) and "_prompts" in tc:
            p = tc.get("_prompts") or {}
            return (p.get("system"), p.get("user"))
        return (None, None)

    def input_artifact(self) -> Any:
        """toolCalls without the _prompts wrapper (the actual input artifact)."""
        tc = self.tool_calls
        if isinstance(tc, dict):
            cleaned = {k: v for k, v in tc.items() if k != "_prompts"}
            # If we wrapped an array under _toolCalls (step rows), unwrap.
            if "_toolCalls" in cleaned and len(cleaned) == 1:
                return cleaned["_toolCalls"]
            return cleaned
        return tc


def fetch_conversations(conn: psycopg.Connection) -> list[dict[str, Any]]:
    """List conversations that have at least one trace, most recent first."""
    with conn.cursor(row_factory=dict_row) as cur:
        # Drizzle uses camelCase column names — Postgres preserves the case
        # because they were quoted in the DDL. Match that quoting here.
        cur.execute(
            """
            select c.id,
                   c.title,
                   c."userId"           as user_id,
                   c."updatedAt"        as updated_at,
                   max(t."createdAt")   as last_trace_at,
                   count(t.id)          as trace_count
            from agent_trace t
            join conversation c on c.id = t."conversationId"
            group by c.id, c.title, c."userId", c."updatedAt"
            order by max(t."createdAt") desc
            limit 100
            """
        )
        return list(cur.fetchall())


def fetch_traces(conn: psycopg.Connection, conversation_id: str) -> list[TraceRow]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            select id,
                   "conversationId" as conversation_id,
                   "messageId"      as message_id,
                   stage,
                   "stepIndex"      as step_index,
                   "finishReason"   as finish_reason,
                   text,
                   "toolCalls"      as tool_calls,
                   "toolResults"    as tool_results,
                   "inputTokens"    as input_tokens,
                   "outputTokens"   as output_tokens,
                   "modelId"        as model_id,
                   "latencyMs"      as latency_ms,
                   "createdAt"      as created_at
            from agent_trace
            where "conversationId" = %s
            order by "createdAt" asc, "stepIndex" asc
            """,
            (conversation_id,),
        )
        return [TraceRow(**row) for row in cur.fetchall()]


# ───────────────────────────────────────── rendering ────────────────────────


def _short_str(s: str, n: int = 80) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _prerender(renderable: Any, fallback: int = PRERENDER_WIDTH) -> Text:
    """
    Render a Rich renderable at its NATURAL width (Rich's max measurement,
    floored at `fallback` and capped to keep runaway JSON sane) and return
    a `Text` whose lines reflect that width. Static measures Text by its
    longest line, so the widget overflows narrow panes and the horizontal
    scrollbar has the full content to scroll through — no clipping at the
    far edge.
    """
    measure_console = Console(width=10_000, legacy_windows=False)
    measurement = Measurement.get(
        measure_console, measure_console.options, renderable
    )
    width = max(fallback, measurement.maximum)
    # Hard cap so a pathological 50k-char JSON line doesn't blow up render.
    width = min(width, 4000)
    console = Console(
        width=width,
        record=True,
        force_terminal=True,
        color_system="truecolor",
        legacy_windows=False,
    )
    console.print(renderable, soft_wrap=False, overflow="ignore", crop=False)
    ansi = console.export_text(styles=True, clear=True)
    # Strip per-line trailing whitespace so the Static measures the actual
    # content width, not the console's right-edge padding.
    lines = [ln.rstrip() for ln in ansi.splitlines()]
    return Text.from_ansi("\n".join(lines), no_wrap=True)


def _summarize_artifact(payload: Any) -> str:
    """One-liner preview of a JSON artifact for a tree node label."""
    if payload is None:
        return "—"
    if isinstance(payload, dict):
        keys = list(payload.keys())[:5]
        more = "" if len(payload) <= len(keys) else f" +{len(payload) - len(keys)}"
        return "{" + ", ".join(keys) + more + "}"
    if isinstance(payload, list):
        return f"[{len(payload)} item{'s' if len(payload) != 1 else ''}]"
    if isinstance(payload, str):
        return f'"{_short_str(payload, 60)}"'
    return str(payload)


# ───────────────────────────────────────── TUI ──────────────────────────────


class TraceInspector(App[None]):
    CSS = """
    Screen { layout: horizontal; }
    #left { width: 42%; min-width: 36; border-right: solid $primary 30%;
             overflow-x: auto; overflow-y: auto; }
    #left Tree { padding: 0 1; width: auto; min-width: 100%; }
    #right { width: 58%; padding: 0 1; overflow-x: auto; overflow-y: auto; }
    #detail { width: auto; min-width: 100%; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "reload", "Reload"),
        Binding("e", "expand_all", "Expand all"),
        Binding("c", "collapse_all", "Collapse all"),
        # Use [ and ] for horizontal scroll so we don't shadow the Tree's
        # arrow-key collapse/expand bindings.
        Binding("bracket_left", "scroll_left", "Scroll ←", show=True, key_display="["),
        Binding("bracket_right", "scroll_right", "Scroll →", show=True, key_display="]"),
    ]

    def __init__(
        self,
        conversation_id: str,
        traces: list[TraceRow],
        title_text: str,
    ) -> None:
        super().__init__()
        self.conversation_id = conversation_id
        self.traces = traces
        self.title_text = title_text
        # Map TreeNode.id → renderable for the right pane.
        self._panel_for: dict[int, Any] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal():
            with ScrollableContainer(id="left"):
                yield Tree(self.title_text, id="trace-tree")
            with ScrollableContainer(id="right"):
                yield Static(_prerender(self._welcome_panel()), id="detail", expand=False)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Agent trace inspector"
        self.sub_title = self.conversation_id
        tree = self.query_one("#trace-tree", Tree)
        tree.show_root = True
        tree.root.expand()
        self._populate(tree.root)
        # Default to first leaf so user has something to read on open.
        if tree.root.children:
            tree.select_node(tree.root.children[0])

    def _welcome_panel(self) -> Any:
        return Panel(
            Text.from_markup(
                "[bold]Pick a row on the left[/bold]\n\n"
                "Each pipeline step is one row in [italic]agent_trace[/italic].\n"
                "Drill in to see the exact prompts, inputs, and outputs.\n\n"
                "[dim]e/c — expand/collapse all   r — reload   q — quit[/dim]"
            ),
            title="trace inspector",
            border_style="cyan",
        )

    # ── tree population ────────────────────────────────────────────────

    def _populate(self, root: TreeNode) -> None:
        # Group by message_id (one full pipeline run per assistant message),
        # preserving creation order across messages.
        by_message: dict[str | None, list[TraceRow]] = {}
        order: list[str | None] = []
        for t in self.traces:
            key = t.message_id
            if key not in by_message:
                by_message[key] = []
                order.append(key)
            by_message[key].append(t)

        for msg_idx, mid in enumerate(order, start=1):
            rows = by_message[mid]
            # Within a message, sort by canonical stage order then step.
            rows.sort(key=lambda r: (r.created_at, r.stage_sort_key()))
            label = (
                f"[bold]message {msg_idx}[/bold]  "
                f"[dim]({mid or 'unknown'})[/dim]  "
                f"{len(rows)} row{'s' if len(rows) != 1 else ''}"
            )
            msg_node = root.add(label, expand=True)
            self._panel_for[msg_node.id] = self._message_panel(mid, rows)
            for i, row in enumerate(rows, start=1):
                self._add_row_node(msg_node, i, row)

    def _add_row_node(self, parent: TreeNode, idx: int, row: TraceRow) -> None:
        stage = row.stage_label
        is_error = stage.startswith("error:")
        prefix = "[red]" if is_error else "[cyan]"
        latency = f"{row.latency_ms}ms" if row.latency_ms is not None else "—"
        tokens = f"{row.input_tokens}/{row.output_tokens}"
        verdict_bit = f" · {row.finish_reason}" if row.finish_reason else ""
        label = (
            f"{idx:2}. {prefix}{stage}[/]  "
            f"[dim]step={row.step_index}  {latency}  tok={tokens}{verdict_bit}[/dim]"
        )
        node = parent.add(label, expand=False)
        self._panel_for[node.id] = self._row_overview_panel(row)

        sys_prompt, user_payload = row.prompts()
        # 1) System prompt
        sys_label = (
            "[bold]prompts.system[/bold]"
            if sys_prompt
            else "[dim]prompts.system (none)[/dim]"
        )
        sys_node = node.add_leaf(sys_label)
        self._panel_for[sys_node.id] = (
            Panel(
                Syntax(sys_prompt, "markdown", word_wrap=False, theme="ansi_dark"),
                title="system prompt",
                border_style="green",
            )
            if sys_prompt
            else Panel(
                Text("(no system prompt recorded for this row)", style="dim"),
                title="system prompt",
                border_style="dim",
            )
        )

        # 2) User prompt
        user_label = (
            "[bold]prompts.user[/bold]"
            if user_payload is not None
            else "[dim]prompts.user (none)[/dim]"
        )
        user_node = node.add_leaf(user_label)
        self._panel_for[user_node.id] = self._user_prompt_panel(user_payload)

        # 3) Input artifact (tool_calls minus _prompts)
        input_artifact = row.input_artifact()
        in_label = f"[bold]tool_calls[/bold]  [dim]{_summarize_artifact(input_artifact)}[/dim]"
        in_node = node.add_leaf(in_label)
        self._panel_for[in_node.id] = self._json_panel(
            input_artifact, "tool_calls (input artifact)", "yellow"
        )

        # 4) Output artifact (tool_results)
        out_label = (
            f"[bold]tool_results[/bold]  [dim]{_summarize_artifact(row.tool_results)}[/dim]"
        )
        out_node = node.add_leaf(out_label)
        self._panel_for[out_node.id] = self._json_panel(
            row.tool_results, "tool_results (output)", "magenta"
        )

        # 5) Free-text (reasoning / fix notes / error message)
        text_label = (
            "[bold]text[/bold]"
            if row.text
            else "[dim]text (none)[/dim]"
        )
        text_node = node.add_leaf(text_label)
        self._panel_for[text_node.id] = (
            Panel(
                Text(row.text, no_wrap=True),
                title="text",
                border_style="blue",
            )
            if row.text
            else Panel(Text("(no text recorded)", style="dim"), title="text", border_style="dim")
        )

        # 6) Meta
        meta_node = node.add_leaf("[bold]meta[/bold]")
        self._panel_for[meta_node.id] = self._meta_panel(row)

    # ── right-pane panels ──────────────────────────────────────────────

    def _row_overview_panel(self, row: TraceRow) -> Any:
        t = Table.grid(padding=(0, 1))
        t.add_column(style="dim", justify="right")
        t.add_column()
        t.add_row("stage", row.stage_label)
        t.add_row("step_index", str(row.step_index))
        t.add_row("finish", row.finish_reason or "—")
        t.add_row("model", row.model_id or "—")
        t.add_row(
            "tokens", f"{row.input_tokens} in / {row.output_tokens} out"
        )
        t.add_row("latency", f"{row.latency_ms}ms" if row.latency_ms is not None else "—")
        t.add_row("created", row.created_at.isoformat(timespec="seconds"))
        t.add_row("trace_id", row.id)
        t.add_row("message_id", row.message_id or "—")
        return Panel(t, title=f"row overview · {row.stage_label}", border_style="cyan")

    def _message_panel(self, mid: str | None, rows: list[TraceRow]) -> Any:
        t = Table.grid(padding=(0, 1))
        t.add_column(style="dim", justify="right")
        t.add_column()
        t.add_row("message_id", mid or "—")
        t.add_row("rows", str(len(rows)))
        if rows:
            first = rows[0].created_at
            last = rows[-1].created_at
            t.add_row("started", first.isoformat(timespec="seconds"))
            t.add_row("ended",   last.isoformat(timespec="seconds"))
            total_in = sum(r.input_tokens for r in rows)
            total_out = sum(r.output_tokens for r in rows)
            t.add_row("tokens", f"{total_in} in / {total_out} out")
            stages = ", ".join(sorted({r.stage_label for r in rows}))
            t.add_row("stages", _short_str(stages, 200))
        return Panel(t, title="message run", border_style="cyan")

    def _user_prompt_panel(self, payload: Any) -> Any:
        if payload is None:
            return Panel(
                Text("(no user prompt recorded for this row)", style="dim"),
                title="user prompt",
                border_style="dim",
            )
        if isinstance(payload, str):
            return Panel(
                Syntax(payload, "markdown", word_wrap=False, theme="ansi_dark"),
                title="user prompt",
                border_style="green",
            )
        # modelMessages is a list of {role, content[]} objects — render JSON.
        return self._json_panel(payload, "user prompt (model messages)", "green")

    def _json_panel(self, payload: Any, title: str, border: str) -> Any:
        if payload is None:
            return Panel(Text("null", style="dim"), title=title, border_style="dim")
        try:
            text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = repr(payload)
        if len(text) <= 200_000:
            return Panel(RichJSON(text), title=title, border_style=border)
        # Very large blobs — fall back to plain text to keep the renderer fast.
        return Panel(
            Text(text[:200_000] + "\n...[truncated]", no_wrap=True),
            title=f"{title} (truncated)",
            border_style=border,
        )

    def _meta_panel(self, row: TraceRow) -> Any:
        meta = {
            "id": row.id,
            "conversation_id": row.conversation_id,
            "message_id": row.message_id,
            "stage": row.stage,
            "step_index": row.step_index,
            "finish_reason": row.finish_reason,
            "input_tokens": row.input_tokens,
            "output_tokens": row.output_tokens,
            "model_id": row.model_id,
            "latency_ms": row.latency_ms,
            "created_at": row.created_at.isoformat(),
        }
        return self._json_panel(meta, "meta", "cyan")

    # ── interactions ───────────────────────────────────────────────────

    def _show(self, node_id: int | None, fallback: Any | None = None) -> None:
        panel = self._panel_for.get(node_id) if node_id is not None else None
        if panel is None:
            if fallback is None:
                return
            panel = fallback
        # Pre-render at a fixed wide width so long JSON / prompt lines extend
        # past the pane and the right-side horizontal scrollbar actually has
        # content to scroll. Cheap enough to do per selection.
        self.query_one("#detail", Static).update(_prerender(panel))
        # Reset right-pane scroll so the user sees the start of the new panel.
        self.query_one("#right", ScrollableContainer).scroll_home(animate=False)

    def on_tree_node_selected(self, event: Tree.NodeSelected[Any]) -> None:
        self._show(event.node.id, fallback=self._welcome_panel())

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[Any]) -> None:
        # Treat highlight (arrow keys) the same as click — feels natural.
        self._show(event.node.id)

    def action_expand_all(self) -> None:
        tree = self.query_one("#trace-tree", Tree)
        _walk_tree_expand(tree.root, expand=True)

    def action_collapse_all(self) -> None:
        tree = self.query_one("#trace-tree", Tree)
        # Keep the very root open so the user isn't staring at one collapsed
        # line — collapse only its descendants.
        for child in tree.root.children:
            _walk_tree_expand(child, expand=False)

    def _focused_pane(self) -> ScrollableContainer:
        # Default to the right pane (the long-content one) unless the tree is focused.
        focused = self.focused
        if focused is not None and focused.id == "trace-tree":
            return self.query_one("#left", ScrollableContainer)
        return self.query_one("#right", ScrollableContainer)

    def action_scroll_left(self) -> None:
        self._focused_pane().scroll_relative(x=-8, animate=False)

    def action_scroll_right(self) -> None:
        self._focused_pane().scroll_relative(x=8, animate=False)

    def action_reload(self) -> None:
        with psycopg.connect(get_database_url()) as conn:
            self.traces = fetch_traces(conn, self.conversation_id)
        tree = self.query_one("#trace-tree", Tree)
        tree.clear()
        self._panel_for.clear()
        self._populate(tree.root)
        tree.root.expand()


def _walk_tree_expand(node: TreeNode, expand: bool) -> None:
    if expand:
        node.expand()
    else:
        node.collapse()
    for child in node.children:
        _walk_tree_expand(child, expand=expand)


# ───────────────────────────────────────── CLI helpers ──────────────────────


def cmd_list(conn: psycopg.Connection) -> int:
    rows = fetch_conversations(conn)
    if not rows:
        print("(no conversations with traces)")
        return 0
    table = Table(title="Recent conversations with traces", box=None)
    table.add_column("#", justify="right", style="dim")
    table.add_column("conversation_id")
    table.add_column("title", max_width=50)
    table.add_column("rows", justify="right")
    table.add_column("last trace at")
    for i, r in enumerate(rows, start=1):
        table.add_row(
            str(i),
            r["id"],
            r["title"] or "—",
            str(r["trace_count"]),
            r["last_trace_at"].isoformat(timespec="seconds"),
        )
    Console().print(table)
    return 0


def cmd_dump(conn: psycopg.Connection, conversation_id: str) -> int:
    """Plain stdout dump — handy for piping into less / grep / a file."""
    traces = fetch_traces(conn, conversation_id)
    if not traces:
        print(f"(no traces for {conversation_id})")
        return 1
    console = Console()
    for i, row in enumerate(traces, start=1):
        sys_prompt, user_payload = row.prompts()
        console.rule(
            f"{i:2}. {row.stage_label}  step={row.step_index}  "
            f"{row.finish_reason or ''}".strip()
        )
        meta = (
            f"model={row.model_id}  tokens={row.input_tokens}/{row.output_tokens}  "
            f"latency={row.latency_ms}ms  at={row.created_at.isoformat(timespec='seconds')}"
        )
        console.print(meta, style="dim")
        if sys_prompt:
            console.print(Panel(sys_prompt, title="system prompt", border_style="green"))
        if user_payload is not None:
            if isinstance(user_payload, str):
                console.print(Panel(user_payload, title="user prompt", border_style="green"))
            else:
                console.print(
                    Panel(
                        RichJSON(json.dumps(user_payload, default=str)),
                        title="user prompt",
                        border_style="green",
                    )
                )
        if row.input_artifact():
            console.print(
                Panel(
                    RichJSON(json.dumps(row.input_artifact(), default=str)),
                    title="tool_calls",
                    border_style="yellow",
                )
            )
        if row.tool_results not in (None, [], {}):
            console.print(
                Panel(
                    RichJSON(json.dumps(row.tool_results, default=str)),
                    title="tool_results",
                    border_style="magenta",
                )
            )
        if row.text:
            console.print(Panel(row.text, title="text", border_style="blue"))
    return 0


def picker(conn: psycopg.Connection) -> str | None:
    rows = fetch_conversations(conn)
    if not rows:
        print("(no conversations with traces)")
        return None
    cmd_list(conn)
    try:
        choice = input("\nPick a conversation # (or paste an id, blank to quit): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not choice:
        return None
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(rows):
            return str(rows[idx]["id"])
        print(f"out of range: {choice}")
        return None
    return choice


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "conversation_id",
        nargs="?",
        help="conversation id; omit to open the picker",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list conversations with traces and exit",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="print all traces to stdout instead of launching the TUI",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="resolve conversation_id to the most recent run with traces",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        url = get_database_url()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    with psycopg.connect(url) as conn:
        if args.list:
            return cmd_list(conn)

        if args.latest and not args.conversation_id:
            rows = fetch_conversations(conn)
            if not rows:
                print("(no conversations with traces)", file=sys.stderr)
                return 1
            cid = rows[0]["id"]
        else:
            cid = args.conversation_id or picker(conn)
        if not cid:
            return 0

        traces = fetch_traces(conn, cid)
        if not traces:
            print(f"(no traces for conversation {cid})")
            return 1

        if args.dump:
            return cmd_dump(conn, cid)

        title = f"conversation {cid} · {len(traces)} rows"
        TraceInspector(cid, traces, title).run()
        return 0


if __name__ == "__main__":
    sys.exit(main())
