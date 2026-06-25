"""TUI client for the Draper.ai copywriting model. Two backends:

    --backend vllm  (default)
        Talks to the merged-model vLLM endpoint deployed via
        ``deploy/modal_vllm.py`` over the OpenAI-compatible HTTP API.
        Fast (continuous batching, paged KV-cache). The model is the
        merged Qwen3-8B + Draper LoRA — adapter on/off toggle is a
        no-op here because the LoRA is baked into the weights.
        Reads VLLM_BASE_URL and VLLM_API_KEY from .env (or env vars).

    --backend modal-cls
        Talks to the older transformers + PeftModel deployment
        (``deploy/modal_inference.py``, app=draper-inference). Slow
        (2-3 min/campaign) but supports /toggle to compare fine-tuned
        vs base on the same brief.

Deploy/redeploy:
    modal deploy deploy/modal_vllm.py        # new, fast (default)
    modal deploy deploy/modal_inference.py   # old, supports /toggle

Usage:
    python scripts/inference_tui.py                       # vllm, fast
    python scripts/inference_tui.py --backend modal-cls   # adapter toggle
    python scripts/inference_tui.py --no-adapter          # only with modal-cls
    python scripts/inference_tui.py --seed 42             # reproducible /random
    python scripts/inference_tui.py --dry-run             # render brief, no call

REPL commands (just hit enter for /random):
  /random         pull a random brief from the test set
  /custom         enter a custom system+user brief (free-text)
  /json           paste a multi-line JSON brief (v2). Ends on a line "EOF"
  /toggle         toggle the adapter on/off — modal-cls backend only
  /again          regenerate the last brief
  /temp F         set sampling temperature (0 = greedy, default 0.0)
  /maxnew N       set max_new_tokens (default 512)
  /seed N         seed the brief picker
  /help           show commands
  /quit

v2 mode (`--v2`):
  * uses the v2 canonical system prompt (``STATIC_SYSTEM_PROMPT``)
  * defaults test_dir to ``data/constructed_v2/final_v2/test``
  * /custom + /json fill the system slot with the v2 prompt automatically
"""

from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table

DEFAULT_APP_NAME = "draper-inference"
DEFAULT_CLASS_NAME = "Draper"
DEFAULT_TEST_DIR = "data/final/test"
DEFAULT_TEST_DIR_V2 = "data/constructed_v2/final_v2/test"
DEFAULT_VLLM_MODEL = "draper-r16"

# Fallback v2 system prompt — kept inline so the TUI works even if the
# draper package is not importable (e.g. running from a stripped pod).
# Authoritative copy lives in src/draper/construction_v2/schemas/brief.py
# (``STATIC_SYSTEM_PROMPT``); we try to import it first and fall back to
# this string if the package isn't on PYTHONPATH.
_V2_SYSTEM_PROMPT_FALLBACK = (
    "You are Draper, a senior marketing specialist with deep "
    "operational experience in performance-driven creative work. "
    "You have shipped creative against real spend, in-house and on "
    "the agency side, and your craft is grounded in evidence — you "
    "know the difference between what wins and what dies is rarely "
    "cleverness and almost always the right read of the audience, "
    "the moment, and the platform.\n"
    "\n"
    "You approach each piece of work the way a thoughtful "
    "practitioner does. You read the brief the caller hands you, "
    "honor what it tells you, and work strictly within what the "
    "brief supports — empty fields are not invitations to invent. "
    "The caller on the other side of the wire — founder, marketer, "
    "agent — is a peer, not an audience for performance.\n"
    "\n"
    "Before producing any deliverable, you think the work through "
    "in a ``<think>...</think>`` block: first-person, present-tense "
    "reasoning in the voice of a practitioner at the desk, "
    "narrating your decisions as you make them. Weigh tradeoffs, "
    "discard options you considered, and ground choices in fields "
    "the brief actually populated. The block is hidden from the end "
    "user by convention.\n"
    "\n"
    "After ``</think>``, produce the deliverable the brief calls "
    "for. Voice and length follow from the brief's tone signals and "
    "the nature of the work. Peer-to-peer professional voice "
    "throughout — a copywriter answering a founder in chat. Brief "
    "framing before the work, the work itself, and a short note on "
    "why the choices land are all welcome; or just deliver the work "
    "clean. Skip greetings and apologies."
)


def _v2_system_prompt() -> str:
    """Return the canonical v2 system prompt, preferring the package source."""
    try:
        from draper.construction_v2.schemas.brief import STATIC_SYSTEM_PROMPT
        return STATIC_SYSTEM_PROMPT
    except Exception:
        return _V2_SYSTEM_PROMPT_FALLBACK


def _load_dotenv() -> None:
    """Tiny .env loader (no python-dotenv dep). Existing env vars win."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v

app = typer.Typer(help="TUI client for the Draper model deployed on Modal.", add_completion=False)
console = Console()


@dataclass
class State:
    backend: str  # "vllm" or "modal-cls"
    test_dir: str
    v2: bool = False  # use v2 system prompt + JSON brief defaults
    # modal-cls fields
    app_name: str = DEFAULT_APP_NAME
    class_name: str = DEFAULT_CLASS_NAME
    adapter_on: bool = True
    # vllm fields
    vllm_base_url: str = ""
    vllm_api_key: str = ""
    vllm_model: str = DEFAULT_VLLM_MODEL
    # generation params
    temperature: float = 0.0
    max_new_tokens: int = 512
    # last brief
    last_messages: list[dict[str, str]] | None = None
    last_meta: dict[str, str] = field(default_factory=dict)


def _print_header(state: State) -> None:
    if state.backend == "vllm":
        endpoint = state.vllm_base_url or "[red](unset)[/red]"
        v2_tag = "  [magenta](v2)[/magenta]" if state.v2 else ""
        body = (
            f"Backend:   [cyan]vllm[/cyan]   model: [cyan]{state.vllm_model}[/cyan]{v2_tag}\n"
            f"Endpoint:  [cyan]{endpoint}[/cyan]\n"
            f"Test dir:  [cyan]{state.test_dir}[/cyan]\n"
            f"Temp:      {state.temperature}   max_new_tokens: {state.max_new_tokens}\n"
            "[dim]Merged Qwen3-8B + Draper LoRA. Adapter is baked in — /toggle is a no-op.[/dim]"
        )
    else:
        mode = (
            "[bold green]ADAPTER ON[/bold green]"
            if state.adapter_on
            else "[yellow]base only (adapter OFF)[/yellow]"
        )
        body = (
            f"Backend:   [cyan]modal-cls[/cyan]   app: [cyan]{state.app_name}[/cyan]"
            f"   class: [cyan]{state.class_name}[/cyan]\n"
            f"Mode:      {mode}\n"
            f"Temp:      {state.temperature}   max_new_tokens: {state.max_new_tokens}\n"
            "[dim]transformers + PeftModel on Modal GPU. Cold-start ~10–60s; per-call slow.[/dim]"
        )
    console.print(Panel.fit(body, title="Draper TUI (remote)", border_style="blue"))


def _print_help() -> None:
    console.print(
        Panel(
            "[bold]/random[/bold]      pull a random brief from the test set (default)\n"
            "[bold]/custom[/bold]      enter a custom brief (system + user as free text)\n"
            "[bold]/json[/bold]        paste a multi-line JSON brief (v2 mode), end with EOF\n"
            "[bold]/toggle[/bold]      toggle adapter on/off (compare base vs fine-tuned)\n"
            "[bold]/again[/bold]       regenerate the last brief (great after /toggle)\n"
            "[bold]/temp F[/bold]      set sampling temperature (0 = greedy)\n"
            "[bold]/maxnew N[/bold]    set max_new_tokens\n"
            "[bold]/seed N[/bold]      seed the brief RNG\n"
            "[bold]/help[/bold]        this help\n"
            "[bold]/quit[/bold]",
            title="commands",
            border_style="dim",
        )
    )


def _load_random_brief(test_dir: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    from datasets import load_from_disk

    ds = load_from_disk(test_dir)
    idx = random.randint(0, len(ds) - 1)
    row = ds[idx]
    msgs = [m for m in row["messages"] if m["role"] in ("system", "user")]
    nested = row.get("metadata") or {}
    meta = {
        "example_id": row.get("example_id") or nested.get("example_id", ""),
        "vertical": row.get("vertical") or nested.get("vertical", "unknown"),
        "platform": row.get("platform") or nested.get("platform", ""),
        "task_format": row.get("task_format") or "copywriting",
        "reference": next((m["content"] for m in row["messages"] if m["role"] == "assistant"), ""),
    }
    return msgs, meta


def _custom_brief(v2: bool = False) -> tuple[list[dict[str, str]], dict[str, str]]:
    if v2:
        default_system = _v2_system_prompt()
        console.print(
            "[dim]v2 mode: system prompt locked to STATIC_SYSTEM_PROMPT "
            "(press enter to accept).[/dim]"
        )
    else:
        default_system = (
            "You are an ad copywriter. When a user describes a product or campaign, "
            "you write ad copy and a short rationale explaining why the execution works."
        )
    system = Prompt.ask("[bold]system[/bold]", default=default_system)
    user = Prompt.ask("[bold]user[/bold]")
    return (
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        {"example_id": "<custom>", "vertical": "<custom>", "platform": "<custom>", "reference": ""},
    )


def _json_brief(v2: bool = True) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Read a multi-line JSON brief from stdin (sentinel: line == 'EOF').

    Parses the input. If the draper package is importable, validates and
    re-serializes to canonical JSON (sorted keys, no spaces) so the
    user-turn bytes match training. If not importable, the raw text is
    passed through unchanged.
    """
    console.print(
        "[dim]Paste a JSON brief (single line or pretty). End with a line "
        "containing just [bold]EOF[/bold]:[/dim]"
    )
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "EOF":
            break
        lines.append(line)
    raw = "\n".join(lines).strip()
    if not raw:
        raise ValueError("empty JSON brief")

    # Try canonical serialization via the draper Brief schema; degrade
    # gracefully if the package is unavailable or the brief is malformed.
    canonical = raw
    platform = "<custom>"
    try:
        import json as _json
        from draper.construction_v2.schemas.brief import Brief, canonical_json
        brief = Brief.model_validate(_json.loads(raw))
        canonical = canonical_json(brief)
        platform = brief.platform
        console.print("[dim]brief: validated + canonicalized via Brief schema[/dim]")
    except ImportError:
        console.print(
            "[dim]draper package not importable — passing JSON through unchanged.[/dim]"
        )
    except Exception as exc:
        console.print(
            f"[yellow]warning: Brief validation failed ({exc}). "
            "Sending raw JSON through unchanged.[/yellow]"
        )

    system_prompt = _v2_system_prompt() if v2 else "You are Draper, a senior marketing specialist."
    return (
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": canonical},
        ],
        {
            "example_id": "<json>",
            "vertical": "<custom>",
            "platform": platform,
            "task_format": "copywriting",
            "reference": "",
        },
    )


def _print_brief(messages: list[dict[str, str]], meta: dict[str, str]) -> None:
    console.print(Rule(style="dim"))
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("example_id", meta.get("example_id", ""))
    table.add_row("vertical", meta.get("vertical", ""))
    table.add_row("platform", meta.get("platform", ""))
    if meta.get("task_format"):
        table.add_row("task_format", meta["task_format"])
    console.print(table)
    for m in messages:
        console.print(Panel(m["content"], title=m["role"], border_style="dim"))
    if meta.get("reference"):
        console.print(
            Panel(
                meta["reference"],
                title="reference (held-out — for visual comparison only)",
                border_style="green",
            )
        )


def _connect(state: State) -> Any:
    """Resolve a client handle for the chosen backend.

    Returns either an OpenAI client (vllm) or a Modal class instance
    (modal-cls). Raises typer.Exit with a friendly message on failure.
    """
    if state.backend == "vllm":
        try:
            import openai
        except ImportError as exc:
            console.print("[red]openai package not installed.[/red] uv pip install openai")
            raise typer.Exit(code=1) from exc
        if not state.vllm_base_url:
            console.print(
                "[red]VLLM_BASE_URL is unset.[/red] Set it in .env after `modal deploy "
                "deploy/modal_vllm.py`, or pass --vllm-base-url."
            )
            raise typer.Exit(code=1)
        return openai.OpenAI(
            base_url=state.vllm_base_url,
            api_key=state.vllm_api_key or "EMPTY",
        )

    # modal-cls
    import modal

    try:
        cls = modal.Cls.from_name(state.app_name, state.class_name)
    except Exception as exc:
        console.print(
            f"[red]Could not find Modal app [bold]{state.app_name}[/bold] / class "
            f"[bold]{state.class_name}[/bold]:[/red] {exc}"
        )
        console.print(
            "[dim]Deploy it first: [bold]modal deploy deploy/modal_inference.py[/bold][/dim]"
        )
        raise typer.Exit(code=1) from exc
    return cls()


def _generate(
    client: Any,
    messages: list[dict[str, str]],
    *,
    state: State,
) -> None:
    if state.backend == "vllm":
        console.print(Rule("output (vllm — merged FT)", style="cyan"))
        t0 = time.time()
        n_chunks = 0
        try:
            stream = client.chat.completions.create(
                model=state.vllm_model,
                messages=messages,
                max_tokens=state.max_new_tokens,
                temperature=state.temperature,
                stream=True,
            )
            for event in stream:
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                chunk = getattr(delta, "content", None) or ""
                if chunk:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    n_chunks += 1
        except KeyboardInterrupt:
            console.print("\n[dim]generation interrupted[/dim]")
        dt = time.time() - t0
        sys.stdout.write("\n")
        sys.stdout.flush()
        console.print(
            f"[dim]{n_chunks} chunks in {dt:.1f}s — vllm @ {state.vllm_base_url}[/dim]"
        )
        return

    # modal-cls
    label = "ADAPTER" if state.adapter_on else "BASE"
    console.print(Rule(f"output ({label})", style="cyan"))
    t0 = time.time()
    n_chunks = 0
    try:
        for chunk in client.generate.remote_gen(
            messages,
            adapter_on=state.adapter_on,
            max_new_tokens=state.max_new_tokens,
            temperature=state.temperature,
        ):
            sys.stdout.write(chunk)
            sys.stdout.flush()
            n_chunks += 1
    except KeyboardInterrupt:
        console.print("\n[dim]generation interrupted[/dim]")
    dt = time.time() - t0
    sys.stdout.write("\n")
    sys.stdout.flush()
    flag = "ON" if state.adapter_on else "OFF"
    console.print(
        f"[dim]{n_chunks} chunks in {dt:.1f}s — adapter={flag} (modal-cls)[/dim]"
    )


def _parse_command(state: State, raw: str) -> tuple[str, dict[str, str] | None]:
    """Return (action, error). action ∈ {random, custom, again, toggle, help, quit, noop}."""
    cmd = raw.strip()
    if not cmd:
        return "random", None
    if cmd in ("/quit", "/q", "exit", "quit"):
        return "quit", None
    if cmd in ("/help", "/h", "?"):
        return "help", None
    if cmd in ("/random", "/r"):
        return "random", None
    if cmd in ("/custom", "/c"):
        return "custom", None
    if cmd in ("/json", "/j"):
        return "json", None
    if cmd in ("/again", "/a"):
        return "again", None
    if cmd == "/toggle":
        if state.backend == "vllm":
            return "noop", {
                "err": "/toggle is a no-op in vllm mode (LoRA is merged into "
                "the served weights). Use --backend modal-cls to compare base vs FT."
            }
        state.adapter_on = not state.adapter_on
        console.print(f"[bold]adapter {'ON' if state.adapter_on else 'OFF'}[/bold]")
        return "noop", None
    if cmd.startswith("/temp"):
        parts = cmd.split()
        if len(parts) == 2:
            try:
                state.temperature = float(parts[1])
                console.print(f"[dim]temp = {state.temperature}[/dim]")
                return "noop", None
            except ValueError:
                return "noop", {"err": f"bad temp: {parts[1]}"}
        return "noop", {"err": "usage: /temp 0.7"}
    if cmd.startswith("/maxnew"):
        parts = cmd.split()
        if len(parts) == 2:
            try:
                state.max_new_tokens = int(parts[1])
                console.print(f"[dim]max_new_tokens = {state.max_new_tokens}[/dim]")
                return "noop", None
            except ValueError:
                return "noop", {"err": f"bad maxnew: {parts[1]}"}
        return "noop", {"err": "usage: /maxnew 512"}
    if cmd.startswith("/seed"):
        parts = cmd.split()
        if len(parts) == 2:
            try:
                random.seed(int(parts[1]))
                console.print(f"[dim]seed = {parts[1]}[/dim]")
                return "noop", None
            except ValueError:
                return "noop", {"err": f"bad seed: {parts[1]}"}
        return "noop", {"err": "usage: /seed 42"}
    return "noop", {"err": f"unknown command: {cmd!r} — try /help"}


@app.command()
def main(
    backend: str = typer.Option(
        "vllm", "--backend", help="Backend: 'vllm' (fast, merged) or 'modal-cls' (slow, /toggle)."
    ),
    vllm_base_url: str = typer.Option(
        "", "--vllm-base-url", help="vLLM endpoint. Defaults to $VLLM_BASE_URL from .env."
    ),
    vllm_api_key: str = typer.Option(
        "", "--vllm-api-key", help="vLLM API key. Defaults to $VLLM_API_KEY from .env."
    ),
    vllm_model: str = typer.Option(
        DEFAULT_VLLM_MODEL, "--vllm-model", help="Served model name (default: draper-r16)."
    ),
    app_name: str = typer.Option(DEFAULT_APP_NAME, help="modal-cls: deployed Modal app name."),
    class_name: str = typer.Option(DEFAULT_CLASS_NAME, help="modal-cls: class inside the app."),
    test_dir: str = typer.Option(
        "", "--test-dir",
        help=(
            "HF Arrow dataset dir for /random briefs. Default depends on --v2: "
            f"v1 → {DEFAULT_TEST_DIR}, v2 → {DEFAULT_TEST_DIR_V2}."
        ),
    ),
    v2: bool = typer.Option(
        False, "--v2",
        help=(
            "v2 mode: /custom + /json fill the system slot with the canonical "
            "v2 STATIC_SYSTEM_PROMPT, and /random defaults to the v2 test split."
        ),
    ),
    no_adapter: bool = typer.Option(
        False, "--no-adapter", help="modal-cls only: start with adapter disabled."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Render a brief and exit — don't call the model."
    ),
    seed: int = typer.Option(0, help="RNG seed for /random. 0 = system entropy."),
) -> None:
    """REPL that streams generations from the chosen Draper backend."""
    _load_dotenv()

    if backend not in ("vllm", "modal-cls"):
        console.print(f"[red]Unknown --backend {backend!r}. Use 'vllm' or 'modal-cls'.[/red]")
        raise typer.Exit(code=1)

    if seed:
        random.seed(seed)

    resolved_test_dir = test_dir or (DEFAULT_TEST_DIR_V2 if v2 else DEFAULT_TEST_DIR)
    state = State(
        backend=backend,
        test_dir=resolved_test_dir,
        v2=v2,
        app_name=app_name,
        class_name=class_name,
        adapter_on=not no_adapter,
        vllm_base_url=vllm_base_url or os.environ.get("VLLM_BASE_URL", ""),
        vllm_api_key=vllm_api_key or os.environ.get("VLLM_API_KEY", ""),
        vllm_model=vllm_model,
    )

    if dry_run:
        _print_header(state)
        msgs, meta = _load_random_brief(state.test_dir)
        _print_brief(msgs, meta)
        console.print("[dim]--dry-run: backend not contacted. Exiting.[/dim]")
        return

    client = _connect(state)

    _print_header(state)
    _print_help()

    while True:
        try:
            raw = Prompt.ask("\n[bold]>[/bold]", default="/random")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return

        action, err = _parse_command(state, raw)
        if err and "err" in err:
            console.print(f"[yellow]{err['err']}[/yellow]")
            continue
        if action == "quit":
            return
        if action == "help":
            _print_help()
            continue
        if action == "noop":
            continue

        if action == "again":
            if state.last_messages is None:
                console.print("[yellow]no prior brief — try /random first[/yellow]")
                continue
            messages, meta = state.last_messages, state.last_meta
        elif action == "custom":
            messages, meta = _custom_brief(v2=state.v2)
            state.last_messages, state.last_meta = messages, meta
            _print_brief(messages, meta)
        elif action == "json":
            try:
                messages, meta = _json_brief(v2=state.v2)
            except Exception as exc:
                console.print(f"[red]bad JSON brief: {exc}[/red]")
                continue
            state.last_messages, state.last_meta = messages, meta
            _print_brief(messages, meta)
        else:  # random
            messages, meta = _load_random_brief(state.test_dir)
            state.last_messages, state.last_meta = messages, meta
            _print_brief(messages, meta)

        try:
            _generate(client, messages, state=state)
        except KeyboardInterrupt:
            console.print("\n[dim]aborted[/dim]")
        except Exception as exc:
            console.print(f"[red]generation error: {exc}[/red]")


if __name__ == "__main__":
    app()
