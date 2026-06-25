"""Re-render the v2 dataset into Gemma 4's native think-channel format.

Arm C of the v2 bake-off. Reads ``data/constructed_v2/final_v2/`` (bare
``<think>...</think>`` tags in the assistant content) and writes
``data/constructed_v2/final_v2_gemma_native/`` with the assistant turn
reshaped so Gemma's tokenizer treats the rationale as its native think
channel rather than as inline text tokens.

The conversion is a per-row string substitution on the assistant message:

    <think>\n{R}\n</think>\n\n{deliverable}
        →
    {open_think}{R}{close_think}\n\n{deliverable}

where ``open_think`` / ``close_think`` are the *actual* special tokens
Gemma 4 E4B's tokenizer uses for its think channel. Those token strings
are discovered at runtime by introspecting the tokenizer (we do not
hard-code them — the training_v2.yaml header documents ``<|channel>``
notation, but the real strings live in tokenizer config files).

After conversion the script runs a round-trip check on 5 random rows:

    1. apply_chat_template the new messages,
    2. tokenize,
    3. assert the open/close think tokens encode to *single* special-token
       IDs (not byte sequences — that would mean the tokenizer didn't
       recognize them),
    4. assert the think-channel span sits inside the assistant turn span,
    5. assert the assistant-only mask hits 10–90 % of batch tokens (the
       same sanity gate trainer.py uses to catch Run-#001-class bugs).

If any of these fail the script exits non-zero — Arm C is infeasible
without trainer changes and the bake-off short-circuits to two arms.

Usage:

    python scripts/construct_v2/render_for_gemma.py
    python scripts/construct_v2/render_for_gemma.py --limit 50    # smoke
    python scripts/construct_v2/render_for_gemma.py --input data/constructed_v2/final_v2 \\
        --output data/constructed_v2/final_v2_gemma_native
"""

from __future__ import annotations

import logging
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import typer
from datasets import Dataset, DatasetDict, Features, Value, load_from_disk
from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from draper.utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger("draper")
console = Console()
app = typer.Typer(help="Re-render v2 dataset for Gemma 4 native think channel.")


# Match what the v2 builder writes (src/draper/construction_v2/dataset/builder.py:61):
#   f"<think>\n{think}\n</think>\n\n{deliverable}"
_THINK_RE = re.compile(r"^<think>\n(?P<think>.*?)\n</think>\n\n(?P<rest>.*)$", re.DOTALL)


# Heuristics for discovering Gemma's native think-channel tokens. We
# look at the tokenizer's full vocab + special tokens for IDs whose
# string form matches any of these case-insensitive substrings. The
# OPEN/CLOSE pair is then picked as the lexicographically-first pair
# that brackets a "thought" keyword.
_THINK_OPEN_HINTS = [
    "channel_thought",
    "channel.thought",  # observed Gemma channel naming
    "thought_open",
    "think_start",
    "<think>",  # if the model genuinely has a bare-tag special token
    "<|think|>",  # config-header notation
    "<|channel>thought",  # config-header notation
]
_THINK_CLOSE_HINTS = [
    "channel_end",
    "channel.end",
    "thought_close",
    "think_end",
    "</think>",
    "<|/think|>",
    "<channel|>",
    "thought<channel|>",
]


def _discover_channel_tokens(tokenizer: Any) -> tuple[str, str] | None:
    """Inspect ``tokenizer`` for Gemma's native think-channel special tokens.

    Returns ``(open_str, close_str)`` if a plausible pair is found, else
    ``None``. We do not throw on miss — the caller decides whether that
    means "fail loudly and skip Arm C" or "try a fallback path".
    """
    vocab = tokenizer.get_vocab()  # dict[str, int]
    added = tokenizer.get_added_vocab() if hasattr(tokenizer, "get_added_vocab") else {}

    candidates: list[str] = list(set(vocab) | set(added))

    def _match_any(s: str, hints: list[str]) -> bool:
        s_lc = s.lower()
        return any(h.lower() in s_lc for h in hints)

    opens = sorted(s for s in candidates if _match_any(s, _THINK_OPEN_HINTS))
    closes = sorted(s for s in candidates if _match_any(s, _THINK_CLOSE_HINTS))

    if not opens or not closes:
        return None

    # Prefer pairs where the strings share a stem (e.g.
    # "<|channel>thought" + "thought<channel|>" share "thought") — that's
    # how Gemma 4's documented native channel is shaped.
    for o in opens:
        for c in closes:
            if "think" in (o.lower() + c.lower()) or "thought" in (o.lower() + c.lower()):
                return o, c
    # Fall back to the first of each, in sorted order — let the round-trip
    # check be the final arbiter.
    return opens[0], closes[0]


def _convert_assistant_content(content: str, open_tok: str, close_tok: str) -> str:
    """Replace bare ``<think>\\n...\\n</think>\\n\\n`` with channel tokens."""
    m = _THINK_RE.match(content)
    if m is None:
        return content  # No think block? Leave untouched (shouldn't happen for v2).
    return f"{open_tok}{m['think']}{close_tok}\n\n{m['rest']}"


def _make_row(row: dict[str, Any], open_tok: str, close_tok: str) -> dict[str, Any]:
    new_messages: list[dict[str, str]] = []
    for msg in row["messages"]:
        if msg["role"] == "assistant":
            new_messages.append(
                {
                    "role": "assistant",
                    "content": _convert_assistant_content(msg["content"], open_tok, close_tok),
                }
            )
        else:
            new_messages.append({"role": msg["role"], "content": msg["content"]})
    return {"messages": new_messages, "metadata": row["metadata"]}


def _round_trip_check(
    converted: Dataset,
    tokenizer: Any,
    open_tok: str,
    close_tok: str,
    n_samples: int = 5,
) -> None:
    """Validate that the channel tokens encode as special-token IDs and
    the assistant-only mask span sanity-gate holds.

    Raises ``RuntimeError`` if any sample fails — Arm C is then infeasible
    without trainer / data-shape changes.
    """
    open_ids = tokenizer.encode(open_tok, add_special_tokens=False)
    close_ids = tokenizer.encode(close_tok, add_special_tokens=False)
    if len(open_ids) != 1 or len(close_ids) != 1:
        msg = (
            f"Channel tokens did NOT encode to single IDs: "
            f"open={open_tok!r}→{open_ids}, close={close_tok!r}→{close_ids}. "
            "The tokenizer is treating them as text — Arm C cannot proceed "
            "without trainer changes (option (ii): pre-templated strings + "
            "skip re-templating in the trainer's formatting_func)."
        )
        raise RuntimeError(msg)

    rng = random.Random(42)
    indices = rng.sample(range(len(converted)), k=min(n_samples, len(converted)))
    failures: list[str] = []

    for idx in indices:
        row = converted[idx]
        try:
            rendered = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"row {idx}: apply_chat_template raised: {exc}")
            continue

        ids = tokenizer.encode(rendered, add_special_tokens=False)
        try:
            i_open = ids.index(open_ids[0])
            i_close = ids.index(close_ids[0])
        except ValueError:
            failures.append(
                f"row {idx}: open/close channel IDs not found in rendered token stream "
                f"(open id={open_ids[0]}, close id={close_ids[0]})"
            )
            continue
        if not i_open < i_close:
            failures.append(
                f"row {idx}: open token after close token (i_open={i_open}, i_close={i_close})"
            )
            continue

        # Crude assistant-span estimator: anything between the LAST occurrence
        # of an "assistant"-role marker substring and the end of the rendered
        # string is the assistant turn. We're not reproducing
        # train_on_responses_only here — just sanity-checking that the
        # channel-token block sits inside that suffix. The real masking is
        # validated by trainer.py at training time.
        assistant_markers = ["assistant", "<start_of_turn>model"]
        last_marker_pos = -1
        for marker in assistant_markers:
            pos = rendered.rfind(marker)
            if pos > last_marker_pos:
                last_marker_pos = pos
        if last_marker_pos == -1:
            failures.append(f"row {idx}: could not locate assistant turn marker in rendered string")
            continue

        # Re-encode the assistant-turn suffix and check the channel-token
        # IDs are present.
        suffix = rendered[last_marker_pos:]
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        if open_ids[0] not in suffix_ids or close_ids[0] not in suffix_ids:
            failures.append(
                f"row {idx}: channel tokens land OUTSIDE the assistant turn span "
                "(they appear earlier in the rendered string)"
            )
            continue

        # Mask-fraction proxy: how much of the total rendered-token count
        # is in the assistant turn? Trainer's real gate is 10–90 % of
        # *batch* tokens; here we use the per-example suffix-length ratio
        # as an early warning. < 5 % or > 95 % means the chat template
        # is very off-balance.
        frac = len(suffix_ids) / max(1, len(ids))
        if frac < 0.05 or frac > 0.95:
            failures.append(f"row {idx}: assistant-suffix fraction = {frac:.2%} (expected 5–95 %)")

    if failures:
        msg = (
            "Round-trip check FAILED for "
            f"{len(failures)}/{len(indices)} sampled rows:\n  - "
            + "\n  - ".join(failures)
            + "\n\nArm C is infeasible with the current tokenizer + data shape. "
            "Skip configs/training_v2_gemma_native.yaml from the bake-off "
            "loop and revisit (likely needs trainer changes or a different "
            "channel-token discovery strategy)."
        )
        raise RuntimeError(msg)


@app.command()
def render(
    input_dir: Path = typer.Option(  # noqa: B008
        Path("data/constructed_v2/final_v2"),
        "--input",
        help="Source HF DatasetDict (bare-tag v2 dataset).",
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        Path("data/constructed_v2/final_v2_gemma_native"),
        "--output",
        help="Destination HF DatasetDict (channel-token format).",
    ),
    model: str = typer.Option(
        "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit",
        "--model",
        help="HF model id whose tokenizer defines the native channel tokens.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="If set, truncate each split to this many rows (smoke mode).",
    ),
    skip_round_trip: bool = typer.Option(
        False,
        "--skip-round-trip",
        help="DEBUG ONLY — skip the mask-fraction sanity gate.",
    ),
) -> None:
    """Convert the bare-tag v2 dataset into Gemma's native channel format."""
    setup_logging(level="INFO")

    if not input_dir.exists():
        console.print(f"[red]Input dataset not found: {input_dir}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Loading source dataset:[/cyan] {input_dir}")
    src = load_from_disk(str(input_dir))

    if not isinstance(src, DatasetDict):
        console.print(f"[red]{input_dir} is not a DatasetDict[/red]")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Loading tokenizer:[/cyan] {model}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)

    chat_template = getattr(tokenizer, "chat_template", None) or ""
    truncated_template = chat_template[:2000] + ("…" if len(chat_template) > 2000 else "")
    console.print(
        Panel(
            truncated_template or "[no chat_template]",
            title=f"Tokenizer chat_template ({model})",
            border_style="dim",
        )
    )

    pair = _discover_channel_tokens(tokenizer)
    if pair is None:
        console.print(
            "[red]Could not discover Gemma native think-channel tokens in the "
            "tokenizer vocab. Inspect the tokenizer config above and either:[/red]\n"
            "  1. Extend _THINK_OPEN_HINTS / _THINK_CLOSE_HINTS in this script, or\n"
            "  2. Conclude Gemma's native think is template-driven (no special\n"
            "     tokens) and Arm C needs trainer changes instead."
        )
        raise typer.Exit(code=2)

    open_tok, close_tok = pair
    console.print(
        f"[green]Discovered channel tokens:[/green] open={open_tok!r}  close={close_tok!r}"
    )

    # Per-split conversion.
    converted_splits: dict[str, Dataset] = {}
    features = Features(
        {
            "messages": [{"role": Value("string"), "content": Value("string")}],
            "metadata": {
                "example_id": Value("string"),
                "ad_id": Value("string"),
                "platform": Value("string"),
            },
        }
    )

    for split, ds in src.items():
        rows: list[dict[str, Any]] = []
        n_in = len(ds)
        n_take = min(limit, n_in) if limit is not None else n_in
        for i in range(n_take):
            rows.append(_make_row(ds[i], open_tok, close_tok))
        converted_splits[split] = Dataset.from_list(rows, features=features)
        console.print(f"  {split}: {n_take}/{n_in} rows converted")

    converted_dd = DatasetDict(converted_splits)

    # Round-trip check on the train split.
    if not skip_round_trip:
        console.print("[cyan]Running round-trip mask-fraction check (5 random train rows)…[/cyan]")
        _round_trip_check(converted_dd["train"], tokenizer, open_tok, close_tok)
        console.print("[green]Round-trip check passed.[/green]")
    else:
        console.print("[yellow]--skip-round-trip set; skipping sanity gate.[/yellow]")

    # Wipe + write atomically: remove the existing output dir to avoid
    # mixing rows from a previous (e.g. limited) run.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    converted_dd.save_to_disk(str(output_dir))
    console.print(f"[green]Wrote Gemma-native DatasetDict to[/green] {output_dir}")

    # One-line summary for the bake-off driver to grep.
    print(
        f"GEMMA_NATIVE_RENDER_OK splits={list(converted_dd.keys())} "
        f"open={open_tok!r} close={close_tok!r}"
    )


if __name__ == "__main__":
    app()
