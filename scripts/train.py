"""QLoRA fine-tuning CLI.

Usage:
    python scripts/train.py inspect                  # Render one templated example
    python scripts/train.py smoke --dry-run          # CPU sanity check (no model load)
    python scripts/train.py smoke                    # Tiny model, 2 steps, ~1 min on any GPU
    python scripts/train.py train                    # Full QLoRA run on Qwen3-8B
    python scripts/train.py train --resume           # Resume from latest checkpoint
    python scripts/train.py merge --adapter PATH     # Merge LoRA into base for vLLM
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

# ---------------------------------------------------------------------------
# Ensure project root is importable when run as a script
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from draper.training import (  # noqa: E402
    Trainer,
    TrainingConfig,
    load_dataset_dict,
    merge_adapter,
    push_folder_to_hub,
    render_first_example,
)
from draper.utils.logging import setup_logging  # noqa: E402

app = typer.Typer(help="QLoRA fine-tuning pipeline.")
console = Console()


def _load(config_path: Path) -> TrainingConfig:
    if not config_path.exists():
        console.print(f"[red]Config not found: {config_path}[/red]")
        raise typer.Exit(code=1)
    return TrainingConfig.from_yaml(config_path)


@app.command()
def inspect(
    config: Path = typer.Option(  # noqa: B008
        Path("configs/training.yaml"), help="Training config YAML."
    ),
    split: str = typer.Option("train", help="Which split to render the first row of."),
) -> None:
    """Render the first dataset row through the tokenizer's chat_template.

    Useful for eyeballing that ``<|im_start|>assistant ... <|im_end|>``
    boundaries are intact so ``assistant_only_loss`` masks the right region.
    """
    setup_logging(level="INFO")
    cfg = _load(config)

    ds = load_dataset_dict(cfg.dataset_dir)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    rendered = render_first_example(ds, tokenizer, split=split)
    console.print(
        Panel(
            rendered,
            title=f"Templated example ({split}[0]) via {cfg.base_model}",
            border_style="cyan",
        )
    )


@app.command()
def smoke(
    config: Path = typer.Option(  # noqa: B008
        Path("configs/training.yaml"), help="Training config YAML."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Skip model load and .train() — validate config + dataset shape only.",
    ),
    push: bool = typer.Option(
        False,
        "--push",
        help="After smoke training, push the (tiny) adapter to "
        "$HF_HUB_REPO/smoke-test/ — validates HF Hub auth + repo write end-to-end.",
    ),
) -> None:
    """Run the trainer with the tiny smoke model (smoke overrides in YAML).

    With ``--dry-run`` no model is loaded, so this works on a CPU-only laptop.
    """
    setup_logging(level="INFO")
    cfg = _load(config)
    trainer = Trainer(cfg, smoke=True)
    setup = trainer.setup(dry_run=dry_run)
    console.print(
        f"[green]Smoke setup OK[/green]  run_dir={setup.run_dir}  "
        f"train_rows={len(setup.dataset['train'])}  "
        f"val_rows={len(setup.dataset['validation'])}"
    )
    if dry_run:
        console.print("[yellow]--dry-run: skipping training step[/yellow]")
        return
    final_dir = trainer.train()
    console.print("[green]Smoke training step completed.[/green]")

    if push:
        repo = os.getenv("HF_HUB_REPO")
        if not repo:
            console.print(
                "[red]--push set but $HF_HUB_REPO is empty; skipping upload.[/red]"
            )
            raise typer.Exit(code=1)
        url = push_folder_to_hub(
            final_dir,
            repo,
            path_in_repo="smoke-test",
            commit_message=f"Smoke push from {final_dir.name} (validates HF auth)",
        )
        console.print(f"[green]Smoke adapter pushed:[/green] {url}/tree/main/smoke-test")


@app.command()
def train(
    config: Path = typer.Option(  # noqa: B008
        Path("configs/training.yaml"), help="Training config YAML."
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Resume from the latest checkpoint in output_dir."
    ),
    push: bool = typer.Option(
        False,
        "--push",
        help="After training, push the final adapter to $HF_HUB_REPO on HF Hub.",
    ),
) -> None:
    """Full QLoRA run on the configured base model."""
    setup_logging(level="INFO")
    cfg = _load(config)
    trainer = Trainer(cfg, smoke=False)
    trainer.setup()
    final_dir = trainer.train(resume=resume)
    console.print(f"[green]Training complete.[/green] Adapter: {final_dir}")

    if push:
        repo = os.getenv("HF_HUB_REPO")
        if not repo:
            console.print(
                "[red]--push set but $HF_HUB_REPO is empty; skipping upload.[/red]"
            )
            raise typer.Exit(code=1)
        url = push_folder_to_hub(
            final_dir,
            repo,
            path_in_repo="adapter",
            commit_message=f"Adapter from run {final_dir.name}",
        )
        console.print(f"[green]Adapter pushed:[/green] {url}/tree/main/adapter")


@app.command()
def merge(
    adapter: Path = typer.Option(  # noqa: B008
        ..., "--adapter", help="Path to the trained adapter directory."
    ),
    out: Path | None = typer.Option(  # noqa: B008
        None, "--out", help="Output directory for merged weights (defaults to config.merged_dir)."
    ),
    base: str | None = typer.Option(
        None, "--base", help="Base model id (defaults to config.base_model)."
    ),
    config: Path = typer.Option(  # noqa: B008
        Path("configs/training.yaml"), help="Training config YAML."
    ),
    save_method: str = typer.Option(
        "merged_16bit", help="Unsloth save method: merged_16bit | merged_4bit | lora."
    ),
    push: bool = typer.Option(
        False,
        "--push",
        help="After merging, push merged weights to $HF_HUB_REPO on HF Hub.",
    ),
) -> None:
    """Merge a trained adapter into the base model and save for vLLM."""
    setup_logging(level="INFO")
    cfg = _load(config)
    base_model = base or cfg.base_model
    out_dir = out or Path(cfg.merged_dir)
    if save_method not in {"merged_16bit", "merged_4bit", "lora"}:
        console.print(f"[red]Invalid save_method: {save_method}[/red]")
        raise typer.Exit(code=1)
    merged = merge_adapter(
        adapter,
        base_model,
        out_dir,
        max_seq_length=cfg.max_length,
        save_method=save_method,  # type: ignore[arg-type]
    )
    console.print(f"[green]Merged model saved to[/green] {merged}")

    if push:
        repo = os.getenv("HF_HUB_REPO")
        if not repo:
            console.print(
                "[red]--push set but $HF_HUB_REPO is empty; skipping upload.[/red]"
            )
            raise typer.Exit(code=1)
        url = push_folder_to_hub(
            merged,
            repo,
            path_in_repo="merged",
            commit_message=f"Merged weights ({save_method})",
        )
        console.print(f"[green]Merged model pushed:[/green] {url}/tree/main/merged")


if __name__ == "__main__":
    app()
