"""Local dev server for the scoring predictor.

Loads the on-disk checkpoint and serves the same FastAPI app the Modal
deployment does, so the Next.js frontend can talk to the predictor with
identical config — only the base URL differs.

Run via the dev dashboard (``infra/start-dev.sh`` Service #5) or directly:

    uv run python scripts/serve_scoring_predictor.py \
        --checkpoint data/scoring_predictor/checkpoints/random/best \
        --port 8001 \
        --api-key "$SCORING_PREDICTOR_API_KEY"

The script exits non-zero with a clear message if the checkpoint is missing
— the dashboard renders that as a "stopped" service instead of failing
silently.
"""

from __future__ import annotations

from pathlib import Path

import typer
import uvicorn

from draper.scoring_predictor import load_predictor
from draper.scoring_predictor.server import build_app

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    help="Local dev server for the scoring predictor.",
)


DEFAULT_CHECKPOINT = "data/scoring_predictor/checkpoints/random/best"


@app.command()
def serve(
    checkpoint: str = typer.Option(
        DEFAULT_CHECKPOINT,
        "--checkpoint",
        "-c",
        help="Path to a trained checkpoint directory.",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(8001, "--port", "-p", help="Bind port."),
    api_key: str = typer.Option(
        ...,
        "--api-key",
        "-k",
        envvar="SCORING_PREDICTOR_API_KEY",
        help="Shared secret expected on the X-API-Key header.",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Torch device. Default cpu — predictor runs ~30ms/ad on CPU.",
    ),
) -> None:
    """Start the local scoring server."""
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        typer.echo(
            f"checkpoint not found: {checkpoint_path}\n"
            f"Train one first: uv run scripts/predict.py train --split random",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"loading predictor from {checkpoint_path} (device={device}) …")
    predictor = load_predictor(checkpoint_path, device=device)
    typer.echo("predictor loaded; building app")

    fastapi_app = build_app(
        predictor=predictor,
        api_key=api_key,
        checkpoint_label=str(checkpoint_path),
    )

    typer.echo(f"serving on http://{host}:{port}  (POST /score, GET /healthz)")
    # log_level=warning to keep the dashboard log file readable; per-request
    # access logs would dominate the file. Errors and startup still surface.
    uvicorn.run(fastapi_app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    app()
