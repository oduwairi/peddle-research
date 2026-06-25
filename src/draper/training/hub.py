"""Push training artifacts to the Hugging Face Hub.

The push is opt-in: callers only invoke these helpers when the user wants
the adapter / merged weights stashed on the Hub (typically to keep them
around after a cloud GPU is torn down). Authentication is via the standard
``HF_TOKEN`` env var that ``huggingface-cli login`` already respects.
"""

from __future__ import annotations

import os
from pathlib import Path

from draper.utils.logging import get_logger

logger = get_logger("draper.training")


def push_folder_to_hub(
    folder: str | Path,
    repo_id: str,
    *,
    path_in_repo: str = "",
    private: bool = True,
    commit_message: str | None = None,
) -> str:
    """Upload a directory to a HF Hub model repo.

    Creates the repo if it doesn't exist (private by default). Returns the
    canonical https URL of the repo so callers can log it.
    """
    from huggingface_hub import HfApi, create_repo

    src = Path(folder)
    if not src.is_dir():
        msg = f"Folder not found: {src}"
        raise FileNotFoundError(msg)

    token = os.getenv("HF_TOKEN")
    create_repo(repo_id, private=private, exist_ok=True, repo_type="model", token=token)

    api = HfApi(token=token)
    msg_text = commit_message or f"Upload {src.name}"
    logger.info("Pushing %s -> %s (path_in_repo=%r)", src, repo_id, path_in_repo)
    api.upload_folder(
        folder_path=str(src),
        repo_id=repo_id,
        path_in_repo=path_in_repo or None,
        commit_message=msg_text,
        repo_type="model",
    )
    url = f"https://huggingface.co/{repo_id}"
    logger.info("Push complete: %s", url)
    return url
