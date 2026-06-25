"""Batch-API judge runs — cuts judge cost by ~50% at the price of 24h latency.

Both OpenAI and Anthropic offer first-party batch APIs with a flat 50%
discount and a 24-hour completion SLA. Gemini's "batch prediction" surface
exists but isn't a flat discount the same way (it's tied to context-cache
billing) — for now this module only supports the two providers that give
a clean 50% off.

Workflow:
    1. Build batch request files: one JSONL line per (pair, example_id, ordering).
       ``build_openai_batch_jsonl`` / ``build_anthropic_batch_requests``.
    2. Submit via ``submit_openai_batch`` / ``submit_anthropic_batch``;
       returns the provider's batch ID. Persist the ID + a manifest of
       custom_id → (pair, example_id, swap_order) mappings to disk.
    3. Wait ~24h (or poll ``status_openai_batch`` / ``status_anthropic_batch``).
    4. ``collect_openai_batch`` / ``collect_anthropic_batch`` pulls results,
       parses provider-specific response shapes back into ``Judgment``
       objects, and writes per-pair judgment JSONs in the SAME shape as
       the live judge-pass output — so ``aggregate`` reads either path
       identically.

custom_id encoding (both providers):
    f"{pair_a}__vs__{pair_b}__{example_id}__{ordering}"
where ordering is "fwd" (config_a as A) or "swp" (config_a as B).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import anthropic
import openai
from anthropic.types import ToolUseBlock

from ..schemas import (
    Brief,
    Inference,
    JudgePerDimension,
    Judgment,
    UrlScenario,
    Winner,
)
from .clients import (
    ANTHROPIC_MAX_TOKENS,
    OPENAI_MAX_TOKENS,
    BatchProvider,
    clip_score,
    provider_for_model,  # noqa: F401 — re-exported; eval.py imports this from batch
)
from .normalize import judge_input_text
from .prompts import (
    PAIRWISE_JSON_SCHEMA,
    PAIRWISE_SYSTEM,
    build_pairwise_user_prompt,
)

# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchRequestKey:
    """Identifies one judge call inside a batch — round-trippable to/from custom_id.

    Format: ``{pair_a}__vs__{pair_b}__[{example_id}]__{ordering}``
    where example_id is URL-encoded to safely handle '__' substrings.
    """

    pair_a: str
    pair_b: str
    example_id: str
    swap_order: bool

    def custom_id(self) -> str:
        ordering = "swp" if self.swap_order else "fwd"
        # URL-encode example_id to handle any special chars including '__'
        encoded_id = self.example_id.replace("__", "%5f%5f").replace("_", "%5f")
        return f"{self.pair_a}__vs__{self.pair_b}__{encoded_id}__{ordering}"

    @classmethod
    def parse(cls, custom_id: str) -> BatchRequestKey:
        """Parse custom_id back to (pair_a, pair_b, example_id, swap_order).

        Raises ValueError if format is invalid or example_id cannot be decoded.
        """
        # Split from the ends: last part is ordering, second-to-last is example_id
        parts = custom_id.split("__")
        if len(parts) < 4:
            raise ValueError(
                f"Cannot parse custom_id {custom_id!r}: expected at least 4 parts, got {len(parts)}"
            )

        ordering = parts[-1]
        if ordering not in ("fwd", "swp"):
            raise ValueError(
                f"Cannot parse custom_id {custom_id!r}: invalid ordering {ordering!r}, "
                "expected 'fwd' or 'swp'"
            )

        encoded_id = parts[-2]
        pair_a = parts[0]

        if parts[1] != "vs":
            raise ValueError(
                f"Cannot parse custom_id {custom_id!r}: expected 'vs' at position 1, "
                f"got {parts[1]!r}"
            )

        pair_b = parts[2]

        # Decode example_id: %5f%5f → '__', %5f → '_'
        try:
            example_id = encoded_id.replace("%5f%5f", "__").replace("%5f", "_")
        except Exception as exc:
            raise ValueError(f"Cannot decode example_id in custom_id {custom_id!r}: {exc}") from exc

        return cls(
            pair_a=pair_a,
            pair_b=pair_b,
            example_id=example_id,
            swap_order=(ordering == "swp"),
        )


@dataclass
class BatchManifest:
    """Round-trip metadata: custom_id → routing + canonical pair frame.

    Persisted alongside the provider's batch_id so ``collect`` knows which
    judgments map back to which (config_a, config_b, example_id) and how
    to reconcile ordering into canonical frame.
    """

    provider: BatchProvider
    judge_model: str
    pair: tuple[str, str]
    keys: list[BatchRequestKey] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_json(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "judge_model": self.judge_model,
            "pair": list(self.pair),
            "created_at": self.created_at,
            "keys": [
                {
                    "pair_a": k.pair_a,
                    "pair_b": k.pair_b,
                    "example_id": k.example_id,
                    "swap_order": k.swap_order,
                }
                for k in self.keys
            ],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> BatchManifest:
        return cls(
            provider=cast(BatchProvider, data["provider"]),
            judge_model=str(data["judge_model"]),
            pair=(str(data["pair"][0]), str(data["pair"][1])),
            keys=[
                BatchRequestKey(
                    pair_a=k["pair_a"],
                    pair_b=k["pair_b"],
                    example_id=k["example_id"],
                    swap_order=bool(k["swap_order"]),
                )
                for k in data.get("keys", [])
            ],
            created_at=str(data.get("created_at") or datetime.now(UTC).isoformat()),
        )


def _user_prompt_for(
    *,
    example_id: str,
    a: Inference,
    b: Inference,
    swap_order: bool,
    briefs_by_id: dict[str, Brief] | None,
    scenarios_by_id: dict[str, UrlScenario] | None,
    clean_root: Path | None = None,
) -> str:
    """Build the same user prompt the live judge would build.

    When ``clean_root`` is provided, candidates are read from the LLM
    extractor's cache (uniform shape across configs); otherwise the
    regex fallback ``clean_copy(assistant_text)`` is used per-candidate.
    """
    platform = ""
    vertical = ""
    user = a.brief
    if briefs_by_id and example_id in briefs_by_id:
        br = briefs_by_id[example_id]
        platform, vertical, user = br.platform, br.vertical, br.user
    elif scenarios_by_id and example_id in scenarios_by_id:
        sc = scenarios_by_id[example_id]
        platform, vertical, user = sc.platform, sc.vertical, sc.user_prompt
    left, right = (a, b) if not swap_order else (b, a)
    return build_pairwise_user_prompt(
        platform=platform,
        vertical=vertical,
        user_prompt=user,
        response_a=judge_input_text(left, clean_root=clean_root),
        response_b=judge_input_text(right, clean_root=clean_root),
    )


def _request_keys(
    *,
    pair: tuple[str, str],
    inferences_a: dict[str, Inference],
    inferences_b: dict[str, Inference],
    swap: bool,
) -> list[BatchRequestKey]:
    common = sorted(set(inferences_a) & set(inferences_b))
    keys: list[BatchRequestKey] = []
    for ex_id in common:
        keys.append(
            BatchRequestKey(pair_a=pair[0], pair_b=pair[1], example_id=ex_id, swap_order=False)
        )
        if swap:
            keys.append(
                BatchRequestKey(pair_a=pair[0], pair_b=pair[1], example_id=ex_id, swap_order=True)
            )
    return keys


# ---------------------------------------------------------------------------
# OpenAI batch
# ---------------------------------------------------------------------------


def build_openai_batch_jsonl(
    *,
    pair: tuple[str, str],
    judge_model: str,
    inferences_a: dict[str, Inference],
    inferences_b: dict[str, Inference],
    briefs_by_id: dict[str, Brief] | None,
    scenarios_by_id: dict[str, UrlScenario] | None,
    swap: bool = True,
    clean_root: Path | None = None,
) -> tuple[list[dict[str, Any]], BatchManifest]:
    """Produce JSONL request lines for the OpenAI Batch API.

    Each line is a chat-completions request with ``response_format`` set to
    the same json_schema used by the live judge. Returns ``(lines, manifest)``;
    callers write the lines to a .jsonl file and persist the manifest.

    ``clean_root`` enables the LLM ad-copy extractor — see ``judge_pair``.
    """
    keys = _request_keys(pair=pair, inferences_a=inferences_a, inferences_b=inferences_b, swap=swap)
    lines: list[dict[str, Any]] = []
    for key in keys:
        a = inferences_a[key.example_id]
        b = inferences_b[key.example_id]
        user = _user_prompt_for(
            example_id=key.example_id,
            a=a,
            b=b,
            swap_order=key.swap_order,
            briefs_by_id=briefs_by_id,
            scenarios_by_id=scenarios_by_id,
            clean_root=clean_root,
        )
        lines.append(
            {
                "custom_id": key.custom_id(),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": judge_model,
                    "messages": [
                        {"role": "system", "content": PAIRWISE_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": PAIRWISE_JSON_SCHEMA,
                    },
                    "temperature": 0.0,
                    "max_completion_tokens": OPENAI_MAX_TOKENS,
                },
            }
        )
    manifest = BatchManifest(provider="openai", judge_model=judge_model, pair=pair, keys=keys)
    return lines, manifest


def submit_openai_batch(
    *,
    jsonl_path: Path,
    description: str | None = None,
) -> str:
    """Upload the JSONL file and create a batch. Returns the batch ID.

    Caller is responsible for writing ``jsonl_path`` first (one request
    per line) and for persisting the returned batch_id alongside the
    manifest.

    Raises on upload or batch creation failure. Logs the upload file ID
    on batch.create failure so the file can be recovered if needed.
    """
    import logging

    logger = logging.getLogger(__name__)
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    with jsonl_path.open("rb") as f:
        upload = client.files.create(file=f, purpose="batch")
    logger.info(f"Uploaded {jsonl_path.name} → OpenAI file ID {upload.id}")
    try:
        batch = client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description or jsonl_path.stem},
        )
    except Exception as exc:
        logger.error(
            f"batch.create failed for upload {upload.id} — file is orphaned. "
            f"Retrieve results via: client.files.content('{upload.id}'). "
            f"Error: {exc}"
        )
        raise
    return batch.id


def status_openai_batch(batch_id: str) -> dict[str, Any]:
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    batch = client.batches.retrieve(batch_id)
    return {
        "id": batch.id,
        "status": batch.status,
        "request_counts": (
            batch.request_counts.model_dump() if batch.request_counts is not None else None
        ),
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
    }


def collect_openai_batch(batch_id: str) -> dict[str, dict[str, Any]]:
    """Pull a completed OpenAI batch's output. Returns ``{custom_id: parsed_json}``.

    Raises if the batch isn't in a terminal state. Errored requests are
    skipped (logged via dict absence) — caller should compare returned
    keys against the manifest to find missing rows.
    """
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        raise RuntimeError(f"OpenAI batch {batch_id} is not completed (status={batch.status!r})")
    if not batch.output_file_id:
        return {}
    raw = client.files.content(batch.output_file_id).read().decode("utf-8")
    out: dict[str, dict[str, Any]] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        custom_id = row.get("custom_id")
        body = row.get("response", {}).get("body") or {}
        choices = body.get("choices") or []
        if not custom_id or not choices:
            continue
        content = choices[0].get("message", {}).get("content") or "{}"
        try:
            out[custom_id] = json.loads(content)
        except json.JSONDecodeError:
            continue
    return out


# ---------------------------------------------------------------------------
# Anthropic batch
# ---------------------------------------------------------------------------


def build_anthropic_batch_requests(
    *,
    pair: tuple[str, str],
    judge_model: str,
    inferences_a: dict[str, Inference],
    inferences_b: dict[str, Inference],
    briefs_by_id: dict[str, Brief] | None,
    scenarios_by_id: dict[str, UrlScenario] | None,
    swap: bool = True,
    clean_root: Path | None = None,
) -> tuple[list[dict[str, Any]], BatchManifest]:
    """Produce request dicts for ``client.messages.batches.create``.

    Uses tool-use forcing for structured output — same shape as the live
    Anthropic judge in ``pairwise._call_anthropic_judge``.

    ``clean_root`` enables the LLM ad-copy extractor — see ``judge_pair``.
    """
    keys = _request_keys(pair=pair, inferences_a=inferences_a, inferences_b=inferences_b, swap=swap)
    schema = cast(dict[str, Any], PAIRWISE_JSON_SCHEMA["schema"])
    requests: list[dict[str, Any]] = []
    for key in keys:
        a = inferences_a[key.example_id]
        b = inferences_b[key.example_id]
        user = _user_prompt_for(
            example_id=key.example_id,
            a=a,
            b=b,
            swap_order=key.swap_order,
            briefs_by_id=briefs_by_id,
            scenarios_by_id=scenarios_by_id,
            clean_root=clean_root,
        )
        requests.append(
            {
                "custom_id": key.custom_id(),
                "params": {
                    "model": judge_model,
                    "max_tokens": ANTHROPIC_MAX_TOKENS,
                    "temperature": 0.0,
                    "system": PAIRWISE_SYSTEM,
                    "messages": [{"role": "user", "content": user}],
                    "tools": [
                        {
                            "name": "submit_judgment",
                            "description": "Submit your scored pairwise judgment.",
                            "input_schema": schema,
                        }
                    ],
                    "tool_choice": {"type": "tool", "name": "submit_judgment"},
                },
            }
        )
    manifest = BatchManifest(provider="anthropic", judge_model=judge_model, pair=pair, keys=keys)
    return requests, manifest


def submit_anthropic_batch(*, requests: list[dict[str, Any]]) -> str:
    """Create an Anthropic message-batch from the request list. Returns the batch ID."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    batch = client.messages.batches.create(requests=requests)  # type: ignore[arg-type]
    return batch.id


def status_anthropic_batch(batch_id: str) -> dict[str, Any]:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    batch = client.messages.batches.retrieve(batch_id)
    return {
        "id": batch.id,
        "processing_status": batch.processing_status,
        "request_counts": (
            batch.request_counts.model_dump() if getattr(batch, "request_counts", None) else None
        ),
    }


def collect_anthropic_batch(batch_id: str) -> dict[str, dict[str, Any]]:
    """Pull a completed Anthropic batch's results. Returns ``{custom_id: tool_use_input}``.

    Raises if not in a terminal state. Errored / non-tool-use rows are
    skipped — caller should diff against manifest for missing keys.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        raise RuntimeError(
            f"Anthropic batch {batch_id} not finished "
            f"(processing_status={batch.processing_status!r})"
        )
    out: dict[str, dict[str, Any]] = {}
    for entry in client.messages.batches.results(batch_id):
        custom_id = entry.custom_id
        if entry.result.type != "succeeded":
            continue
        message = entry.result.message
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                out[custom_id] = cast(dict[str, Any], block.input)
                break
    return out


# ---------------------------------------------------------------------------
# Decoding parsed batch results into Judgment objects
# ---------------------------------------------------------------------------


def parsed_to_judgments(
    *,
    parsed: dict[str, dict[str, Any]],
    manifest: BatchManifest,
) -> tuple[list[Judgment], list[str]]:
    """Convert provider-parsed JSON back into ``Judgment`` objects.

    The judgments are emitted in the *judge's frame* (i.e., A=left,
    B=right at the moment of the call), matching what the live
    ``judge_pair`` returns. ``reconcile_pair`` later maps both orderings
    back to canonical (config_a, config_b).

    Returns: (judgments, missing_custom_ids)
      - judgments: list of successfully parsed Judgment objects
      - missing_custom_ids: list of custom_ids from manifest that weren't
        in the parsed results (failed requests, parsing errors, etc.)
    """
    import logging

    logger = logging.getLogger(__name__)
    by_key = {key.custom_id(): key for key in manifest.keys}
    judgments: list[Judgment] = []
    now = datetime.now(UTC)
    missing: list[str] = []

    for custom_id, key in by_key.items():
        data = parsed.get(custom_id)
        if data is None:
            missing.append(custom_id)
            continue
        per_dim = JudgePerDimension(
            strategic_relevance=clip_score(data.get("strategic_relevance", 0)),
            creativity=clip_score(data.get("creativity", 0)),
            actionability=clip_score(data.get("actionability", 0)),
            channel_appropriateness=clip_score(data.get("channel_appropriateness", 0)),
            predicted_performance=clip_score(data.get("predicted_performance", 0)),
        )
        winner_raw = data.get("overall_winner", "tie")
        winner: Winner = winner_raw if winner_raw in ("A", "B", "tie") else "tie"
        # In the swapped ordering the judge sees pair_b as A; pair_a as B.
        if key.swap_order:
            judge_pair_a, judge_pair_b = key.pair_b, key.pair_a
        else:
            judge_pair_a, judge_pair_b = key.pair_a, key.pair_b
        judgments.append(
            Judgment(
                example_id=key.example_id,
                pair_a=judge_pair_a,
                pair_b=judge_pair_b,
                swap_order=key.swap_order,
                judge_model=manifest.judge_model,
                per_dim=per_dim,
                overall_winner=winner,
                rationale=str(data.get("rationale", "")),
                raw_response=json.dumps(data),
                created_at=now,
            )
        )

    if missing:
        logger.warning(
            f"Missing {len(missing)} results from batch (of {len(by_key)} total): "
            f"check provider error file or logs. "
            f"First few: {missing[:5]}"
        )

    return judgments, missing
