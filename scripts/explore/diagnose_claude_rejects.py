"""One-off: refetch a completed Anthropic batch and dump rejected responses.

Sidecars (and thus source_ad_ids) are gone from the registry once
``batch-collect`` succeeds, so we can't run the per-format ingestion
check here. Instead we just parse the model's tagged response and show
the assistant_response so a human can judge whether Claude:

  - reproduced the real ad verbatim    → our check is fine
  - paraphrased the ad                 → model's fault (style transfer)
  - wrote rationale-only / fresh copy  → model's fault (ignored prompt)

Usage:
    uv run python scripts/explore/diagnose_claude_rejects.py BATCH_ID
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from draper.construction.batch.anthropic_client import AnthropicBatchClient
from draper.construction.bundle import parse_bundle_output


async def main(batch_id: str) -> None:
    client = AnthropicBatchClient()
    responses = await client.fetch_results(batch_id)
    print(f"Fetched {len(responses)} responses from {batch_id}\n")

    for resp in responses:
        parsed = parse_bundle_output(resp.content)
        has_tags = bool(parsed.user_prompt and parsed.assistant_response)
        print("=" * 80)
        print(f"custom_id: {resp.custom_id}")
        print(f"output_tokens: {resp.output_tokens}")
        print(f"has_tags: {has_tags}  error: {resp.error or '-'}")
        print("-" * 80)
        if has_tags:
            print("USER_PROMPT:")
            print(parsed.user_prompt[:500])
            print()
            print("ASSISTANT_RESPONSE:")
            print(parsed.assistant_response[:1500])
        else:
            print("RAW (first 2000 chars):")
            print(resp.content[:2000])
        print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
