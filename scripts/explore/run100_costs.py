"""Re-fetch run100 batches from each provider and tally token usage + cost.

One-shot diagnostic. Not part of the construction CLI.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from draper.construction.batch import make_batch_client

SUBMISSIONS = Path("data/constructed_v2/runs/run100/submissions.json")

# Batch-tier (50% off list) USD per 1M tokens. Sources:
# - Anthropic claude-sonnet-4-6: list $3 in / $15 out → batch $1.50 / $7.50
# - OpenAI gpt-5.4 (full, non-mini): list $1.25 in / $10 out → batch $0.625 / $5.00
# - Google gemini-3.1-pro-preview: list $1.25 in / $10 out (≤200K context) → batch $0.625 / $5.00
RATES_BATCH_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (1.50, 7.50),
    "gpt-5.4": (0.625, 5.00),
    "gemini-3.1-pro-preview": (0.625, 5.00),
}


async def tally(submission: dict[str, object]) -> dict[str, object]:
    model = str(submission["model"])
    batch_id = str(submission["batch_id"])
    client = make_batch_client(model)
    results = await client.fetch_results(batch_id)
    in_tok: int = 0
    out_tok: int = 0
    ok: int = 0
    errors: int = 0
    for r in results:
        if r.error:
            errors += 1
            continue
        ok += 1
        in_tok += r.input_tokens
        out_tok += r.output_tokens
    rate_in, rate_out = RATES_BATCH_PER_MTOK[model]
    usd = (in_tok / 1_000_000) * rate_in + (out_tok / 1_000_000) * rate_out
    return {
        "provider": submission["provider"],
        "model": model,
        "batch_id": batch_id,
        "requests": submission["request_count"],
        "ok": ok,
        "errors": errors,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "usd_in": (in_tok / 1_000_000) * rate_in,
        "usd_out": (out_tok / 1_000_000) * rate_out,
        "usd_total": usd,
    }


async def main() -> None:
    submissions = json.loads(SUBMISSIONS.read_text())
    rows = await asyncio.gather(*(tally(s) for s in submissions))
    total_usd: float = 0.0
    total_in: int = 0
    total_out: int = 0
    print(
        f"{'provider':<10} {'model':<28} {'ok':>4} {'err':>4} "
        f"{'in_tok':>10} {'out_tok':>10} {'usd_in':>8} {'usd_out':>8} {'usd':>8}"
    )
    for r in rows:
        provider = str(r["provider"])
        model = str(r["model"])
        ok = int(r["ok"]) if isinstance(r["ok"], int) else 0
        errors = int(r["errors"]) if isinstance(r["errors"], int) else 0
        in_tokens = int(r["input_tokens"]) if isinstance(r["input_tokens"], int) else 0
        out_tokens = int(r["output_tokens"]) if isinstance(r["output_tokens"], int) else 0
        usd_in = float(r["usd_in"]) if isinstance(r["usd_in"], (int, float)) else 0.0
        usd_out = float(r["usd_out"]) if isinstance(r["usd_out"], (int, float)) else 0.0
        usd_total = float(r["usd_total"]) if isinstance(r["usd_total"], (int, float)) else 0.0
        print(
            f"{provider:<10} {model:<28} {ok:>4} {errors:>4} "
            f"{in_tokens:>10,} {out_tokens:>10,} "
            f"{usd_in:>8.3f} {usd_out:>8.3f} {usd_total:>8.3f}"
        )
        total_usd += usd_total
        total_in += in_tokens
        total_out += out_tokens
    print(
        f"{'TOTAL':<10} {'':<28} {'':>4} {'':>4} "
        f"{total_in:>10,} {total_out:>10,} {'':>8} {'':>8} {total_usd:>8.3f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
