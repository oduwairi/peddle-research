"""Tests for async rate limiter."""

import time

import pytest

from draper.scraping.rate_limiter import RateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_burst() -> None:
    """Burst requests should complete immediately."""
    limiter = RateLimiter(requests_per_minute=60, burst_size=5)
    start = time.monotonic()
    for _ in range(5):
        await limiter.acquire()
    elapsed = time.monotonic() - start
    assert elapsed < 0.5  # burst should be near-instant


@pytest.mark.asyncio
async def test_rate_limiter_context_manager() -> None:
    limiter = RateLimiter(requests_per_minute=60, burst_size=3)
    async with limiter:
        pass  # should not raise
