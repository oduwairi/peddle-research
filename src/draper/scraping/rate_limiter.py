"""Async token bucket rate limiter for API requests."""

from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """Token bucket rate limiter for async API calls.

    Args:
        requests_per_minute: Sustained request rate.
        burst_size: Maximum burst of concurrent requests.
    """

    def __init__(self, requests_per_minute: int = 30, burst_size: int = 5) -> None:
        self._rate = requests_per_minute / 60.0  # tokens per second
        self._max_tokens = float(burst_size)
        self._tokens = float(burst_size)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it."""
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Calculate wait time for next token
                wait = (1.0 - self._tokens) / self._rate

            await asyncio.sleep(wait)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._rate)
        self._last_refill = now

    async def __aenter__(self) -> RateLimiter:
        await self.acquire()
        return self

    async def __aexit__(self, *args: object) -> None:
        pass
