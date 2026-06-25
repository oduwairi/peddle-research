"""Ad data collection — scrapers, API clients, and shared schemas.

Primary client: ``AdFlexClient`` (async httpx, cursor pagination, rate limiting).
Supplementary scrapers: Meta, Google, TikTok, BigSpy.
All sources normalize to the ``RawAd`` Pydantic schema in ``schemas.py``.
"""
