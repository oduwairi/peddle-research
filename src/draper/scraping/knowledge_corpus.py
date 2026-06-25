"""Marketing knowledge corpus extractor.

Pipeline: URL list → trafilatura extraction → Claude structured extraction → JSONL
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import trafilatura

from draper.scraping.schemas import KnowledgeArticle
from draper.utils.llm_client import complete

logger = logging.getLogger("draper")

EXTRACTION_SYSTEM_PROMPT = (
    "You are a marketing knowledge extractor. Given the text content of a "
    "marketing article, case study, or expert blog post, extract structured "
    "information.\n\n"
    "Return a JSON object with these fields:\n"
    '- "title": article title\n'
    '- "topic": primary topic (e.g. "channel selection", '
    '"audience targeting", "creative optimization")\n'
    '- "channel": list of marketing channels discussed '
    '(e.g. ["facebook", "google_ads", "tiktok"])\n'
    '- "strategies": list of specific strategies or tactics described\n'
    '- "frameworks": list of marketing frameworks referenced '
    '(e.g. ["AIDA", "PAS", "BAB"])\n'
    '- "metrics": dict of quantified results mentioned '
    '(e.g. {"roas": 3.2, "ctr_percent": 2.5})\n'
    '- "key_insights": list of 3-5 key takeaways from the article\n\n'
    "Return ONLY valid JSON, no markdown or explanation."
)

EXTRACTION_MODEL = "claude-haiku-4-5-20251001"


async def extract_from_url(url: str, client: httpx.AsyncClient | None = None) -> str | None:
    """Fetch a URL and extract main text content using trafilatura.

    Returns:
        Extracted text content, or None if extraction fails.
    """
    try:
        if client is None:
            async with httpx.AsyncClient(timeout=15.0) as c:
                response = await c.get(url, follow_redirects=True)
                html = response.text
        else:
            response = await client.get(url, follow_redirects=True)
            html = response.text

        result = trafilatura.extract(html, include_comments=False, include_tables=True)
        return str(result) if result else None
    except Exception as e:
        logger.warning(f"Failed to extract content from {url}: {e}")
        return None


async def structure_content(
    raw_text: str,
    url: str,
    source_name: str = "",
    model: str = EXTRACTION_MODEL,
) -> KnowledgeArticle:
    """Use LLM to extract structured knowledge from raw article text.

    Args:
        raw_text: Extracted article text from trafilatura.
        url: Source URL.
        source_name: Name of the source (e.g. "HubSpot").
        model: LLM model for extraction.

    Returns:
        Structured KnowledgeArticle.
    """
    # Truncate very long articles to fit context
    truncated = raw_text[:12000] if len(raw_text) > 12000 else raw_text

    user_msg = f"Extract structured marketing knowledge from this article:\n\n{truncated}"
    response_text = await complete(
        messages=[{"role": "user", "content": user_msg}],
        model=model,
        system=EXTRACTION_SYSTEM_PROMPT,
        max_tokens=2048,
        temperature=0.0,
    )

    try:
        extracted: dict[str, Any] = json.loads(response_text)
    except json.JSONDecodeError:
        logger.warning(f"LLM returned invalid JSON for {url}, using raw text only")
        extracted = {}

    return KnowledgeArticle(
        url=url,
        source_name=source_name,
        title=extracted.get("title", ""),
        topic=extracted.get("topic", ""),
        channel=extracted.get("channel", []),
        strategies=extracted.get("strategies", []),
        frameworks=extracted.get("frameworks", []),
        metrics=extracted.get("metrics", {}),
        key_insights=extracted.get("key_insights", []),
        raw_text=raw_text,
        extraction_model=model,
    )


async def process_url_list(
    urls: list[dict[str, str]],
    max_concurrent: int = 5,
) -> list[KnowledgeArticle]:
    """Process a list of URLs into structured knowledge articles.

    Args:
        urls: List of {"url": "...", "source_name": "..."} dicts.
        max_concurrent: Maximum concurrent extractions.

    Returns:
        List of extracted KnowledgeArticle objects.
    """
    import asyncio

    semaphore = asyncio.Semaphore(max_concurrent)
    articles: list[KnowledgeArticle] = []

    async def _process_one(entry: dict[str, str]) -> KnowledgeArticle | None:
        async with semaphore:
            url = entry["url"]
            source = entry.get("source_name", "")
            logger.info(f"Processing: {url}")

            raw_text = await extract_from_url(url)
            if not raw_text:
                return None

            return await structure_content(raw_text, url, source)

    tasks = [_process_one(entry) for entry in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, KnowledgeArticle):
            articles.append(result)
        elif isinstance(result, Exception):
            logger.warning(f"Knowledge extraction failed: {result}")

    logger.info(f"Successfully extracted {len(articles)}/{len(urls)} articles")
    return articles
