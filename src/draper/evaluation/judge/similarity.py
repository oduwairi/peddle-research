"""Similarity-to-gold diagnostics — emitted as columns, never as scores.

These metrics compare a model's generated copy to the real winning ad
(``Brief.reference_assistant``). They're **not** used to pick winners or
gate any judgment — text similarity rewards mimicry, which is exactly the
opposite of what fine-tuning should produce.

But the columns are useful as a diagnostic alongside win-rate-vs-GOLD:
  - High similarity + high win-rate vs GOLD = the model is regurgitating
    training-distribution copy. The headline number is misleading.
  - Low similarity + high win-rate vs GOLD = the model is producing genuinely
    novel copy that the judge prefers over the real ad. That's the goal.
  - High similarity + low win-rate = poor mimicry. Rare; usually means the
    model copied the wrong parts of the gold.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenization. Unicode-aware; strips punctuation."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t]


def rouge_l_f1(reference: str, candidate: str) -> float:
    """ROUGE-L F1 score: longest-common-subsequence overlap.

    Returns 0.0 for empty inputs. Uses a standard O(m*n) DP. Token-level,
    case-insensitive, punctuation-insensitive.
    """
    ref = _tokenize(reference)
    cand = _tokenize(candidate)
    if not ref or not cand:
        return 0.0

    m, n = len(ref), len(cand)
    # Two-row DP to keep memory bounded — eval inputs can run ~500 tokens.
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == cand[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    lcs = prev[n]

    if lcs == 0:
        return 0.0
    precision = lcs / n
    recall = lcs / m
    return 2.0 * precision * recall / (precision + recall)


@lru_cache(maxsize=1)
def _embedder() -> Any | None:
    """Lazy-load sentence-transformers; return None if unavailable.

    Cached so repeated similarity calls don't reload the model.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not installed — cosine_to_gold will return None")
        return None
    try:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as exc:  # pragma: no cover — model download / disk failure
        logger.warning("Failed to load embedder: %s", exc)
        return None


def cosine_similarity(reference: str, candidate: str) -> float | None:
    """Cosine similarity on MiniLM embeddings. Returns None if model unavailable.

    Soft-fails on any embedder error so the eval pipeline never crashes
    on a missing optional dep — callers should treat None as "not measured".
    """
    if not reference or not candidate:
        return 0.0
    model = _embedder()
    if model is None:
        return None
    try:
        embs = model.encode([reference, candidate], normalize_embeddings=True)
    except Exception as exc:  # pragma: no cover
        logger.warning("Embedder.encode failed: %s", exc)
        return None
    # Normalized embeddings → dot product is cosine similarity.
    return float(embs[0] @ embs[1])


def similarity_to_gold(model_text: str, gold_text: str) -> dict[str, float | None]:
    """Compute both rouge_l_f1 and cosine_to_gold. Cosine may be None."""
    return {
        "rouge_l_f1": rouge_l_f1(gold_text, model_text),
        "cosine_to_gold": cosine_similarity(gold_text, model_text),
    }
