"""Strict copy extraction for judge inputs.

Different runners (single-shot OpenAI, vLLM-base, vLLM-FT, the frontend
agent loop) leak different non-copy artifacts into ``Inference.assistant_text``:

  - The Qwen FT model emits an empty reasoning block: ``<think>...</think>``.
  - GPT-4o single-shot prefixes copy with markdown chrome like
    ``**Ad Copy:**\\n\\n…`` or ``**Headline:** …``.
  - All single-shot configs sometimes lead with conversational preambles
    like ``"Here's a strong ad execution:"`` or ``"Here is the ad:"``.
  - The frontend pipeline path already strips its ``meta`` block before
    flattening copy fields, but its emitted text may still wrap fields in
    label-style headers.

Without normalization the judge scores copy+chrome for some configs and
clean copy for others — an unmeasured asymmetry. ``clean_copy`` runs at
judge time only; ``Inference.assistant_text`` on disk stays raw for forensics.
"""

from __future__ import annotations

import re

# <think>...</think> blocks (with or without inner content/whitespace).
# ``re.DOTALL`` so dotall covers newlines inside the block.
_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)

# A leading bold/emphasized header followed by optional whitespace, e.g.:
#   **Ad Copy:**\n\n  → strip
#   **Ad Copy**\n      → strip
#   **Headline:** Foo  → keep "Foo" by stripping just the header
# We only strip when the header is ON ITS OWN LINE (followed by newline
# or end-of-string). Inline labels like ``**Headline:** Foo`` stay because
# the label is part of the platform-shaped output for some Google RSA copy.
_LEADING_HEADER = re.compile(
    r"""
    ^\s*                                # leading whitespace
    \*{1,3}                              # 1-3 asterisks
    \s*
    (?:Ad\s*Copy|Copy|Final\s*Copy|Output|Response|Answer|Result)
    \s*[:.\-]?\s*                        # optional colon/dash/period
    \*{1,3}                              # closing asterisks
    \s*\n+                               # MUST be followed by a newline
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Conversational preambles. Match the FIRST line if it's clearly not copy.
# We cap line length to avoid eating an entire opening sentence of an ad
# that happens to start with "Here's …" as a hook.
_PREAMBLE_PATTERNS = (
    re.compile(
        r"""
        ^\s*
        (?:
            here(?:'s|\s+is)\s+
            (?:a|an|the|my|your|some|one|my\s+take|the\s+ad|the\s+copy|the\s+execution)
            [^\n]{0,80}                  # rest of preamble (bounded)
            [:.]                         # ends with colon or period
            \s*\n+
            |
            sure[!,]?\s+(?:here|let|i)[^\n]{0,80}[:.\n]\s*\n+
            |
            (?:below|above)\s+is[^\n]{0,80}[:.\n]\s*\n+
            |
            i(?:'ll|\s+will|'ve|\s+have)?\s+
            (?:write|craft|draft|put\s+together|come\s+up\s+with|create)
            [^\n]{0,80}[:.\n]\s*\n+
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
)

# Trailing meta-commentary block ("This works because…", "Why this works:")
# at the END of the response. Strip from the marker to end-of-string.
_TRAILING_META = re.compile(
    r"""
    \n\s*
    (?:
        \*{0,3}\s*(?:why\s+this\s+works|rationale|analysis|notes?|why|reasoning|explanation)\s*[:.]?\s*\*{0,3}
        |
        \*{0,3}\s*(?:strategy|approach|angle|audience)\s*[:.]\s*\*{0,3}
    )
    .*\Z
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


def clean_copy(raw: str) -> str:
    """Normalize a raw model response into copy-only text for judging.

    Idempotent: ``clean_copy(clean_copy(x)) == clean_copy(x)``.
    Preserves the rest of the response verbatim — does not collapse
    internal whitespace, de-emoji, or otherwise touch the body.
    """
    if not raw:
        return raw

    text = raw

    # 1. Strip <think>...</think> blocks anywhere.
    text = _THINK_BLOCK.sub("", text)

    # 2. Strip a leading section header on its own line, then re-check
    #    for preambles (they may sit after the header).
    for _ in range(2):
        before = text
        text = _LEADING_HEADER.sub("", text.lstrip())
        for pat in _PREAMBLE_PATTERNS:
            text = pat.sub("", text.lstrip())
        if text == before:
            break

    # 3. Strip trailing meta-commentary blocks.
    text = _TRAILING_META.sub("", text)

    return text.strip()
