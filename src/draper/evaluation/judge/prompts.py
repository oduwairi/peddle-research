"""Prompt templates for the pairwise LLM-as-judge.

Five dimensions per the IMPLEMENTATION_PLAN:
  - strategic_relevance: does the copy address the brief's product/audience?
  - creativity:          is the angle / hook fresh, not generic?
  - actionability:       is there a clear, motivating CTA?
  - channel_appropriateness: tone, length, format match the platform?
  - predicted_performance:   would this likely beat a typical ad in this category?

Scores are integers in [-2, +2] FROM RESPONSE A's PERSPECTIVE:
  +2 = A is much better, +1 = A is somewhat better, 0 = tie,
  -1 = B somewhat better, -2 = B much better.
The judge declares an overall winner ∈ {A, B, tie} and a one-sentence rationale.
"""

from __future__ import annotations

PAIRWISE_SYSTEM = """\
You are an expert advertising critic evaluating two candidate ad-copy responses
to the same brief.

You will:
1. Read the brief, including the platform and the product/business context.
2. Read Response A and Response B in full.
3. Score five dimensions independently. Each score is an integer in [-2, +2]
   FROM RESPONSE A's PERSPECTIVE:
   +2 = A is much better, +1 = A is somewhat better, 0 = tie,
   -1 = B is somewhat better, -2 = B is much better.
4. Pick an overall winner: "A", "B", or "tie".
5. Justify the winner in one tight sentence.

Be terse. Do NOT reward verbosity, generic enthusiasm, or padding. Reward
concrete buyer-relevant claims, on-platform craft, and a real call to action."""


PAIRWISE_USER_TEMPLATE = """\
# Brief
- Platform: {platform}
- Vertical: {vertical}
- User prompt: {user_prompt}

# Response A
{response_a}

# Response B
{response_b}

Score the five dimensions and declare a winner. Output strict JSON matching the schema."""


def build_pairwise_user_prompt(
    *,
    platform: str,
    vertical: str,
    user_prompt: str,
    response_a: str,
    response_b: str,
) -> str:
    return PAIRWISE_USER_TEMPLATE.format(
        platform=platform,
        vertical=vertical,
        user_prompt=user_prompt,
        response_a=response_a or "(empty response)",
        response_b=response_b or "(empty response)",
    )


# JSON schema given to providers that support structured output (OpenAI's
# response_format=json_schema; Gemini's response_schema). The integer
# bounds aren't enforced in the schema (JSON Schema "minimum"/"maximum"
# are not always enforced) — we clip in code.
PAIRWISE_JSON_SCHEMA: dict[str, object] = {
    "name": "pairwise_judgment",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "strategic_relevance": {"type": "integer"},
            "creativity": {"type": "integer"},
            "actionability": {"type": "integer"},
            "channel_appropriateness": {"type": "integer"},
            "predicted_performance": {"type": "integer"},
            "overall_winner": {"type": "string", "enum": ["A", "B", "tie"]},
            "rationale": {"type": "string"},
        },
        "required": [
            "strategic_relevance",
            "creativity",
            "actionability",
            "channel_appropriateness",
            "predicted_performance",
            "overall_winner",
            "rationale",
        ],
    },
    "strict": True,
}
