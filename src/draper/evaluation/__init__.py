"""Evaluation module — four independent arms plus methodology validation.

**Four eval arms:**
  - ``judge/`` — pairwise LLM-as-judge (tournament + reference GOLD) with
    position-swap reconciliation.
  - ``learned_scorer`` — absolute scorer arm via trained DeBERTa-v3 regressor
    (composite + 3 per-head scores).
  - ``mauve_scorer`` — distribution-matching arm (population-level text
    feature overlap vs real high-tier ads).
  - ``proxy_validation`` — methodology validation against external A/B-test
    ground truth (Upworthy, etc).

**Sub-packages:**
  - ``inference/`` — per-config runners (OpenAI-compat, vLLM, frontend pipeline).
  - ``judge/``     — pairwise LLM-as-judge infrastructure and aggregation math.

**Top-level modules:**
  - ``briefs``          — load held-out test briefs and URL scenarios.
  - ``config``          — ``EvalConfig`` Pydantic model for ``configs/eval.yaml``.
  - ``driver``          — orchestration helpers (inference + judge runs).
  - ``gold``            — GOLD sentinel config (real winning ad at judge time).
  - ``learned_scorer``  — absolute-scorer arm: predictor inference +
    bootstrap aggregation.
  - ``mauve_scorer``    — distribution-matching arm: featurization +
    per-platform slices.
  - ``mauve_reference`` — reference corpus loader for MAUVE arm
    (cached, contamination-filtered).
  - ``proxy_validation`` — statistical primitives for methodology validation.
  - ``schemas``         — shared ``Brief``, ``Inference``, ``Judgment``, etc.
  - ``*_loader``        — dataset loaders (AdFlex, IRA, Meta EU, Upworthy, …).
"""
