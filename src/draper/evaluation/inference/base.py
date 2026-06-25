"""Inference runner protocol — one per config (A / B / C / A_pipe / D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..schemas import Brief, Inference, UrlScenario


@runtime_checkable
class InferenceRunner(Protocol):
    """Stateless runner that produces an Inference for a Brief or UrlScenario.

    Implementations are async because every backing call is network-bound:
    direct OpenAI, vLLM HTTP, or the frontend pipeline POST.
    """

    config_name: str
    arm: str  # "arm1" or "arm2"

    async def run_brief(self, brief: Brief) -> Inference: ...

    async def run_scenario(self, scenario: UrlScenario) -> Inference: ...
