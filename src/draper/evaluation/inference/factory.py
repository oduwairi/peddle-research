"""Build the right InferenceRunner for a config name from configs/eval.yaml."""

from __future__ import annotations

from ..config import ConfigDef
from .base import InferenceRunner
from .frontend_runner import FrontendRunner
from .openai_runner import OpenAIRunner
from .vllm_runner import VLLMRunner


def build_runner(config_name: str, definition: ConfigDef) -> InferenceRunner:
    """Return the runner instance matching ``definition.runner``."""
    if definition.runner == "openai":
        if not definition.model:
            raise ValueError(f"Config {config_name} (openai) requires a model name.")
        return OpenAIRunner(
            config_name=config_name,
            model=definition.model,
            max_tokens=definition.max_tokens,
            temperature=definition.temperature,
        )
    if definition.runner == "vllm":
        if not definition.model:
            raise ValueError(f"Config {config_name} (vllm) requires a model name.")
        return VLLMRunner(
            config_name=config_name,
            model=definition.model,
            base_url_env=definition.base_url_env or "VLLM_BASE_URL",
            api_key_env=definition.api_key_env or "VLLM_API_KEY",
            max_tokens=definition.max_tokens,
            temperature=definition.temperature,
        )
    if definition.runner == "frontend":
        if not definition.base_url_env:
            raise ValueError(f"Config {config_name} (frontend) requires base_url_env.")
        return FrontendRunner(
            config_name=config_name,
            base_url_env=definition.base_url_env,
            token_env=definition.token_env or "EVAL_SERVICE_TOKEN",
            timeout_s=definition.timeout_s,
        )
    raise ValueError(f"Unknown runner type: {definition.runner}")
