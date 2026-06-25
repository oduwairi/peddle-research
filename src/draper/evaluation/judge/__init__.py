"""Pairwise LLM-as-judge for the eval pipeline."""

from .aggregation import bootstrap_win_rate_ci, elo_ratings, win_rates_table
from .pairwise import judge_pair, reconcile_pair
from .prompts import PAIRWISE_SYSTEM, build_pairwise_user_prompt

__all__ = [
    "PAIRWISE_SYSTEM",
    "bootstrap_win_rate_ci",
    "build_pairwise_user_prompt",
    "elo_ratings",
    "judge_pair",
    "reconcile_pair",
    "win_rates_table",
]
