"""openai-cost: cache-aware cost calculator for OpenAI API calls.

Calculates exact USD cost for OpenAI API calls from token counts, including:

* Prompt/completion token pricing for all major models
* 50 % cached-input discount (GPT-4o, GPT-4o-mini, o1, o3, o4)
* 50 % batch discount (Batch API)
* Model-id normalization (aliases → canonical id)
* Built-in 2026-05 price table; BYO via ``Pricing``

Zero external dependencies.  Mirrors the ``claude-cost-py`` API.

Quick start::

    from openai_cost import cost, default_pricing, normalize_model_id

    usd = cost(
        model="gpt-4o",
        prompt_tokens=1000,
        completion_tokens=500,
    )
    # → float in USD

    # With cache hits (50% discount on cache-read tokens)
    usd = cost(
        model="gpt-4o",
        prompt_tokens=1000,
        completion_tokens=500,
        cached_tokens=800,   # 800 of the 1000 prompt tokens were cache hits
    )

    # Batch API (50% discount on everything)
    usd = cost(
        model="gpt-4o-mini",
        prompt_tokens=2000,
        completion_tokens=1000,
        batch=True,
    )
"""

from .core import (
    DEFAULT_PRICING_TABLE,
    Pricing,
    Usage,
    cost,
    default_pricing,
    known_models,
    normalize_model_id,
    usage,
)

__all__ = [
    "DEFAULT_PRICING_TABLE",
    "Pricing",
    "Usage",
    "cost",
    "default_pricing",
    "known_models",
    "normalize_model_id",
    "usage",
]

__version__ = "0.1.0"
