"""Cost calculator for OpenAI API calls.

Price table last updated: 2026-05-24.
Source: https://openai.com/api/pricing/

All prices are per 1,000,000 tokens (per-million).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Pricing model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pricing:
    """Per-million-token prices for a single model.

    Args:
        prompt_per_m: USD per 1M prompt (input) tokens.
        completion_per_m: USD per 1M completion (output) tokens.
        cached_prompt_per_m: USD per 1M cached prompt tokens.
            ``None`` means no cache discount (full prompt price applies).
        batch_discount: Fraction applied to both prompt and completion when
            using the Batch API (0.5 = 50% off).  ``None`` means no batch
            pricing is available.
    """

    prompt_per_m: float
    completion_per_m: float
    cached_prompt_per_m: float | None = None
    batch_discount: float | None = None   # e.g. 0.5 for 50 % off


class Usage(NamedTuple):
    """Token usage for one API call.

    Attributes:
        model: The model id used (after normalization).
        prompt_tokens: Total prompt tokens (before any cache adjustment).
        completion_tokens: Completion (output) tokens.
        cached_tokens: How many of the prompt tokens were served from cache.
            Pass ``0`` if there were no cache hits.
        batch: Whether the Batch API was used (50 % discount).
        usd: Computed USD cost.
    """

    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    batch: bool
    usd: float


# ---------------------------------------------------------------------------
# Built-in price table (2026-05-24)
# ---------------------------------------------------------------------------

#: Built-in price table.  Keys are canonical model ids.
DEFAULT_PRICING_TABLE: dict[str, Pricing] = {
    # ---- GPT-4o family ----
    "gpt-4o": Pricing(
        prompt_per_m=2.50,
        completion_per_m=10.00,
        cached_prompt_per_m=1.25,
        batch_discount=0.5,
    ),
    "gpt-4o-2024-11-20": Pricing(
        prompt_per_m=2.50,
        completion_per_m=10.00,
        cached_prompt_per_m=1.25,
        batch_discount=0.5,
    ),
    "gpt-4o-2024-08-06": Pricing(
        prompt_per_m=2.50,
        completion_per_m=10.00,
        cached_prompt_per_m=1.25,
        batch_discount=0.5,
    ),
    "gpt-4o-audio-preview": Pricing(
        prompt_per_m=2.50,
        completion_per_m=10.00,
    ),
    # ---- GPT-4o-mini family ----
    "gpt-4o-mini": Pricing(
        prompt_per_m=0.15,
        completion_per_m=0.60,
        cached_prompt_per_m=0.075,
        batch_discount=0.5,
    ),
    "gpt-4o-mini-2024-07-18": Pricing(
        prompt_per_m=0.15,
        completion_per_m=0.60,
        cached_prompt_per_m=0.075,
        batch_discount=0.5,
    ),
    # ---- o1 reasoning family ----
    "o1": Pricing(
        prompt_per_m=15.00,
        completion_per_m=60.00,
        cached_prompt_per_m=7.50,
        batch_discount=0.5,
    ),
    "o1-2024-12-17": Pricing(
        prompt_per_m=15.00,
        completion_per_m=60.00,
        cached_prompt_per_m=7.50,
        batch_discount=0.5,
    ),
    "o1-mini": Pricing(
        prompt_per_m=1.10,
        completion_per_m=4.40,
        cached_prompt_per_m=0.55,
        batch_discount=0.5,
    ),
    "o1-mini-2024-09-12": Pricing(
        prompt_per_m=1.10,
        completion_per_m=4.40,
        cached_prompt_per_m=0.55,
        batch_discount=0.5,
    ),
    "o1-preview": Pricing(
        prompt_per_m=15.00,
        completion_per_m=60.00,
    ),
    # ---- o3 reasoning family ----
    "o3": Pricing(
        prompt_per_m=10.00,
        completion_per_m=40.00,
        cached_prompt_per_m=2.50,
        batch_discount=0.5,
    ),
    "o3-mini": Pricing(
        prompt_per_m=1.10,
        completion_per_m=4.40,
        cached_prompt_per_m=0.275,
        batch_discount=0.5,
    ),
    # ---- o4 reasoning family ----
    "o4-mini": Pricing(
        prompt_per_m=1.10,
        completion_per_m=4.40,
        cached_prompt_per_m=0.275,
        batch_discount=0.5,
    ),
    # ---- GPT-4 Turbo ----
    "gpt-4-turbo": Pricing(
        prompt_per_m=10.00,
        completion_per_m=30.00,
        batch_discount=0.5,
    ),
    "gpt-4-turbo-2024-04-09": Pricing(
        prompt_per_m=10.00,
        completion_per_m=30.00,
        batch_discount=0.5,
    ),
    # ---- GPT-4 legacy ----
    "gpt-4": Pricing(
        prompt_per_m=30.00,
        completion_per_m=60.00,
    ),
    "gpt-4-32k": Pricing(
        prompt_per_m=60.00,
        completion_per_m=120.00,
    ),
    # ---- GPT-3.5 Turbo ----
    "gpt-3.5-turbo": Pricing(
        prompt_per_m=0.50,
        completion_per_m=1.50,
        batch_discount=0.5,
    ),
    "gpt-3.5-turbo-0125": Pricing(
        prompt_per_m=0.50,
        completion_per_m=1.50,
        batch_discount=0.5,
    ),
    "gpt-3.5-turbo-instruct": Pricing(
        prompt_per_m=1.50,
        completion_per_m=2.00,
    ),
    # ---- Embeddings ----
    "text-embedding-3-small": Pricing(
        prompt_per_m=0.02,
        completion_per_m=0.00,
        batch_discount=0.5,
    ),
    "text-embedding-3-large": Pricing(
        prompt_per_m=0.13,
        completion_per_m=0.00,
        batch_discount=0.5,
    ),
    "text-embedding-ada-002": Pricing(
        prompt_per_m=0.10,
        completion_per_m=0.00,
        batch_discount=0.5,
    ),
    # ---- GPT-4.1 family (new 2026 series) ----
    "gpt-4.1": Pricing(
        prompt_per_m=2.00,
        completion_per_m=8.00,
        cached_prompt_per_m=0.50,
        batch_discount=0.5,
    ),
    "gpt-4.1-mini": Pricing(
        prompt_per_m=0.40,
        completion_per_m=1.60,
        cached_prompt_per_m=0.10,
        batch_discount=0.5,
    ),
    "gpt-4.1-nano": Pricing(
        prompt_per_m=0.10,
        completion_per_m=0.40,
        cached_prompt_per_m=0.025,
        batch_discount=0.5,
    ),
    # ---- GPT-5 family ----
    "gpt-5": Pricing(
        prompt_per_m=5.00,
        completion_per_m=20.00,
        cached_prompt_per_m=1.25,
        batch_discount=0.5,
    ),
    "gpt-5.4": Pricing(
        prompt_per_m=5.00,
        completion_per_m=20.00,
        cached_prompt_per_m=1.25,
        batch_discount=0.5,
    ),
}

# ---------------------------------------------------------------------------
# Alias map: maps known aliases to canonical ids
# ---------------------------------------------------------------------------

_ALIASES: dict[str, str] = {
    # GPT-4o
    "gpt-4o-latest": "gpt-4o",
    "chatgpt-4o-latest": "gpt-4o",
    # GPT-4o-mini
    "gpt-4o-mini-latest": "gpt-4o-mini",
    # o1
    "o1-latest": "o1",
    # o3
    "o3-latest": "o3",
    "o3-mini-latest": "o3-mini",
    # o4
    "o4-mini-latest": "o4-mini",
    # GPT-4 Turbo
    "gpt-4-turbo-preview": "gpt-4-turbo",
    "gpt-4-0125-preview": "gpt-4-turbo",
    "gpt-4-1106-preview": "gpt-4-turbo",
    # GPT-3.5
    "gpt-3.5-turbo-16k": "gpt-3.5-turbo",
    "gpt-3.5-turbo-1106": "gpt-3.5-turbo",
    # GPT-4.1
    "gpt-4.1-latest": "gpt-4.1",
    "gpt-4.1-mini-latest": "gpt-4.1-mini",
    # GPT-5
    "gpt-5-latest": "gpt-5",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def normalize_model_id(model: str) -> str:
    """Return the canonical model id for *model*.

    Resolves common aliases and strips irrelevant suffixes. Returns the input
    unchanged if it is already canonical or unknown.

    Args:
        model: A raw model id string (e.g. ``"gpt-4o-latest"``).

    Returns:
        Canonical id (e.g. ``"gpt-4o"``).

    Example::

        >>> normalize_model_id("gpt-4o-latest")
        'gpt-4o'
        >>> normalize_model_id("gpt-4.1-mini")
        'gpt-4.1-mini'
    """
    m = model.strip()
    # Try alias lookup first
    if m in _ALIASES:
        return _ALIASES[m]
    # Already canonical or unknown → return as-is
    return m


def default_pricing(model: str) -> Pricing | None:
    """Return the built-in :class:`Pricing` for *model*, or ``None``.

    Performs :func:`normalize_model_id` before lookup.

    Args:
        model: Model id (alias or canonical).

    Returns:
        :class:`Pricing` if the model is in the built-in table, else ``None``.
    """
    return DEFAULT_PRICING_TABLE.get(normalize_model_id(model))


def known_models() -> list[str]:
    """Return all canonical model ids in the built-in price table (sorted)."""
    return sorted(DEFAULT_PRICING_TABLE)


def cost(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    batch: bool = False,
    pricing_table: dict[str, Pricing] | None = None,
) -> float:
    """Calculate the USD cost of an OpenAI API call.

    Args:
        model: Model id (alias or canonical).
        prompt_tokens: Total prompt (input) tokens.
        completion_tokens: Completion (output) tokens.
        cached_tokens: Subset of *prompt_tokens* served from cache (cheaper).
            Pass ``0`` (default) if cache hits are unknown or irrelevant.
        batch: ``True`` if the Batch API was used (50 % discount).
        pricing_table: Optional override for the built-in price table.
            Keys must be canonical model ids.

    Returns:
        Cost in USD as a ``float``.

    Raises:
        KeyError: If the model is not found in the price table.

    Example::

        >>> from openai_cost import cost
        >>> round(cost(model="gpt-4o-mini", prompt_tokens=1000, completion_tokens=200), 8)
        0.00027
    """
    table = pricing_table or DEFAULT_PRICING_TABLE
    canonical = normalize_model_id(model)
    try:
        p = table[canonical]
    except KeyError:
        raise KeyError(
            f"Model {model!r} (canonical: {canonical!r}) not found in pricing table. "
            f"Pass a custom pricing_table or use one of: {known_models()}"
        ) from None

    # Cap cached_tokens to prompt_tokens; compute non-cached portion
    effective_cached = min(cached_tokens, prompt_tokens)
    uncached_tokens = prompt_tokens - effective_cached

    # Choose cached-input rate
    cache_rate = p.cached_prompt_per_m if p.cached_prompt_per_m is not None else p.prompt_per_m

    prompt_cost = (uncached_tokens * p.prompt_per_m + effective_cached * cache_rate) / 1_000_000
    completion_cost = completion_tokens * p.completion_per_m / 1_000_000

    total = prompt_cost + completion_cost

    if batch and p.batch_discount is not None:
        total *= 1 - p.batch_discount

    return total


def usage(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    batch: bool = False,
    pricing_table: dict[str, Pricing] | None = None,
) -> Usage:
    """Like :func:`cost` but returns a :class:`Usage` named-tuple with full detail.

    Args:
        model: Model id (alias or canonical).
        prompt_tokens: Total prompt tokens.
        completion_tokens: Completion tokens.
        cached_tokens: Cached subset of prompt tokens.
        batch: Whether the Batch API was used.
        pricing_table: Optional override for the built-in price table.

    Returns:
        :class:`Usage` with all fields and computed ``usd``.
    """
    usd = cost(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        batch=batch,
        pricing_table=pricing_table,
    )
    return Usage(
        model=normalize_model_id(model),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        batch=batch,
        usd=usd,
    )
