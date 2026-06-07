"""Tests for openai-cost-py."""

from __future__ import annotations

import pytest

import openai_cost
from openai_cost import (
    DEFAULT_PRICING_TABLE,
    Pricing,
    Usage,
    cost,
    default_pricing,
    known_models,
    normalize_model_id,
    usage,
)

# ---------------------------------------------------------------------------
# normalize_model_id
# ---------------------------------------------------------------------------


def test_normalize_known_alias():
    assert normalize_model_id("gpt-4o-latest") == "gpt-4o"


def test_normalize_chatgpt_alias():
    assert normalize_model_id("chatgpt-4o-latest") == "gpt-4o"


def test_normalize_canonical_passthrough():
    assert normalize_model_id("gpt-4o-mini") == "gpt-4o-mini"


def test_normalize_unknown_passthrough():
    assert normalize_model_id("my-custom-model-v1") == "my-custom-model-v1"


def test_normalize_strips_whitespace():
    assert normalize_model_id("  gpt-4o-latest  ") == "gpt-4o"


def test_normalize_o_series_aliases():
    assert normalize_model_id("o1-latest") == "o1"
    assert normalize_model_id("o3-mini-latest") == "o3-mini"
    assert normalize_model_id("o4-mini-latest") == "o4-mini"


def test_normalize_gpt4_turbo_aliases():
    assert normalize_model_id("gpt-4-turbo-preview") == "gpt-4-turbo"
    assert normalize_model_id("gpt-4-0125-preview") == "gpt-4-turbo"


def test_normalize_gpt41_aliases():
    assert normalize_model_id("gpt-4.1-latest") == "gpt-4.1"
    assert normalize_model_id("gpt-4.1-mini-latest") == "gpt-4.1-mini"


# ---------------------------------------------------------------------------
# default_pricing
# ---------------------------------------------------------------------------


def test_default_pricing_known_model():
    p = default_pricing("gpt-4o")
    assert p is not None
    assert isinstance(p, Pricing)


def test_default_pricing_via_alias():
    p = default_pricing("gpt-4o-latest")
    assert p is not None
    assert p == default_pricing("gpt-4o")


def test_default_pricing_unknown_returns_none():
    assert default_pricing("not-a-real-model-xyz") is None


def test_default_pricing_embedding():
    p = default_pricing("text-embedding-3-small")
    assert p is not None
    assert p.completion_per_m == 0.0


# ---------------------------------------------------------------------------
# known_models
# ---------------------------------------------------------------------------


def test_known_models_returns_list():
    models = known_models()
    assert isinstance(models, list)
    assert len(models) > 0


def test_known_models_sorted():
    models = known_models()
    assert models == sorted(models)


def test_known_models_includes_gpt4o():
    assert "gpt-4o" in known_models()


def test_known_models_includes_gpt35():
    assert "gpt-3.5-turbo" in known_models()


# ---------------------------------------------------------------------------
# cost — basic
# ---------------------------------------------------------------------------


def test_cost_gpt4o_no_cache():
    # gpt-4o: $2.50/M prompt, $10.00/M completion
    # 1000 prompt + 500 completion
    # = 1000 * 2.50/1e6 + 500 * 10.00/1e6
    # = 0.0025 + 0.005 = 0.0075
    usd = cost(model="gpt-4o", prompt_tokens=1000, completion_tokens=500)
    assert usd == pytest.approx(0.0075)


def test_cost_gpt4o_mini_no_cache():
    # gpt-4o-mini: $0.15/M prompt, $0.60/M completion
    # 1000 prompt + 200 completion
    # = 1000 * 0.15/1e6 + 200 * 0.60/1e6
    # = 0.00015 + 0.00012 = 0.00027
    usd = cost(model="gpt-4o-mini", prompt_tokens=1000, completion_tokens=200)
    assert usd == pytest.approx(0.00027)


def test_cost_o1():
    # o1: $15.00/M prompt, $60.00/M completion
    # 500 + 100
    # = 500 * 15/1e6 + 100 * 60/1e6 = 0.0075 + 0.006 = 0.0135
    usd = cost(model="o1", prompt_tokens=500, completion_tokens=100)
    assert usd == pytest.approx(0.0135)


def test_cost_gpt35_turbo():
    # $0.50/M prompt, $1.50/M completion
    # 1000 + 1000 = 1000 * 0.5/1e6 + 1000 * 1.5/1e6 = 0.0005 + 0.0015 = 0.002
    usd = cost(model="gpt-3.5-turbo", prompt_tokens=1000, completion_tokens=1000)
    assert usd == pytest.approx(0.002)


def test_cost_zero_tokens():
    usd = cost(model="gpt-4o", prompt_tokens=0, completion_tokens=0)
    assert usd == pytest.approx(0.0)


def test_cost_embedding_zero_completion():
    # text-embedding-3-small: $0.02/M, no completion
    usd = cost(
        model="text-embedding-3-small",
        prompt_tokens=1_000_000,
        completion_tokens=0,
    )
    assert usd == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# cost — cached tokens
# ---------------------------------------------------------------------------


def test_cost_with_all_cached():
    # gpt-4o: cache rate $1.25/M (vs $2.50/M full)
    # 1000 prompt all cached, 500 completion
    # = 1000 * 1.25/1e6 + 500 * 10.00/1e6 = 0.00125 + 0.005 = 0.00625
    usd = cost(
        model="gpt-4o",
        prompt_tokens=1000,
        completion_tokens=500,
        cached_tokens=1000,
    )
    assert usd == pytest.approx(0.00625)


def test_cost_with_partial_cache():
    # gpt-4o: 800 cached, 200 uncached, 500 completion
    # = 200 * 2.50/1e6 + 800 * 1.25/1e6 + 500 * 10.00/1e6
    # = 0.0005 + 0.001 + 0.005 = 0.0065
    usd = cost(
        model="gpt-4o",
        prompt_tokens=1000,
        completion_tokens=500,
        cached_tokens=800,
    )
    assert usd == pytest.approx(0.0065)


def test_cost_cached_exceeds_prompt_clipped_to_zero():
    # cached_tokens > prompt_tokens → uncached = 0
    usd_normal = cost(model="gpt-4o", prompt_tokens=100, completion_tokens=0)
    usd_over = cost(
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=0,
        cached_tokens=200,  # more than prompt_tokens
    )
    # Uncached = max(0, 100 - 200) = 0 → all tokens at cache rate
    assert usd_over < usd_normal


def test_cost_no_cache_support_uses_full_rate():
    # gpt-4-32k has no cached_prompt_per_m → falls back to full prompt rate
    # 1000 prompt "cached" but no discount
    usd_cached = cost(
        model="gpt-4-32k",
        prompt_tokens=1000,
        completion_tokens=0,
        cached_tokens=1000,
    )
    usd_full = cost(model="gpt-4-32k", prompt_tokens=1000, completion_tokens=0)
    assert usd_cached == pytest.approx(usd_full)


# ---------------------------------------------------------------------------
# cost — batch
# ---------------------------------------------------------------------------


def test_cost_batch_50pct_discount():
    normal = cost(model="gpt-4o", prompt_tokens=1000, completion_tokens=500)
    batched = cost(model="gpt-4o", prompt_tokens=1000, completion_tokens=500, batch=True)
    assert batched == pytest.approx(normal * 0.5)


def test_cost_batch_not_available_ignored():
    # gpt-4-32k has no batch_discount in the table
    normal = cost(model="gpt-4-32k", prompt_tokens=1000, completion_tokens=100)
    batched = cost(model="gpt-4-32k", prompt_tokens=1000, completion_tokens=100, batch=True)
    assert batched == pytest.approx(normal)


def test_cost_batch_mini():
    normal = cost(model="gpt-4o-mini", prompt_tokens=2000, completion_tokens=1000)
    batched = cost(
        model="gpt-4o-mini",
        prompt_tokens=2000,
        completion_tokens=1000,
        batch=True,
    )
    assert batched == pytest.approx(normal * 0.5)


# ---------------------------------------------------------------------------
# cost — alias resolution
# ---------------------------------------------------------------------------


def test_cost_via_alias():
    usd_alias = cost(model="gpt-4o-latest", prompt_tokens=1000, completion_tokens=100)
    usd_canonical = cost(model="gpt-4o", prompt_tokens=1000, completion_tokens=100)
    assert usd_alias == pytest.approx(usd_canonical)


def test_cost_unknown_model_raises():
    with pytest.raises(KeyError, match="not found in pricing table"):
        cost(model="not-a-real-model-xyz", prompt_tokens=100, completion_tokens=50)


# ---------------------------------------------------------------------------
# cost — custom pricing table
# ---------------------------------------------------------------------------


def test_cost_custom_pricing_table():
    custom = {"my-model": Pricing(prompt_per_m=1.00, completion_per_m=2.00)}
    usd = cost(
        model="my-model",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        pricing_table=custom,
    )
    assert usd == pytest.approx(3.00)


def test_cost_custom_table_ignores_builtin():
    # Override gpt-4o with cheap pricing
    custom = {"gpt-4o": Pricing(prompt_per_m=0.01, completion_per_m=0.01)}
    usd = cost(model="gpt-4o", prompt_tokens=1_000_000, completion_tokens=0, pricing_table=custom)
    assert usd == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# usage()
# ---------------------------------------------------------------------------


def test_usage_returns_usage_namedtuple():
    u = usage(model="gpt-4o-mini", prompt_tokens=500, completion_tokens=100)
    assert isinstance(u, Usage)


def test_usage_canonical_model_stored():
    u = usage(model="gpt-4o-latest", prompt_tokens=100, completion_tokens=10)
    assert u.model == "gpt-4o"


def test_usage_usd_matches_cost():
    u = usage(
        model="gpt-4o",
        prompt_tokens=1000,
        completion_tokens=500,
        cached_tokens=800,
        batch=False,
    )
    expected = cost(
        model="gpt-4o",
        prompt_tokens=1000,
        completion_tokens=500,
        cached_tokens=800,
    )
    assert u.usd == pytest.approx(expected)


def test_usage_fields_preserved():
    u = usage(
        model="o1",
        prompt_tokens=300,
        completion_tokens=100,
        cached_tokens=150,
        batch=True,
    )
    assert u.prompt_tokens == 300
    assert u.completion_tokens == 100
    assert u.cached_tokens == 150
    assert u.batch is True


# ---------------------------------------------------------------------------
# Pricing dataclass
# ---------------------------------------------------------------------------


def test_pricing_frozen():
    p = Pricing(prompt_per_m=1.0, completion_per_m=2.0)
    with pytest.raises(Exception):
        p.prompt_per_m = 99.0  # type: ignore[misc]


def test_pricing_no_cache_by_default():
    p = Pricing(prompt_per_m=1.0, completion_per_m=2.0)
    assert p.cached_prompt_per_m is None
    assert p.batch_discount is None


# ---------------------------------------------------------------------------
# Default price table sanity
# ---------------------------------------------------------------------------


def test_all_prices_positive():
    for model_id, p in DEFAULT_PRICING_TABLE.items():
        assert p.prompt_per_m >= 0, f"{model_id}: prompt price must be >= 0"
        assert p.completion_per_m >= 0, f"{model_id}: completion price must be >= 0"


def test_gpt4o_has_cache_pricing():
    p = DEFAULT_PRICING_TABLE["gpt-4o"]
    assert p.cached_prompt_per_m is not None
    assert p.cached_prompt_per_m < p.prompt_per_m


def test_gpt4o_has_batch_pricing():
    p = DEFAULT_PRICING_TABLE["gpt-4o"]
    assert p.batch_discount == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_usage_is_public_export():
    # usage() is documented as a top-level API function and must be exported.
    assert "usage" in openai_cost.__all__
    assert openai_cost.usage is usage


def test_public_all_is_importable():
    # Everything listed in __all__ must be a real attribute of the package.
    for name in openai_cost.__all__:
        assert hasattr(openai_cost, name), f"{name} listed in __all__ but missing"


# ---------------------------------------------------------------------------
# cost — batch + cache interaction
# ---------------------------------------------------------------------------


def test_cost_batch_and_cache_combined():
    # gpt-4o: 800 cached + 200 uncached prompt, 500 completion, then 50% batch off.
    # uncached = 200 * 2.50/1e6 = 0.0005
    # cached   = 800 * 1.25/1e6 = 0.001
    # output   = 500 * 10.00/1e6 = 0.005
    # subtotal = 0.0065 → batch *0.5 → 0.00325
    usd = cost(
        model="gpt-4o",
        prompt_tokens=1000,
        completion_tokens=500,
        cached_tokens=800,
        batch=True,
    )
    assert usd == pytest.approx(0.00325)
