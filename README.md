# openai-cost-py

Cache-aware USD cost calculator for OpenAI API calls. Zero dependencies.

Sibling to [claude-cost-py](https://github.com/MukundaKatta/claude-cost-py). Same API surface, same zero-dep design.

Part of the [@mukundakatta agent-stack](https://github.com/MukundaKatta).

## Install

```bash
pip install openai-cost
```

## Quickstart

```python
from openai_cost import cost

# Basic cost
usd = cost(model="gpt-4o", prompt_tokens=1000, completion_tokens=500)
print(f"${usd:.6f}")  # $0.007500

# With prompt cache hits (50% discount on cached tokens)
usd = cost(
    model="gpt-4o",
    prompt_tokens=1000,
    completion_tokens=500,
    cached_tokens=800,   # 800 of 1000 prompt tokens were cache hits
)

# Batch API (50% off everything)
usd = cost(
    model="gpt-4o-mini",
    prompt_tokens=2000,
    completion_tokens=1000,
    batch=True,
)
```

## Models covered (2026-05-24 prices)

| Family | Models |
| --- | --- |
| GPT-4o | `gpt-4o`, `gpt-4o-2024-11-20`, `gpt-4o-2024-08-06` |
| GPT-4o-mini | `gpt-4o-mini`, `gpt-4o-mini-2024-07-18` |
| o1 reasoning | `o1`, `o1-mini`, `o1-preview` |
| o3 reasoning | `o3`, `o3-mini` |
| o4 reasoning | `o4-mini` |
| GPT-4 Turbo | `gpt-4-turbo`, `gpt-4-turbo-2024-04-09` |
| GPT-4 | `gpt-4`, `gpt-4-32k` |
| GPT-3.5 | `gpt-3.5-turbo`, `gpt-3.5-turbo-0125`, `gpt-3.5-turbo-instruct` |
| GPT-4.1 | `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano` |
| GPT-5 | `gpt-5`, `gpt-5.4` |
| Embeddings | `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002` |

Aliases resolved automatically: `gpt-4o-latest`, `chatgpt-4o-latest`, `o1-latest`, `gpt-4-turbo-preview`, etc.

## API

### `cost(...) -> float`

```python
cost(
    model="gpt-4o",
    prompt_tokens=1000,
    completion_tokens=500,
    cached_tokens=0,        # prompt tokens served from cache
    batch=False,            # Batch API discount
    pricing_table=None,     # override built-in table
) -> float  # USD
```

Raises `KeyError` for unknown models.

### `usage(...) -> Usage`

Same args as `cost()`. Returns a `Usage` NamedTuple with all fields plus `usd`.

### `normalize_model_id(model) -> str`

Resolves aliases to canonical ids.

### `default_pricing(model) -> Pricing | None`

Returns the `Pricing` dataclass for a model, or `None` if unknown.

### `known_models() -> list[str]`

Sorted list of all canonical model ids in the built-in table.

## Custom pricing

```python
from openai_cost import cost, Pricing

my_table = {
    "my-fine-tune": Pricing(prompt_per_m=3.00, completion_per_m=12.00),
}
usd = cost(model="my-fine-tune", prompt_tokens=500, completion_tokens=100, pricing_table=my_table)
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

45 tests. Covers all pricing variants (cache, batch, combined batch+cache, no-cache fallback), alias resolution, custom tables, `Usage` NamedTuple, `Pricing` immutability, and the public API surface.

## License

MIT.
