# AITextGeneration — Persian AI Text Dataset Generator

Generates a Persian AI-generated text dataset by sampling topics, prompt styles
and models across free OpenAI-compatible endpoints, appending results to a CSV
as they arrive.

## Layout

```
config.json           non-secret config: providers, models, sampling params, prompts
.env                  secrets (API keys) — copy from .env.example, never committed
.topics.txt           1417 Persian topics, one per line
aitextgen/            package: config, clients, worker, pipeline, io
scripts/              one-off helpers (random_topic.py)
notebooks/            data-cleaning notebooks (simple_process.ipynb)
tests/                pytest guardrails
data/                 generated CSVs (gitignored)
```

## Setup

```bash
uv sync
cp .env.example .env    # then fill in API keys
```

## Run

```bash
uv run python -m aitextgen
```

- Picks a random provider (only ones with a key in `.env`), model, topic and
  prompt style per sample; varies temperature/top-p for diversity.
- Appends to `data/persian_ai_generated_dataset4.csv` every `buffer_size`
  samples, so progress survives crashes.
- Ctrl+C (or SIGTERM) stops new requests, drains in-flight ones, saves
  everything completed.

## Add a provider

Add an entry to `config.json` under `providers` (base_url, key_env, proxy,
models) and its key to `.env`. No code change needed.

## Tests

```bash
uv run pytest
```

## License

MIT — see [LICENSE](LICENSE).
