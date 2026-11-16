# AITextGeneration — Persian AI Text Dataset Generator

Generates a Persian AI-generated text dataset by sampling topics, prompt
styles and models across free OpenAI-compatible endpoints, appending results
to a CSV as they arrive.

## Layout

```
main.py                the generator (config, providers, worker loop)
topics.txt             1417 Persian topics, one per line
random_topic.py        prints a random topic (copies to clipboard)
simpleProcess.ipynb    merges/cleans the raw CSV outputs into a final dataset
tests/                 pytest guardrails (topics, CSV append, no hardcoded secrets)
.env.example           environment variable template — copy to .env
```

Datasets (`*.csv`) and `.env` are gitignored — they never enter git.

## Setup

```bash
uv sync
cp .env.example .env    # then fill in API keys
```

## Run

```bash
uv run python main.py
```

- Picks a random provider (only ones with a key in `.env`), model, topic and
  prompt style per sample; varies temperature/top-p for diversity.
- Appends to `persian_ai_generated_dataset4.csv` every 5 samples, so progress
  survives crashes.
- Ctrl+C (or SIGTERM) stops new requests, drains in-flight ones, saves
  everything completed.

## Providers

Configured in the `PROVIDERS` table in `main.py` — base URL, key env var,
proxy flag and model list in one place. To add a provider: add an entry there
and its key to `.env`.

## Clean the raw CSVs

```bash
uv run jupyter notebook simpleProcess.ipynb
```

Strips reasoning sections, drops empty rows and merges the raw exports into
one final dataset.

## Tests

```bash
uv run pytest
```

## License

MIT — see [LICENSE](LICENSE).
