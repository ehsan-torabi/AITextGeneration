"""Load and validate project config from config.json (non-secret only).

Secrets (API keys) live in .env and are read by clients.build_clients, never here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

REQUIRED_TOP_KEYS = (
    "providers",
    "system_prompts",
    "total_samples",
    "max_workers",
    "buffer_size",
    "csv_filename",
    "topics_file",
)

PROVIDER_KEYS = ("base_url", "key_env", "models", "proxy")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load config.json and validate its shape."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        config: dict[str, Any] = json.load(f)

    for key in REQUIRED_TOP_KEYS:
        if key not in config:
            raise ValueError(f"config.json missing required key: {key}")

    for name, cfg in config["providers"].items():
        if not isinstance(cfg, dict):
            raise ValueError(f"provider {name} must be an object")
        for key in PROVIDER_KEYS:
            if key not in cfg:
                raise ValueError(f"provider {name} missing config key: {key}")
        if not cfg["models"]:
            raise ValueError(f"provider {name} has no models")

    if not config["system_prompts"]:
        raise ValueError("config.json system_prompts is empty")

    return config
