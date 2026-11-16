"""Build OpenAI clients from config; one per provider whose key is set in .env."""

from __future__ import annotations

import os
from typing import Any

import httpx
from openai import OpenAI


def build_clients(config: dict[str, Any]) -> dict[str, OpenAI]:
    """Return {provider_name: OpenAI client} for providers with a configured key."""
    clients: dict[str, OpenAI] = {}
    proxy_url = (config.get("proxy_url") or "").strip()

    for name, cfg in config["providers"].items():
        key = os.environ.get(cfg["key_env"], "").strip()
        if not key:
            continue
        http_client = httpx.Client(proxy=proxy_url) if cfg.get("proxy") and proxy_url else None
        clients[name] = OpenAI(base_url=cfg["base_url"], api_key=key, http_client=http_client)

    return clients
