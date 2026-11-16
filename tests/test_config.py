"""Guardrails for the aitextgen package: config validation and no hardcoded secrets."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

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


def _config() -> dict:
    with (ROOT / "config.json").open(encoding="utf-8") as f:
        return json.load(f)


def test_config_has_required_top_keys():
    cfg = _config()
    for key in REQUIRED_TOP_KEYS:
        assert key in cfg, key


def test_provider_config_complete():
    cfg = _config()
    assert cfg["providers"]
    for name, prov in cfg["providers"].items():
        for key in PROVIDER_KEYS:
            assert key in prov, f"{name}.{key}"
        assert isinstance(prov["models"], list) and prov["models"], name


def test_system_prompts_have_style_and_prompt():
    cfg = _config()
    assert cfg["system_prompts"]
    for p in cfg["system_prompts"]:
        assert p.get("style") and p.get("prompt")


def test_no_hardcoded_secrets_in_committed_files():
    skip = {".venv", "node_modules", ".git", ".zcode", "__pycache__", ".ruff_cache", "data"}
    pats = [
        re.compile(r"sk-[A-Za-z0-9]{16,}"),
        re.compile(r"nvapi-[A-Za-z0-9]{16,}"),
        re.compile(r"ix_[a-f0-9]{24,}"),
    ]
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in {
            ".py",
            ".ipynb",
            ".md",
            ".json",
            ".toml",
            ".txt",
        }:
            continue
        if skip & set(path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pat in pats:
            assert not pat.search(text), f"hardcoded secret in {path}"
