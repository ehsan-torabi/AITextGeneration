"""Guardrail tests: behavior that must survive refactors of main.py."""

import re
from pathlib import Path

import main

ROOT = Path(__file__).resolve().parent.parent


def test_load_topics_returns_many_nonempty_strings():
    assert len(main.topics_list) > 1000
    assert all(isinstance(t, str) and t for t in main.topics_list)


def test_append_to_csv_writes_header_once(tmp_path):
    path = tmp_path / "out.csv"
    rows = [{"topic": "a", "generated_text": "x", "model": "m"}]
    main.append_to_csv(rows, filename=path)
    main.append_to_csv(rows, filename=path)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    assert lines[0] == "topic,generated_text,model"
    assert len(lines) == 3


def test_append_to_csv_ignores_empty_batch(tmp_path):
    path = tmp_path / "out.csv"
    main.append_to_csv([], filename=path)
    assert not path.exists()


def test_providers_table_is_consistent():
    assert main.PROVIDERS
    for name, cfg in main.PROVIDERS.items():
        assert cfg["base_url"].startswith("https://"), name
        assert cfg["key_env"], name
        assert cfg["models"], name
        assert isinstance(cfg["proxy"], bool), name


def test_system_prompts_have_style_and_prompt():
    assert main.SYSTEM_PROMPTS
    for p in main.SYSTEM_PROMPTS:
        assert p["style"] and p["prompt"]


def test_no_hardcoded_secrets():
    # Pattern pieces are concatenated so this file never matches itself.
    patterns = [
        re.compile("sk-" + r"[A-Za-z0-9]{16,}"),
        re.compile("nvapi-" + r"[A-Za-z0-9]{16,}"),
        re.compile("ix_" + r"[a-f0-9]{24,}"),
    ]
    skip = {".venv", "node_modules", ".git", ".zcode", "__pycache__"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".ipynb", ".md", ".toml", ".txt"}:
            continue
        if skip & set(path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pat in patterns:
            assert not pat.search(text), f"hardcoded secret in {path}"
