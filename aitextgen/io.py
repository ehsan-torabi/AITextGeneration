"""Append generated rows to CSV without reading the whole file into RAM."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd


def append_rows(rows: list[dict[str, Any]], path: str) -> None:
    """Append rows to `path`, writing the header only on first creation."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    file_exists = os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=not file_exists, encoding="utf-8-sig")
