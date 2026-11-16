"""Main generation loop: submit, drain on SIGINT/SIGTERM, flush to CSV."""

from __future__ import annotations

import signal
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

from .clients import build_clients
from .config import load_config
from .io import append_rows
from .worker import generate_sample


def load_topics(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8") as f:
            topics = [line.strip() for line in f if line.strip()]
        if not topics:
            raise ValueError("Topics file is empty.")
        print(f"Loaded {len(topics)} topics successfully.")
        return topics
    except FileNotFoundError:
        print(f"Error: '{path}' not found. Please create it.")
        sys.exit(1)


def run(config_path: str = "config.json") -> None:
    config = load_config(config_path)
    clients = build_clients(config)
    if not clients:
        sys.exit("No API keys configured. Copy .env.example to .env and fill in keys.")

    topics = load_topics(config["topics_file"])
    prompts = config["system_prompts"]

    total = config["total_samples"]
    max_workers = config["max_workers"]
    buffer_size = config["buffer_size"]
    csv_path = config["csv_filename"]

    print(f"Starting generation of {total} samples using {max_workers} workers...")

    buffer: list[dict[str, Any]] = []
    completed = 0
    shutdown_flag = False
    drain_shown = False

    def handle_shutdown(signum, frame):
        nonlocal shutdown_flag
        print(
            f"\nTermination signal ({signum}) received! Stopping new requests, draining in-flight..."
        )
        shutdown_flag = True

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()
        sample_id = 1

        def submit_pending():
            nonlocal sample_id
            while sample_id <= total and len(futures) < max_workers and not shutdown_flag:
                futures.add(
                    executor.submit(generate_sample, sample_id, clients, config, topics, prompts)
                )
                sample_id += 1

        submit_pending()

        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                if result:
                    buffer.append(result)
                    completed += 1
                    print(
                        f"[{completed}/{total}] Model: {result['model']} | "
                        f"Style: {result['prompt_style']} | Words: {result['word_count']}"
                    )
                    if len(buffer) >= buffer_size:
                        append_rows(buffer, csv_path)
                        buffer.clear()

            if shutdown_flag:
                if not drain_shown:
                    print("Waiting for remaining running requests to complete...")
                    drain_shown = True
            else:
                submit_pending()

    if buffer:
        append_rows(buffer, csv_path)
        buffer.clear()

    if shutdown_flag:
        print(f"\nExecution interrupted. Saved completed progress to {csv_path}")
    else:
        print(f"\nGeneration complete! Dataset saved to {csv_path}")


if __name__ == "__main__":  # pragma: no cover
    run()
