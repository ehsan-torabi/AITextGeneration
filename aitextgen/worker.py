"""Generate one sample: pick provider/model/topic/prompt, call the API."""

from __future__ import annotations

import random
import time
from typing import Any

from openai import OpenAI


def generate_sample(
    sample_id: int,
    clients: dict[str, OpenAI],
    config: dict[str, Any],
    topics: list[str],
    prompts: list[dict[str, str]],
) -> dict[str, Any] | None:
    provider = random.choice(list(clients))
    cfg = config["providers"][provider]
    model = random.choice(cfg["models"])
    topic = random.choice(topics)
    prompt_obj = random.choice(prompts)

    temperature = round(random.uniform(0.3, 0.95), 2)
    top_p = round(random.uniform(0.8, 1.0), 2)
    time.sleep(random.uniform(0.5, 1.5))

    try:
        response = clients[provider].chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt_obj["prompt"]},
                {"role": "user", "content": f"موضوع: {topic}"},
            ],
            temperature=temperature,
            top_p=top_p,
            timeout=300.0,
        )
    except Exception as exc:
        print(f"[Sample {sample_id}] Error using {model} ({provider}): {exc}")
        return None

    generated_text = (response.choices[0].message.content or "").strip()
    if not generated_text:
        return None

    return {
        "topic": topic,
        "generated_text": generated_text,
        "model": model,
        "provider": provider,
        "prompt_style": prompt_obj["style"],
        "temperature": temperature,
        "top_p": top_p,
        "char_count": len(generated_text),
        "word_count": len(generated_text.split()),
    }
