"""Generate a Persian AI-generated text dataset by sampling free OpenAI-compatible endpoints.

Run:  uv run python main.py
Keys: copy .env.example to .env and fill in (never commit .env).
"""

import os
import random
import signal
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import httpx
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
TOTAL_SAMPLES = 5000  # Total samples to generate
MAX_WORKERS = 20  # Concurrent requests (adjust based on rate limits)
CSV_FILENAME = "persian_ai_generated_dataset4.csv"
BUFFER_SIZE = 5  # Save to disk every 5 samples
PROXY_URL = os.environ.get("PROXY_URL", "").strip()

# ==========================================
# PROVIDERS: one table for base URL, key env var, proxy flag and model list.
# A provider is used only when its key env var is set in .env.
# ==========================================
PROVIDERS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "proxy": True,
        "models": [
            "poolside/laguna-s-2.1:free",
            "poolside/laguna-xs-2.1:free",
            "cohere/north-mini-code:free",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "google/gemma-4-26b-a4b-it:free",
            "google/gemma-4-31b-it:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "nvidia/nemotron-3-nano-30b-a3b:free",
            "nvidia/nemotron-nano-12b-v2-vl:free",
            "nvidia/nemotron-nano-9b-v2:free",
            "inclusionai/ling-2.6-flash",
            "qwen/qwen3.7-flash",
            "meta-llama/llama-3.1-8b-instruct",
            "ibm-granite/granite-4.0-h-micro",
            "openai/gpt-oss-120b",
            "mistralai/mistral-nemo",
            "tencent/hy3-preview",
            "microsoft/phi-4",
            "z-ai/glm-4.7-flash",
            "deepseek/deepseek-v4-flash",
            "stepfun/step-3.5-flash",
            "xiaomi/mimo-v2.5",
            "inclusionai/ring-2.6-1t",
            "openai/gpt-4o-mini",
            "minimax/minimax-m2.5",
            "deepseek/deepseek-v4-pro",
            "z-ai/glm-5.2:free",
            "moonshotai/kimi-k3",
            "openai/gpt-5.6-luna",
            "google/gemini-2.5-flash-lite",
            "google/gemini-3.5-flash-lite",
            "deepseek/deepseek-v4-flash-0731",
            "openai/gpt-5.4-mini",
            "openai/gpt-5.4-nano",
            "qwen/qwen3.7-plus",
            "x-ai/grok-4.3",
            "qwen/qwen3.8-max",
            "meta/muse-spark-1.2",
            "stealth/ox-alpha",
            "liquid/lfm-2.5-2.6b:free",
            "nvidia/nemotron-3.5-lightning:free",
            "dots-studio/dots-3-note-preview:free",
        ],
    },
    "NVIDIA": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "key_env": "NVIDIA_API_KEY",
        "proxy": True,
        "models": [
            "nvidia/nemotron-3-super-120b-a12b",
            "openai/gpt-oss-120b",
            "meta/llama-3.3-70b-instruct",
            "openai/gpt-oss-20b",
            "minimaxai/minimax-m3",
            "meta/llama-3.2-90b-vision-instruct",
            "google/gemma-4-31b-it",
            "deepseek-ai/deepseek-v4-flash-0731",
            "meta/muse-glimmer-30b",
            "nvidia/nemotron-3.5-lightning-30b-a3b",
        ],
    },
    "chat.b.ai": {
        "base_url": "https://api.b.ai/v1",
        "key_env": "CHATBAI_API_KEY",
        "proxy": False,
        "models": ["deepseek-v4-flash", "deepseek-v4-flash-vision-exp", "hy3"],
    },
    "free_endpoint": {
        "base_url": "https://model.inferx.net/endpoints/v1",
        "key_env": "INFERENCEX_API_KEY",
        "proxy": False,
        "models": [
            "Devstral-2-123B-Instruct-2512-int4-AutoRound",
            "Qwen3.6-35B-A3B-FP8",
        ],
    },
    "anymodel_endpoint": {
        "base_url": "https://anymodel.org/v1",
        "key_env": "ANYMODEL_API_KEY",
        "proxy": True,
        "models": [
            "am/deepseek-v4-pro",
            "am/deepseek-v4-flash",
            "am/minimax-m3",
            "am/free",
            "am/gemma-4-31b-it",
            "am/nemotron-3-ultra-550b-a55b",
            "am/gpt-oss-20b",
            "am/laguna-xs-2.1",
            "am/llama-3.2-11b-vision-instruct",
            "am/llama-3.1-nemotron-nano-vl-8b-v1",
            "am/nemotron-3.5-lightning-30b-a3b",
            "am/step-3.7-flash",
            "am/nemotron-3-super-120b-a12b",
            "am/mistral-nemotron",
        ],
    },
    "tabitoken_endpoint": {
        "base_url": "https://tabitoken.com/v1",
        "key_env": "TABITOKEN_API_KEY",
        "proxy": True,
        "models": [
            "claude-opus-4-8",
            "claude-opus-4-8-thinking",
            "claude-opus-5",
            "claude-opus-5-thinking",
        ],
    },
    "gorouter_endpoint": {
        "base_url": "https://gorouter.app/v1",
        "key_env": "GOROUTER_API_KEY",
        "proxy": True,
        "models": [
            "claude-opus-4-8",
            "claude-opus-4-8-thinking",
            "claude-opus-5",
            "claude-opus-5-thinking",
        ],
    },
}


def build_clients():
    """Create an OpenAI client per provider whose key is set."""
    clients = {}
    for name, cfg in PROVIDERS.items():
        key = os.environ.get(cfg["key_env"], "").strip()
        if not key:
            continue
        http_client = httpx.Client(proxy=PROXY_URL) if cfg["proxy"] and PROXY_URL else None
        clients[name] = OpenAI(base_url=cfg["base_url"], api_key=key, http_client=http_client)
    return clients


# ==========================================
# DIVERSE SYSTEM PROMPTS (Crucial for Detection)
# ==========================================
SYSTEM_PROMPTS = [
    {
        "style": "formal_essay",
        "prompt": "یک مقاله مفصل، رسمی و تحلیلی به زبان فارسی درباره موضوع مشخص شده بنویس. از لحن پارسی اکادمیک استفاده کن.",
    },
    {
        "style": "conversational_blog",
        "prompt": "یک پست وبلاگی جذاب، صمیمی و روان به زبان فارسی درباره موضوع داده شده بنویس. لحن گفتگو محور باشد.",
    },
    {
        "style": "qa_explainer",
        "prompt": "به عنوان یک متخصص، به سوال یا موضوع زیر به صورت کاملاً جامع، گام‌به‌گام و ساده به زبان فارسی پاسخ بده.",
    },
    {
        "style": "opinion_critical",
        "prompt": "یک متن انتقادی و استدلالی به زبان فارسی درباره موضوع زیر بنویس و جنبه‌های مختلف آن را نقد بررسی کن.",
    },
    {
        "style": "summary_overview",
        "prompt": "یک خلاصه جامع و ساختاریافته شامل مقدمه، نکات اصلی و نتیجه‌گیری درباره موضوع زیر به زبان فارسی بنویس.",
    },
]

# ==========================================
# SHUTDOWN SIGNAL
# ==========================================
shutdown_flag = False


def handle_shutdown(signum, frame):
    global shutdown_flag
    print(
        f"\nTermination signal ({signum}) received! "
        "Stopping new requests, waiting for remaining active requests to complete..."
    )
    shutdown_flag = True


signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


# ==========================================
# LOAD TOPICS
# ==========================================
def load_topics(file_path="topics.txt"):
    try:
        with open(file_path, encoding="utf-8") as f:
            topics = [line.strip() for line in f if line.strip()]
        if not topics:
            raise ValueError("Topics file is empty.")
        print(f"Loaded {len(topics)} topics successfully.")
        return topics
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found. Please create it.")
        sys.exit(1)


topics_list = load_topics()


# ==========================================
# DATASET SAVE HELPER (Append Mode)
# ==========================================
def append_to_csv(data_rows, filename=CSV_FILENAME):
    """Appends data directly to CSV without reading existing content into RAM."""
    if not data_rows:
        return
    df = pd.DataFrame(data_rows)
    file_exists = os.path.exists(filename)
    df.to_csv(filename, mode="a", index=False, header=not file_exists, encoding="utf-8-sig")
    print(f"Saved {len(data_rows)} rows to {filename}")


# ==========================================
# GENERATION WORKER FUNCTION
# ==========================================
def generate_sample(sample_id, clients):
    provider = random.choice(list(clients))
    model = random.choice(PROVIDERS[provider]["models"])
    topic = random.choice(topics_list)
    prompt_obj = random.choice(SYSTEM_PROMPTS)

    # Vary hyper-parameters for perplexity diversity
    temperature = round(random.uniform(0.3, 0.95), 2)
    top_p = round(random.uniform(0.8, 1.0), 2)

    # Simple sleep for server not rate-limiting
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

        generated_text = response.choices[0].message.content.strip()

        # Guard against empty/faulty output
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

    except Exception as e:
        print(f"[Sample {sample_id}] Error using {model} ({provider}): {e}")
        return None


# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
def main():
    clients = build_clients()
    if not clients:
        sys.exit("No API keys configured. Copy .env.example to .env and fill in keys.")

    print(f"Starting generation of {TOTAL_SAMPLES} samples using {MAX_WORKERS} workers...")

    buffer = []
    completed_count = 0
    drain_notice_shown = False

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = set()
        sample_id = 1

        def submit_pending():
            nonlocal sample_id
            while sample_id <= TOTAL_SAMPLES and len(futures) < MAX_WORKERS and not shutdown_flag:
                futures.add(executor.submit(generate_sample, sample_id, clients))
                sample_id += 1

        submit_pending()

        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                if result:
                    buffer.append(result)
                    completed_count += 1
                    print(
                        f"[{completed_count}/{TOTAL_SAMPLES}] Model: {result['model']} | "
                        f"Style: {result['prompt_style']} | Words: {result['word_count']}"
                    )
                    if len(buffer) >= BUFFER_SIZE:
                        append_to_csv(buffer)
                        buffer.clear()

            if shutdown_flag:
                if not drain_notice_shown:
                    print("Waiting for remaining running requests to complete...")
                    drain_notice_shown = True
            else:
                submit_pending()

    # Final flush on exit
    if buffer:
        append_to_csv(buffer)
        buffer.clear()

    if shutdown_flag:
        print(f"\nExecution interrupted. Saved all completed progress to {CSV_FILENAME}")
    else:
        print(f"\nGeneration complete! Dataset saved to {CSV_FILENAME}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user. Saved completed progress.")
        sys.exit(0)
