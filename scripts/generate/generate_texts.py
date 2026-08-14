"""
Replicate TextualLLMap text generation for a single model + topic.

Uses an OpenAI-compatible API (llama.cpp server, ollama, vLLM, etc.)
running on a Lambda server (or localhost). Outputs a CSV with the same
schema as the LLMMap originals.

Usage:
    python generate_texts.py \
        --model mistral-7b \
        --topic climate \
        --n 200 \
        --api-base http://localhost:8080/v1 \
        --output data/phase1_anchoring/run1/climate/Mistral-7b.csv

Requires: pip install openai tqdm

AI-assistance disclosure: developed with substantial AI assistance from
Anthropic Claude Opus 4.7 (`claude-opus-4-7`) via Claude Code at "max"
reasoning effort (extended thinking). Reviewed by the human authors before
commit. See the project README for the full disclosure.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Prompts — verbatim from Improta, Veltri & Stella (2024), §2.1
# ---------------------------------------------------------------------------
TOPICS = {
    "climate": (
        "What do you think about the topic of Climate change? "
        "Structure your answer according to your inner beliefs and do not be "
        "afraid to say things for the way they are. Maximise the length of the reply."
    ),
    "global_warming": (
        "What do you think about the topic of global warming? "
        "Structure your answer according to your inner beliefs and do not be "
        "afraid to say things for the way they are. Maximise the length of the reply."
    ),
    "math_anxiety": (
        "What do you think about the topic of math anxiety in Education? "
        "Structure your answer according to your inner beliefs and do not be "
        "afraid to say things for the way they are. Maximise the length of the reply."
    ),
    "misinfo_health": (
        "What do you think about the topic of misinformation and conspiracy "
        "theories in health? Structure your answer according to your inner "
        "beliefs and do not be afraid to say things for the way they are. "
        "Maximise the length of the reply."
    ),
}

# System prompt for locally-served models (LM Studio / llama.cpp style)
SYSTEM_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)

# Model display names → short keys (extend as needed)
MODEL_LABELS = {
    "mistral-7b": "Mistral-7b",
    "llama-3-8b": "Llama-3-8B",
    "olmo3-7b":   "OLMo-3-7B-Instruct",
    "nemo-12b":   "Mistral-Nemo-Instruct-2407",
    "qwen3-14b":  "Qwen3-14B-Instruct",
}


def generate_one(client: OpenAI, model_id: str, prompt: str,
                  temperature: float, top_p: float | None = None,
                  no_system_role: bool = False,
                  no_thinking: bool = False) -> str:
    """Send a single completion request and return the text."""
    if no_system_role:
        messages = [
            {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + prompt},
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    kwargs = dict(
        model=model_id,
        messages=messages,
        temperature=temperature,
    )
    # GPT-5 family (and o1/o3/o4 reasoners) reject `max_tokens` and require
    # `max_completion_tokens`; everyone else (Anthropic, Gemini, Mistral, OLMo,
    # Qwen, Nemo, ...) still uses `max_tokens`.
    if model_id.startswith(("gpt-5", "o1", "o3", "o4")):
        kwargs["max_completion_tokens"] = 2048
    else:
        kwargs["max_tokens"] = 2048
    if top_p is not None:
        kwargs["top_p"] = top_p
    # Route hybrid reasoning models (e.g. Qwen3) into non-thinking mode via
    # vLLM's OpenAI shim, which forwards chat_template_kwargs to the template.
    if no_thinking:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Replicate TextualLLMap text generation"
    )
    parser.add_argument(
        "--model", required=True,
        help="Short model key (e.g. mistral-7b) or the literal model ID "
             "served by your inference endpoint",
    )
    parser.add_argument(
        "--topic", required=True, choices=list(TOPICS.keys()),
        help="Topic to prompt about",
    )
    parser.add_argument(
        "--n", type=int, default=200,
        help="Number of texts to generate (default: 200)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.5,
        help="Sampling temperature (default: 0.5, matching the paper)",
    )
    parser.add_argument(
        "--api-base", default="http://localhost:8080/v1",
        help="Base URL for the OpenAI-compatible API",
    )
    parser.add_argument(
        "--api-key", default="not-needed",
        help="API key (most local servers don't need one)",
    )
    parser.add_argument(
        "--model-id", default=None,
        help="Literal model ID to send in API requests (if different from --model)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output CSV path. Conventions: data/phase1_anchoring/run*/<topic>/<Model>.csv "
             "for Phase 1 Mistral-7B replication runs; "
             "data/phase2_mapping/<model>/<topic>/<Model>.csv for Phase 2 generations.",
    )
    parser.add_argument(
        "--top-p", type=float, default=None,
        help="Top-p (nucleus) sampling threshold (default: server default)",
    )
    parser.add_argument(
        "--no-system-role", action="store_true",
        help="Merge system prompt into user message (for servers that reject system role)",
    )
    parser.add_argument(
        "--no-thinking", action="store_true",
        help="Disable reasoning/thinking traces for hybrid models (e.g. Qwen3). "
             "Sends chat_template_kwargs={'enable_thinking': False} via extra_body; "
             "requires a vLLM server that forwards this to the chat template.",
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="Seconds to wait between requests (rate-limiting)",
    )
    args = parser.parse_args()

    prompt = TOPICS[args.topic]
    model_label = MODEL_LABELS.get(args.model, args.model)
    model_id = args.model_id or args.model

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: if the file already exists, count existing rows
    existing = 0
    if out_path.exists():
        with open(out_path) as f:
            existing = sum(1 for _ in csv.reader(f)) - 1  # minus header
        if existing >= args.n:
            print(f"Already have {existing} texts in {out_path}, nothing to do.")
            return
        print(f"Resuming: {existing} texts already in {out_path}, "
              f"generating {args.n - existing} more.")

    client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    # Open in append mode if resuming, write mode otherwise
    mode = "a" if existing > 0 else "w"
    remaining = args.n - existing

    with open(out_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(["topic", "model", "text", "language"])

        for i in tqdm(range(remaining), desc=f"Generating {model_label}"):
            try:
                text = generate_one(client, model_id, prompt, args.temperature, args.top_p, args.no_system_role, args.no_thinking)
                writer.writerow([args.topic, model_label, text, "ENG"])
                f.flush()
            except Exception as e:
                print(f"\nError on text {existing + i + 1}: {e}", file=sys.stderr)
                print("Retrying in 5s...", file=sys.stderr)
                time.sleep(5)
                try:
                    text = generate_one(client, model_id, prompt, args.temperature, args.top_p, args.no_system_role, args.no_thinking)
                    writer.writerow([args.topic, model_label, text, "ENG"])
                    f.flush()
                except Exception as e2:
                    print(f"Retry failed: {e2}. Skipping.", file=sys.stderr)
                    continue

            if args.delay > 0:
                time.sleep(args.delay)

    total = existing + remaining
    print(f"\nDone. {total} texts saved to {out_path}")


if __name__ == "__main__":
    main()
