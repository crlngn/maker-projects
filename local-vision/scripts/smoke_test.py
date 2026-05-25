"""Smoke-test a multimodal Ollama model end to end.

Sends a single image + yes/no-style prompt to a locally running Ollama
server, prints the response and latency.

Usage:
    uv run python scripts/smoke_test.py \\
        [--model gemma4:e2b] \\
        [--image resources/poring-sheet.png] \\
        [--prompt "..."] \\
        [--runs 1]
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma4:e2b"
DEFAULT_IMAGE = Path(__file__).resolve().parent.parent / "resources" / "poring-sheet.png"
DEFAULT_PROMPT = (
    "Look at this image. Is there a pink, round cartoon creature or slime character "
    "visible? Respond with exactly one word ('yes' or 'no'), then a colon, then one "
    "short sentence describing what you see. Example: 'yes: I see a small pink slime'."
)


def _ms(ns: int | None) -> int:
    return (ns or 0) // 1_000_000


def run_once(model: str, image_b64: str, prompt: str) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 200},
    }
    start = time.perf_counter()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    wall_ms = int((time.perf_counter() - start) * 1000)
    return {
        "wall_ms": wall_ms,
        "response": data.get("response", "").strip(),
        "load_ms": _ms(data.get("load_duration")),
        "prompt_eval_ms": _ms(data.get("prompt_eval_duration")),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_ms": _ms(data.get("eval_duration")),
        "eval_count": data.get("eval_count"),
        "total_ms_reported": _ms(data.get("total_duration")),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--image", default=str(DEFAULT_IMAGE))
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--runs", type=int, default=1)
    args = p.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"image not found: {image_path}", file=sys.stderr)
        return 1

    image_b64 = base64.b64encode(image_path.read_bytes()).decode()

    print(f"model:  {args.model}")
    print(f"image:  {image_path} ({len(image_b64) // 1024} KB base64)")
    print(f"prompt: {args.prompt[:100]}{'...' if len(args.prompt) > 100 else ''}\n")

    for i in range(1, args.runs + 1):
        label = "cold" if i == 1 else f"warm #{i - 1}"
        try:
            r = run_once(args.model, image_b64, args.prompt)
        except Exception as exc:
            print(f"run {i} ({label}): ERROR {exc}")
            return 2
        gap_ms = r["wall_ms"] - (r["load_ms"] + r["prompt_eval_ms"] + r["eval_ms"])
        gen_rate = (r["eval_count"] / (r["eval_ms"] / 1000)) if r["eval_ms"] else 0
        print(f"run {i} ({label}): {r['wall_ms']/1000:>5.1f}s wall | "
              f"load {r['load_ms']:>5}ms | "
              f"prompt {r['prompt_eval_ms']:>5}ms ({r['prompt_eval_count']} tok) | "
              f"gen {r['eval_ms']:>5}ms ({r['eval_count']} tok, {gen_rate:.1f} tok/s) | "
              f"unreported gap {gap_ms:>5}ms")
        print(f"         response: {r['response'][:120]}{'...' if len(r['response']) > 120 else ''}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
