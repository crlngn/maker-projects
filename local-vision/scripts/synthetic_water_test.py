"""End-to-end test of the synthetic-loop video source + bottle-presence VQA.

Plays a synthetic sequence (part-1 × random 2..4 → part-2 → repeat), sampling
at a configurable cadence (default 2 Hz of virtual time). Per sampled frame
asks the model whether the bottle is on the desk, then counts yes→no
transitions as drink events.

This validates two things before we wire either into the M1 UI:
  1. The SyntheticLoopSource yields frames with correct virtual timestamps.
  2. Gemma 4 can reliably distinguish "bottle on desk" vs "bottle being
     grabbed / off desk" on real-time (non-timelapse) video.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2

# Add src/ to path so we can import local_vision.* when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from local_vision.pipeline.source import Segment, SyntheticLoopSource  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:e2b"

PROMPT = (
    "Look at this image of a desk. Is there a clear plastic water bottle "
    "standing upright on the desk surface? Answer with exactly one word: "
    "'yes' if the bottle is on the desk, 'no' if it is not on the desk "
    "(e.g. being held, missing, or out of frame)."
)


def query_model(image_b64: str) -> tuple[str, float]:
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "images": [image_b64],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 10},
    }
    start = time.perf_counter()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data.get("response", "").strip(), time.perf_counter() - start


def parse_yes_no(raw: str) -> str:
    s = raw.strip().lower()
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    return "?"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--part1", default="video-samples/water-desk-parts/loop-water-part-1.mp4")
    p.add_argument("--part2", default="video-samples/water-desk-parts/loop-water-part-2.mp4")
    p.add_argument("--loops-min", type=int, default=2)
    p.add_argument("--loops-max", type=int, default=4)
    p.add_argument("--sample-hz", type=float, default=2.0,
                   help="Samples per second of virtual time (2 = every 0.5s)")
    p.add_argument("--max-virtual-seconds", type=float, default=120.0,
                   help="Stop after this much virtual time has elapsed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resize-w", type=int, default=640)
    p.add_argument("--resize-h", type=int, default=360)
    args = p.parse_args()

    # Fail fast if Ollama isn't reachable
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2):
            pass
    except Exception as exc:
        print(f"Ollama server not reachable at localhost:11434 ({exc}).\n"
              f"Start it with: ollama serve", file=sys.stderr)
        return 1

    src = SyntheticLoopSource(
        [
            Segment(args.part1, loops=(args.loops_min, args.loops_max)),
            Segment(args.part2, loops=1),
        ],
        seed=args.seed,
    )

    sample_interval = 1.0 / args.sample_hz
    next_sample_at = 0.0
    n_samples = 0
    last_print_ts = -1.0

    print(f"sources:  part1={args.part1}")
    print(f"          part2={args.part2}")
    print(f"sampling: every {sample_interval:.2f}s virtual time = {args.sample_hz} Hz")
    print(f"limit:    {args.max_virtual_seconds:.0f}s virtual time")
    print()
    print(f"{'ts(s)':>7}  {'ans':<3}  {'mark':<2}  {'lat':>5}  raw")
    print("-" * 75)

    results: list[tuple[float, str, str]] = []   # (virtual_ts, ans, raw)
    wall_start = time.perf_counter()

    for ts, frame in src.frames():
        if ts > args.max_virtual_seconds:
            break
        if ts < next_sample_at:
            continue
        next_sample_at = ts + sample_interval

        # Resize for upload speed; Gemma rescales internally anyway
        small = cv2.resize(frame, (args.resize_w, args.resize_h),
                           interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buf.tobytes()).decode()

        try:
            raw, lat = query_model(img_b64)
        except Exception as exc:
            print(f"{ts:>7.2f}  ERROR {exc}")
            continue
        ans = parse_yes_no(raw)
        results.append((ts, ans, raw))
        n_samples += 1
        mark = "█" if ans == "yes" else "░" if ans == "no" else "?"
        print(f"{ts:>7.2f}  {ans:<3}  {mark:<2}  {lat:>4.1f}s  {raw[:35]}")

    src.close()
    wall = time.perf_counter() - wall_start

    # Build the timeline visual
    print("\n--- timeline (one char = one sample; █=on-desk, ░=off-desk) ---")
    line = "".join("█" if r[1] == "yes" else "░" if r[1] == "no" else "?" for r in results)
    for i in range(0, len(line), 80):
        print(line[i:i+80])

    # Count off-desk episodes (consecutive 'no' runs)
    events = []
    in_event = False
    event_start_ts = None
    for ts, ans, _ in results:
        if ans == "no" and not in_event:
            in_event = True; event_start_ts = ts
        elif ans != "no" and in_event:
            in_event = False
            events.append((event_start_ts, ts))
    if in_event:
        events.append((event_start_ts, results[-1][0]))

    print(f"\n--- drink events: {len(events)} ---")
    for s, e in events:
        print(f"  virtual t={s:6.2f}s → t={e:6.2f}s  (duration={e-s:.2f}s off-desk)")

    yes_count = sum(1 for _, a, _ in results if a == "yes")
    no_count  = sum(1 for _, a, _ in results if a == "no")
    print(f"\n  samples: {len(results)}  (yes={yes_count}, no={no_count})")
    if results:
        print(f"  wall:    {wall:.1f}s   ({wall / len(results):.2f}s avg per sample)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
