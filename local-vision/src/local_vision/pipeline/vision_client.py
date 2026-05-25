"""VisionClient — sends a frame + prompt to a local VLM, returns the raw text.

The Protocol exists so we can swap backends without touching the pipeline
(per PLAN.md §4 — Phase 2 swaps in llama.cpp on Pi, Phase 3 adds a
Hailo-accelerated detector fast-path).

Phase 1 has exactly one implementation: ``OllamaGemma4Client``.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


@dataclass
class VisionResult:
    raw_response: str
    latency_sec: float


class VisionClient(Protocol):
    def query(self, *, frame_bgr: np.ndarray, prompt: str) -> VisionResult: ...


# ---------------------------------------------------------------------------
# Ollama client for Gemma 4 E2B-it
# ---------------------------------------------------------------------------


class OllamaGemma4Client:
    """Posts a single frame + prompt to a local Ollama server.

    Sends ``think: False`` and a tight ``num_predict`` because we want short
    constrained answers (see PROMPTS.md). Frames are JPEG-encoded; Gemma
    rescales internally to its vision token budget so we don't pre-resize.
    """

    def __init__(
        self,
        *,
        model: str = "gemma4:e2b",
        url: str = "http://localhost:11434/api/generate",
        timeout_sec: float = 60.0,
        jpeg_quality: int = 85,
        num_predict: int = 60,
    ):
        self.model = model
        self.url = url
        self.timeout = timeout_sec
        self.jpeg_quality = jpeg_quality
        self.num_predict = num_predict

    def query(self, *, frame_bgr: np.ndarray, prompt: str) -> VisionResult:
        ok, buf = cv2.imencode(
            ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        img_b64 = base64.b64encode(buf.tobytes()).decode()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0, "num_predict": self.num_predict},
        }
        start = time.perf_counter()
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        return VisionResult(
            raw_response=(data.get("response") or "").strip(),
            latency_sec=time.perf_counter() - start,
        )
