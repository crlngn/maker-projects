"""Sampler — async orchestrator that wires source → decider → trigger.

One ``ObservationRun`` represents a single active observation. Lifecycle:

1. Caller constructs the run with a source, decider, trigger, and a callback
   (or an asyncio.Queue) to receive ``SamplerUpdate``s.
2. Caller awaits ``run()``. It yields control between frames so the
   FastAPI event loop stays responsive.
3. Caller calls ``stop()`` (sets an event) to terminate cleanly.

All wall-clock math goes through ``time.monotonic()``; virtual time comes
from the source.
"""

from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass
from typing import AsyncIterator

import cv2

from .decider import Decider, Decision
from .source import VideoSource
from .trigger import (
    NegativeTracker,
    NegativeTriggerEvent,
    NegativeTriggerSpec,
    TriggerEvaluator,
    TriggerEvent,
    TriggerSpec,
)


@dataclass
class SamplerUpdate:
    """Pushed out for every sampled frame (whether or not the trigger fires)."""
    run_id: str
    ts_virtual_sec: float
    ts_wall_unix: float
    decision: Decision | None        # None on inference error (rare)
    fired: TriggerEvent | None
    fired_negative: NegativeTriggerEvent | None
    # Persistent state, sent on every update so the UI is robust to reloads.
    is_negative_state: bool
    # Small JPEG preview of the sampled frame (b64-encoded) — for the UI.
    preview_jpeg_b64: str
    # Useful for the "progress toward firing" status line in the UI.
    true_episode_duration_sec: float | None


class ObservationRun:
    def __init__(
        self,
        *,
        run_id: str,
        source: VideoSource,
        decider: Decider,
        trigger_spec: TriggerSpec,
        negative_spec: NegativeTriggerSpec | None = None,
        sample_hz: float = 2.0,
        preview_max_width: int = 480,
        preview_jpeg_quality: int = 70,
    ):
        self.run_id = run_id
        self.source = source
        self.decider = decider
        self.evaluator = TriggerEvaluator(trigger_spec)
        self.negative_tracker = NegativeTracker(
            negative_spec or NegativeTriggerSpec(enabled=False),
            start_at_sec=0.0,
        )
        self.sample_interval_sec = 1.0 / sample_hz
        self.preview_max_width = preview_max_width
        self.preview_jpeg_quality = preview_jpeg_quality

        self._stop = asyncio.Event()
        self._queue: asyncio.Queue[SamplerUpdate] = asyncio.Queue(maxsize=64)

    @property
    def updates(self) -> asyncio.Queue[SamplerUpdate]:
        return self._queue

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        """Drive the source, sample at the configured cadence, push updates.

        Critical invariants:
          * Inference only runs on **sampled** frames. Non-sampled frames are
            pulled from the source and discarded — cheap, no model call.
          * Wall time is paced to virtual time when we're ahead of schedule
            (so playback feels real-time). If inference falls behind, we don't
            sleep — we just process the next sample as soon as possible.
          * Iteration cooperates with asyncio every ~30 discarded frames so
            the event loop can serve other requests even during the (fast)
            skip phase.
        """
        loop = asyncio.get_running_loop()
        start_wall = loop.time()
        next_sample_virtual = 0.0
        frames_since_yield = 0

        for ts_virtual, frame in self.source.frames():
            if self._stop.is_set():
                break

            # Cooperatively yield during the fast skip-frames phase.
            frames_since_yield += 1
            if frames_since_yield >= 30:
                await asyncio.sleep(0)
                frames_since_yield = 0

            # Skip frames that fall between sample points — no inference cost.
            if ts_virtual < next_sample_virtual:
                continue

            # Real-time pacing: only sleep when we're ahead of wall clock.
            target_wall = start_wall + ts_virtual
            now_wall = loop.time()
            if now_wall < target_wall:
                await asyncio.sleep(target_wall - now_wall)

            # Schedule the next sample at virtual time +interval.
            next_sample_virtual = ts_virtual + self.sample_interval_sec

            # Run model inference in a thread (urllib + OpenCV are blocking).
            try:
                decision = await asyncio.to_thread(self.decider.decide, frame)
            except Exception:
                decision = None

            fired = None
            fired_negative = None
            true_dur: float | None = None
            if decision is not None and decision.parsed:
                fired = self.evaluator.observe(
                    condition=decision.condition,
                    reason=decision.reason,
                    timestamp_sec=ts_virtual,
                )
                if self.evaluator.state.true_episode_started_at is not None:
                    true_dur = ts_virtual - self.evaluator.state.true_episode_started_at

                # Negative tracker observes AFTER positive resets, so a frame
                # that fires positive resets the timer in the same tick.
                if fired is not None:
                    self.negative_tracker.on_positive_fire(ts_virtual)
                fired_negative = self.negative_tracker.observe(ts_virtual)
            # Always tick negative tracker (so timeout can fire even on
            # inference-error frames as long as virtual time advances).
            elif self.negative_tracker.spec.enabled:
                fired_negative = self.negative_tracker.observe(ts_virtual)

            preview = await asyncio.to_thread(
                _encode_preview, frame, self.preview_max_width, self.preview_jpeg_quality
            )
            update = SamplerUpdate(
                run_id=self.run_id,
                ts_virtual_sec=ts_virtual,
                ts_wall_unix=time.time(),
                decision=decision,
                fired=fired,
                fired_negative=fired_negative,
                is_negative_state=self.negative_tracker.in_negative_state,
                preview_jpeg_b64=preview,
                true_episode_duration_sec=true_dur,
            )
            await self._queue.put(update)

        # Sentinel so consumers can break their iteration loop cleanly.
        await self._queue.put(None)  # type: ignore[arg-type]
        self.source.close()


def _encode_preview(frame, max_width: int, quality: int) -> str:
    h, w = frame.shape[:2]
    if w > max_width:
        new_w = max_width
        new_h = int(h * new_w / w)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode()


async def drain(run: ObservationRun) -> AsyncIterator[SamplerUpdate]:
    """Helper: iterate updates from a run's queue until the sentinel arrives."""
    while True:
        upd = await run.updates.get()
        if upd is None:  # type: ignore[comparison-overlap]
            return
        yield upd
