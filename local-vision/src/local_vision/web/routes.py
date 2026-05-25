"""HTTP + SSE routes for the M1 UI.

The app holds ONE active observation run at a time (per PLAN.md scope).
Endpoints:

* ``GET /``             — serves the single-page UI
* ``POST /start``       — start a new run (prepared or upload mode)
* ``POST /stop``        — stop the current run, if any
* ``GET  /events``      — server-sent events stream: per-frame updates + fires
* ``GET  /status``      — small JSON describing the current run (or null)

A single in-process broker fan-outs the sampler's queue to all connected SSE
clients (so opening a second tab during a run gets the same live feed).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ..pipeline.decider import Decider
from ..pipeline.sampler import ObservationRun, SamplerUpdate
from ..pipeline.source import Segment, SyntheticLoopSource, FileVideoSource
from ..pipeline.trigger import NegativeTriggerSpec, TriggerMode, TriggerSpec
from ..pipeline.vision_client import OllamaGemma4Client
from ..store.events import EventStore


# ---------------------------------------------------------------------------
# Defaults for the "prepared mode" demo (drinking-water counter)
# ---------------------------------------------------------------------------

PREPARED_PART1 = "video-samples/water-desk-parts/loop-water-part-1.mp4"
PREPARED_PART2 = "video-samples/water-desk-parts/loop-water-part-2.mp4"
# Phrased to match what the model actually perceives. The earlier attempt
# ("is there a bottle on the desk?", negated) failed because Gemma read it
# as "is a bottle visible?" and answered yes even when the bottle was clearly
# being held. Asking the affirmative question lets the model's own reasoning
# do the work — it spontaneously says "being held by hands" when it sees that.
PREPARED_OBSERVATION = (
    "Is a person's hand currently holding or lifting the clear plastic "
    "water bottle (as if picking it up to drink)?"
)
PREPARED_NEGATE = False
PREPARED_TRIGGER = TriggerSpec(mode=TriggerMode.SUSTAINED, min_duration_sec=1.5)
# 20 seconds without a drink → poring goes green-and-sad. Re-arms on the next
# drink event.
PREPARED_NEGATIVE = NegativeTriggerSpec(timeout_sec=20.0, enabled=True)
# Inference takes ~0.7s per frame on M3 Max + native arm64 Ollama. Sampling
# at 1 Hz leaves comfortable headroom and keeps playback real-time.
# (At 2 Hz we'd be processing faster than the model can keep up.)
PREPARED_SAMPLE_HZ = 1.0


# ---------------------------------------------------------------------------
# A tiny pub/sub so multiple browser tabs can subscribe to the same run.
# ---------------------------------------------------------------------------

class _Broker:
    def __init__(self) -> None:
        self.subscribers: set[asyncio.Queue[SamplerUpdate | None]] = set()

    def publish(self, update: SamplerUpdate | None) -> None:
        for q in list(self.subscribers):
            try:
                q.put_nowait(update)
            except asyncio.QueueFull:
                pass  # slow consumer; drop

    async def subscribe(self) -> "asyncio.Queue[SamplerUpdate | None]":
        q: asyncio.Queue[SamplerUpdate | None] = asyncio.Queue(maxsize=64)
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: "asyncio.Queue[SamplerUpdate | None]") -> None:
        self.subscribers.discard(q)


# ---------------------------------------------------------------------------
# Active-run state (singleton)
# ---------------------------------------------------------------------------

class _RunSlot:
    def __init__(self) -> None:
        self.run: ObservationRun | None = None
        self.task: asyncio.Task[None] | None = None
        self.observation_text: str = ""
        self.mode_label: str = ""
        self.broker = _Broker()


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

def build_router(*, store: EventStore, project_root: Path) -> APIRouter:
    r = APIRouter()
    slot = _RunSlot()

    def template_index() -> str:
        return (project_root / "src" / "local_vision" / "web" / "templates" / "index.html").read_text()

    @r.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(template_index())

    @r.get("/status")
    async def status() -> JSONResponse:
        if slot.run is None:
            return JSONResponse({"running": False})
        spec = slot.run.evaluator.state.spec
        return JSONResponse({
            "running": True,
            "run_id": slot.run.run_id,
            "mode": slot.mode_label,
            "observation": slot.observation_text,
            "sample_hz": 1.0 / slot.run.sample_interval_sec,
            "trigger_mode": spec.mode.value,
            "min_duration_sec": spec.min_duration_sec,
        })

    @r.post("/stop")
    async def stop() -> JSONResponse:
        if slot.run is None:
            return JSONResponse({"stopped": False, "reason": "no run"})
        slot.run.stop()
        if slot.task:
            try:
                await asyncio.wait_for(slot.task, timeout=3.0)
            except asyncio.TimeoutError:
                slot.task.cancel()
        slot.run = None
        slot.task = None
        return JSONResponse({"stopped": True})

    @r.post("/start")
    async def start(
        request: Request,
        mode: Annotated[str, Form()] = "prepared",
        observation: Annotated[str, Form()] = "",
        trigger_mode: Annotated[str, Form()] = "sustained",
        min_duration_sec: Annotated[float, Form()] = 1.5,
        sample_hz: Annotated[float, Form()] = 2.0,
        video: Annotated[UploadFile | None, File()] = None,
    ) -> JSONResponse:
        if slot.run is not None:
            raise HTTPException(409, "an observation is already running; stop it first")

        run_id = str(uuid.uuid4())[:8]

        negative_spec: NegativeTriggerSpec | None = None
        if mode == "prepared":
            source = SyntheticLoopSource(
                [
                    Segment(project_root / PREPARED_PART1, loops=(2, 4)),
                    Segment(project_root / PREPARED_PART2, loops=1),
                ],
                seed=None,
            )
            decider = Decider(
                OllamaGemma4Client(),
                question=PREPARED_OBSERVATION,
                negate=PREPARED_NEGATE,
            )
            trigger_spec = PREPARED_TRIGGER
            negative_spec = PREPARED_NEGATIVE
            sample_hz = PREPARED_SAMPLE_HZ
            obs_text = "Drinking water (preset)"
            mode_label = "prepared"
        elif mode == "upload":
            if video is None or not observation.strip():
                raise HTTPException(400, "upload mode requires both video and observation")
            # Save the uploaded video to a tmp path the source can re-open.
            uploaded_path = project_root / ".uploads" / f"{run_id}_{video.filename}"
            uploaded_path.parent.mkdir(parents=True, exist_ok=True)
            with uploaded_path.open("wb") as f:
                f.write(await video.read())
            source = FileVideoSource(uploaded_path)
            decider = Decider(
                OllamaGemma4Client(),
                question=observation.strip(),
                negate=False,
            )
            try:
                tmode = TriggerMode(trigger_mode)
            except ValueError:
                raise HTTPException(400, f"invalid trigger_mode={trigger_mode!r}")
            trigger_spec = TriggerSpec(mode=tmode, min_duration_sec=min_duration_sec)
            obs_text = observation.strip()
            mode_label = "upload"
        else:
            raise HTTPException(400, f"unknown mode={mode!r}")

        run = ObservationRun(
            run_id=run_id,
            source=source,
            decider=decider,
            trigger_spec=trigger_spec,
            negative_spec=negative_spec,
            sample_hz=sample_hz,
        )
        slot.run = run
        slot.observation_text = obs_text
        slot.mode_label = mode_label

        await store.record_run(
            run_id=run_id,
            started_at_unix=time.time(),
            mode=mode_label,
            observation=obs_text,
            trigger_mode=trigger_spec.mode.value,
            min_duration_sec=trigger_spec.min_duration_sec,
            sample_hz=sample_hz,
        )

        async def _run_and_relay() -> None:
            # Start the sampler in the background
            sampler_task = asyncio.create_task(run.run())
            try:
                while True:
                    update = await run.updates.get()
                    if update is None:  # sentinel
                        slot.broker.publish(None)
                        break
                    slot.broker.publish(update)
                    # Persist to SQLite (no blocking — aiosqlite + bg)
                    if update.decision is not None:
                        await store.record_observation(
                            run_id=update.run_id,
                            ts_virtual_sec=update.ts_virtual_sec,
                            ts_wall_unix=update.ts_wall_unix,
                            condition=update.decision.condition,
                            raw_verdict=update.decision.raw_verdict,
                            reason=update.decision.reason,
                            raw_response=update.decision.raw_response,
                            latency_sec=update.decision.latency_sec,
                            parsed=update.decision.parsed,
                        )
                    if update.fired is not None:
                        await store.record_trigger(
                            run_id=update.run_id,
                            fired_at_virtual_sec=update.fired.fired_at_sec,
                            fired_at_wall_unix=update.ts_wall_unix,
                            reason=update.fired.reason,
                            observation=obs_text,
                            trigger_mode=trigger_spec.mode.value,
                            min_duration_sec=trigger_spec.min_duration_sec,
                        )
            finally:
                # Ensure sampler is cleaned up on cancel or exit
                if not sampler_task.done():
                    run.stop()
                    try:
                        await asyncio.wait_for(sampler_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        sampler_task.cancel()
                slot.run = None
                slot.task = None

        slot.task = asyncio.create_task(_run_and_relay())

        return JSONResponse({
            "started": True,
            "run_id": run_id,
            "mode": mode_label,
            "observation": obs_text,
        })

    @r.get("/events")
    async def events(request: Request) -> StreamingResponse:
        q = await slot.broker.subscribe()

        async def gen():
            try:
                # Tell the client we're connected so it knows to expect data
                yield f"event: hello\ndata: {{}}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        update = await asyncio.wait_for(q.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        # Heartbeat so proxies don't kill the connection
                        yield ": keepalive\n\n"
                        continue
                    if update is None:
                        yield "event: end\ndata: {}\n\n"
                        break
                    payload = _serialize_update(update)
                    yield f"data: {json.dumps(payload)}\n\n"
            finally:
                slot.broker.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })

    return r


def _serialize_update(u: SamplerUpdate) -> dict:
    decision = None
    if u.decision is not None:
        decision = {
            "condition": u.decision.condition,
            "raw_verdict": u.decision.raw_verdict,
            "reason": u.decision.reason,
            "latency_sec": u.decision.latency_sec,
            "parsed": u.decision.parsed,
        }
    fired = None
    if u.fired is not None:
        fired = asdict(u.fired)
    fired_negative = None
    if u.fired_negative is not None:
        fired_negative = asdict(u.fired_negative)
    return {
        "ts_virtual_sec": u.ts_virtual_sec,
        "ts_wall_unix": u.ts_wall_unix,
        "decision": decision,
        "fired": fired,
        "fired_negative": fired_negative,
        "is_negative_state": u.is_negative_state,
        "preview_b64": u.preview_jpeg_b64,
        "true_episode_duration_sec": u.true_episode_duration_sec,
    }
