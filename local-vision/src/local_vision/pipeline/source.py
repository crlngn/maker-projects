"""Video sources for the observation pipeline.

A `VideoSource` yields `(virtual_timestamp_sec, frame_bgr)` pairs. The virtual
timestamp is the elapsed time **as if the source were a live camera**: it
keeps increasing across loop iterations, so consumers can reason about
real-world drink-events even when the underlying media is being looped.

Implementations:

* `FileVideoSource(path)` — plays a single mp4 once, exhausting at EOF.
* `SyntheticLoopSource(segments)` — chains multiple file sources with optional
  random loop counts per segment, looping the whole sequence forever.

These are intentionally pull-based generators (caller asks for the next frame
when ready). The sampler in `pipeline/sampler.py` decides cadence on top.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np


class VideoSource(Protocol):
    """A source of (virtual_timestamp_seconds, frame_bgr) tuples."""

    def frames(self) -> Iterator[tuple[float, np.ndarray]]: ...
    def fps(self) -> float: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# FileVideoSource: a single mp4, played once.
# ---------------------------------------------------------------------------

class FileVideoSource:
    """Reads a single video file front-to-back; exhausts at EOF."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"could not open video: {self.path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def duration_sec(self) -> float:
        return self._n_frames / self._fps

    def fps(self) -> float:
        return self._fps

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        # Read sequentially; emit timestamps based on frame index / native fps.
        idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield idx / self._fps, frame
            idx += 1

    def close(self) -> None:
        self._cap.release()


# ---------------------------------------------------------------------------
# SyntheticLoopSource: chain multiple files with optional random loop counts.
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """One leg of the synthetic loop sequence.

    `loops` is either an int (fixed count) or a (min, max) tuple — re-rolled
    on every iteration of the outer sequence. `1` plays once.
    """
    path: str | Path
    loops: int | tuple[int, int] = 1

    def roll_loops(self, rng: random.Random) -> int:
        if isinstance(self.loops, tuple):
            lo, hi = self.loops
            return rng.randint(lo, hi)
        return int(self.loops)


class SyntheticLoopSource:
    """Chain multiple file sources in a repeating sequence.

    The whole sequence repeats forever. Virtual timestamps are continuous —
    if part-1 is 8 s, the third loop of it ends at virtual t=24 s and the
    next segment starts at 24.0.

    Example
    -------
    >>> src = SyntheticLoopSource([
    ...     Segment("part-1.mp4", loops=(2, 4)),    # random 2-4 each cycle
    ...     Segment("part-2.mp4", loops=1),         # always once
    ... ])
    >>> for ts, frame in src.frames():
    ...     ...  # consumer breaks out when it's seen enough
    """

    def __init__(self, segments: list[Segment], seed: int | None = None):
        if not segments:
            raise ValueError("SyntheticLoopSource requires at least one segment")
        self.segments = segments
        self._rng = random.Random(seed)
        # Pre-open each file to fail-fast on missing paths; we'll rewind on each loop.
        self._caps: list[cv2.VideoCapture] = []
        self._fps_per_seg: list[float] = []
        self._dur_per_seg: list[float] = []
        for seg in self.segments:
            cap = cv2.VideoCapture(str(seg.path))
            if not cap.isOpened():
                raise FileNotFoundError(f"could not open video: {seg.path}")
            self._caps.append(cap)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps_per_seg.append(fps)
            self._dur_per_seg.append(n / fps)

    def fps(self) -> float:
        return self._fps_per_seg[0]

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        """Yield (virtual_ts_sec, frame) forever, advancing through the sequence."""
        virtual_t = 0.0
        cycle_num = 0
        while True:
            cycle_num += 1
            for seg_i, seg in enumerate(self.segments):
                loops_this_cycle = seg.roll_loops(self._rng)
                cap = self._caps[seg_i]
                fps = self._fps_per_seg[seg_i]
                dur = self._dur_per_seg[seg_i]
                for loop_i in range(loops_this_cycle):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        ts = virtual_t + frame_idx / fps
                        yield ts, frame
                        frame_idx += 1
                    virtual_t += dur

    def close(self) -> None:
        for cap in self._caps:
            cap.release()
