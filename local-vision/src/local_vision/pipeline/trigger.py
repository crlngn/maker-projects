"""Trigger evaluator: turns a stream of per-frame answers into fire events.

Two modes for Phase 1 (per PLAN.md §3):

* ``INSTANT`` — fires the moment ``condition`` first becomes True
  (with optional debounce via ``min_duration_sec`` to avoid 1-frame flickers).
* ``SUSTAINED`` — fires when ``condition`` has been continuously True for
  ≥ ``min_duration_sec`` seconds.

Both use **rising-edge semantics**: the trigger fires at most once per
true-episode, then re-arms when ``condition`` returns to False. This is
load-bearing — it's what prevents the same drink from firing N times while
the bottle is off the desk.

The synthetic-water experiment showed why debouncing matters: even with a
correct model, a 0.5-second blip (yes→no→yes) was being counted as a drink.
Setting ``min_duration_sec`` ≥ 1-2 seconds filters those cleanly.

``CUMULATIVE`` (e.g. "≥ 2h total within an 8h window") is M3 work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TriggerMode(str, Enum):
    INSTANT = "instant"
    SUSTAINED = "sustained"


@dataclass
class TriggerSpec:
    """User-facing description of what to fire on."""
    mode: TriggerMode = TriggerMode.INSTANT
    min_duration_sec: float = 0.0   # debounce / sustained threshold


@dataclass
class TriggerEvent:
    """Emitted when the trigger fires."""
    fired_at_sec: float       # virtual/wall timestamp from the source
    reason: str               # the model's reason text for the firing frame


@dataclass
class TriggerState:
    """In-memory state of one observation task's trigger evaluator."""
    spec: TriggerSpec
    # When did the current true-episode start? None if condition is False.
    true_episode_started_at: float | None = None
    # Has this true-episode already fired (rising-edge guard)?
    fired_this_episode: bool = False
    # Optional reason from the most recent True observation (so we can attach
    # something meaningful when we eventually fire).
    last_true_reason: str = ""


@dataclass
class NegativeTriggerSpec:
    """Fires when no positive trigger has fired for ``timeout_sec`` seconds.

    Disabled by default; enable explicitly for use cases like
    "alert me if the user hasn't taken a sip in 20 seconds".
    """
    timeout_sec: float = 20.0
    enabled: bool = False


@dataclass
class NegativeTriggerEvent:
    fired_at_sec: float
    seconds_without_positive: float


class NegativeTracker:
    """Tracks elapsed time since the last positive fire (or run start).

    Fires once on entering the negative state; stays in that state until a
    positive fire arrives via ``on_positive_fire()``, at which point it
    re-arms.
    """

    def __init__(self, spec: NegativeTriggerSpec, *, start_at_sec: float = 0.0):
        self.spec = spec
        self.last_positive_at: float = start_at_sec
        self.in_negative_state: bool = False

    def on_positive_fire(self, ts_sec: float) -> None:
        self.last_positive_at = ts_sec
        # Exit negative state if we were in one — the caller (sampler) will
        # observe is_negative_state == False on the next update.
        self.in_negative_state = False

    def observe(self, ts_sec: float) -> NegativeTriggerEvent | None:
        if not self.spec.enabled:
            return None
        if self.in_negative_state:
            return None  # already negative — don't re-fire
        gap = ts_sec - self.last_positive_at
        if gap >= self.spec.timeout_sec:
            self.in_negative_state = True
            return NegativeTriggerEvent(
                fired_at_sec=ts_sec, seconds_without_positive=gap
            )
        return None


class TriggerEvaluator:
    """Pure state machine. Feed it observations, get back optional events."""

    def __init__(self, spec: TriggerSpec):
        if spec.min_duration_sec < 0:
            raise ValueError("min_duration_sec must be >= 0")
        self.state = TriggerState(spec=spec)

    # The condition arg here is what the *decider* produced for THIS frame.
    # It's already been negated if the user wanted "fire when X is NOT true".
    def observe(
        self, *, condition: bool, reason: str, timestamp_sec: float
    ) -> TriggerEvent | None:
        s = self.state
        spec = s.spec

        if not condition:
            # Episode is over — reset.
            s.true_episode_started_at = None
            s.fired_this_episode = False
            s.last_true_reason = ""
            return None

        # condition == True
        if s.true_episode_started_at is None:
            # Start of a new true-episode.
            s.true_episode_started_at = timestamp_sec
        s.last_true_reason = reason

        if s.fired_this_episode:
            return None  # rising-edge: don't re-fire mid-episode

        duration = timestamp_sec - s.true_episode_started_at
        threshold = spec.min_duration_sec
        ready_to_fire = (
            spec.mode == TriggerMode.INSTANT and duration >= threshold
            or spec.mode == TriggerMode.SUSTAINED and duration >= threshold
        )
        if ready_to_fire:
            s.fired_this_episode = True
            return TriggerEvent(fired_at_sec=timestamp_sec, reason=reason)
        return None

    @property
    def current_true_duration(self) -> float | None:
        """For UI: how long has the current true-episode lasted, if any?"""
        return None  # populated by the caller using the last timestamp
