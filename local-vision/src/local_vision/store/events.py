"""SQLite event store.

Two tables:

* ``frame_observations`` — one row per sampled frame (so we can audit what
  the model said and why a trigger fired or didn't).
* ``triggers_fired`` — one row per trigger event (what the UI displays in
  the event log).

Async via ``aiosqlite`` so the sampler loop never blocks on disk I/O.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS frame_observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,
    ts_virtual_sec  REAL    NOT NULL,
    ts_wall_unix    REAL    NOT NULL,
    condition       INTEGER NOT NULL,         -- 0/1, post-negation
    raw_verdict     TEXT    NOT NULL,         -- yes/no/empty as model said
    reason          TEXT    NOT NULL,
    raw_response    TEXT    NOT NULL,
    latency_sec     REAL    NOT NULL,
    parsed          INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_obs_run_ts ON frame_observations(run_id, ts_virtual_sec);

CREATE TABLE IF NOT EXISTS triggers_fired (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,
    fired_at_virtual_sec REAL NOT NULL,
    fired_at_wall_unix   REAL NOT NULL,
    reason          TEXT    NOT NULL,
    observation     TEXT    NOT NULL,         -- the user's natural-language question
    trigger_mode    TEXT    NOT NULL,
    min_duration_sec REAL   NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_fired_run_ts ON triggers_fired(run_id, fired_at_virtual_sec);

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    started_at_unix REAL    NOT NULL,
    mode            TEXT    NOT NULL,          -- 'prepared' or 'upload'
    observation     TEXT    NOT NULL,
    trigger_mode    TEXT    NOT NULL,
    min_duration_sec REAL   NOT NULL,
    sample_hz       REAL    NOT NULL,
    notes           TEXT    NOT NULL DEFAULT ''
);
"""


class EventStore:
    def __init__(self, db_path: str | Path = "events.db"):
        self.db_path = str(db_path)

    async def init(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(SCHEMA)
            await db.commit()

    async def record_run(
        self,
        *,
        run_id: str,
        started_at_unix: float,
        mode: str,
        observation: str,
        trigger_mode: str,
        min_duration_sec: float,
        sample_hz: float,
        notes: str = "",
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO runs(run_id, started_at_unix, mode, observation,"
                " trigger_mode, min_duration_sec, sample_hz, notes)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, started_at_unix, mode, observation,
                 trigger_mode, min_duration_sec, sample_hz, notes),
            )
            await db.commit()

    async def record_observation(
        self,
        *,
        run_id: str,
        ts_virtual_sec: float,
        ts_wall_unix: float,
        condition: bool,
        raw_verdict: str,
        reason: str,
        raw_response: str,
        latency_sec: float,
        parsed: bool,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO frame_observations"
                " (run_id, ts_virtual_sec, ts_wall_unix, condition, raw_verdict,"
                "  reason, raw_response, latency_sec, parsed)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, ts_virtual_sec, ts_wall_unix, int(condition),
                 raw_verdict, reason, raw_response, latency_sec, int(parsed)),
            )
            await db.commit()

    async def record_trigger(
        self,
        *,
        run_id: str,
        fired_at_virtual_sec: float,
        fired_at_wall_unix: float,
        reason: str,
        observation: str,
        trigger_mode: str,
        min_duration_sec: float,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO triggers_fired"
                " (run_id, fired_at_virtual_sec, fired_at_wall_unix, reason,"
                "  observation, trigger_mode, min_duration_sec)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, fired_at_virtual_sec, fired_at_wall_unix, reason,
                 observation, trigger_mode, min_duration_sec),
            )
            await db.commit()
