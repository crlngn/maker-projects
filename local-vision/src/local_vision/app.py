"""FastAPI entrypoint for the M1 observation prototype."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .store.events import EventStore
from .web.routes import build_router

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def create_app() -> FastAPI:
    app = FastAPI(title="Local Vision — Phase 1")

    store = EventStore(db_path=PROJECT_ROOT / "events.db")
    router = build_router(store=store, project_root=PROJECT_ROOT)
    app.include_router(router)

    # Static: app.js / app.css / poring strip etc., plus generated assets
    app.mount(
        "/static",
        StaticFiles(directory=str(Path(__file__).parent / "web" / "static")),
        name="static",
    )
    app.mount(
        "/resources",
        StaticFiles(directory=str(PROJECT_ROOT / "resources")),
        name="resources",
    )

    @app.on_event("startup")
    async def _on_startup() -> None:
        await store.init()

    return app


app = create_app()
