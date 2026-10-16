"""FastAPI server: WebXR + dashboard websockets, REST API, static web client."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from questcollect.config import ROOT
from questcollect.runtime import Runtime
from questcollect.server.protocol import parse_input

log = logging.getLogger(__name__)
WEB_DIST = ROOT / "web" / "dist"
SEND_PERIOD_S = 0.01
STATUS_PERIOD_S = 0.1


class Hub:
    """Fans runtime events (from the sim thread) out to every connected websocket."""

    def __init__(self):
        self.queues: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def push_threadsafe(self, event: dict) -> None:
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._broadcast, event)

    def _broadcast(self, event: dict) -> None:
        for q in self.queues:
            if q.qsize() < 256:
                q.put_nowait(event)


def create_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="QuestCollect")
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    hub = Hub()
    runtime.add_listener(hub.push_threadsafe)

    # -- REST -------------------------------------------------------------------------------------
    @app.get("/")
    async def index():
        return RedirectResponse("/dashboard/")

    @app.get("/api/scene")
    async def scene():
        js, _ = runtime.scene()
        return JSONResponse(js, headers={"Cache-Control": "no-store"})

    @app.get("/api/scene.bin")
    async def scene_bin():
        _, blob = runtime.scene()
        return Response(blob, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    async def status():
        return runtime.status()

    @app.get("/api/episodes")
    async def episodes():
        return runtime.episodes()

    @app.delete("/api/episodes")
    async def delete_episode(path: str):
        p = Path(path).resolve()
        out = runtime.cfg.out_dir.resolve()
        if out not in p.parents or p.suffix != ".hdf5" or not p.exists():
            raise HTTPException(400, "invalid episode path")
        p.unlink()
        return {"deleted": str(p)}

    @app.post("/api/command")
    async def command(body: dict):
        name = body.get("name")
        if not name:
            raise HTTPException(400, "missing command name")
        runtime.command(name, **(body.get("args") or {}))
        return {"ok": True}

    @app.get("/api/preview/{cam}.jpg")
    async def preview(cam: str):
        img = runtime.previews().get(cam)
        if img is None:
            raise HTTPException(404, "no preview yet")
        return Response(img, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    # -- websockets ---------------------------------------------------------------------------------
    async def _sender(ws: WebSocket, q: asyncio.Queue, pose_period: float):
        last_seq, last_pose_t, last_status_t = -1, 0.0, 0.0
        while True:
            while not q.empty():
                await ws.send_text(json.dumps(q.get_nowait()))
            now = time.monotonic()
            if now - last_pose_t >= pose_period:
                seq, frame = runtime.latest_pose_frame()
                if seq != last_seq and frame:
                    await ws.send_bytes(frame)
                    last_seq, last_pose_t = seq, now
            if now - last_status_t >= STATUS_PERIOD_S:
                st = runtime.status()
                if st:
                    await ws.send_text(json.dumps(st))
                last_status_t = now
            await asyncio.sleep(SEND_PERIOD_S)

    async def _handle_text(text: str):
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            return
        if msg.get("type") == "command" and msg.get("name"):
            runtime.command(msg["name"], **(msg.get("args") or {}))

    async def _session(ws: WebSocket, accept_input: bool, pose_period: float):
        await ws.accept()
        hub.loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        hub.queues.add(q)
        sender = asyncio.create_task(_sender(ws, q, pose_period))
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None and accept_input:
                    try:
                        runtime.submit_input(parse_input(msg["bytes"]))
                    except ValueError as e:
                        log.warning("bad input frame: %s", e)
                elif msg.get("text") is not None:
                    await _handle_text(msg["text"])
        except WebSocketDisconnect:
            pass
        finally:
            sender.cancel()
            hub.queues.discard(q)

    @app.websocket("/ws/xr")
    async def ws_xr(ws: WebSocket):
        await _session(ws, accept_input=True, pose_period=0.0)

    @app.websocket("/ws/dash")
    async def ws_dash(ws: WebSocket):
        await _session(ws, accept_input=False, pose_period=1 / 30)

    if WEB_DIST.exists():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    else:
        @app.get("/dashboard/")
        async def no_web():
            return Response("Web client not built. Run: cd web && npm ci && npm run build", media_type="text/plain")

    return app
