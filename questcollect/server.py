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

from questcollect.audit import (CURATION_FIELDS, audit_dir, forget_curation, kept_spans, labelled_success,
                               load_curation, update_curation, verdict)
from questcollect.config import ROOT
from questcollect.learn.train import list_runs
from questcollect.protocol import parse_input
from questcollect.recorder import episode_files, list_episodes
from questcollect.runtime import Runtime

log = logging.getLogger(__name__)
WEB_DIST = ROOT / "web" / "dist"
SEND_PERIOD_S = 0.01
EVAL_PREFIX = "eval/"  # dataset names of policy rollouts: eval/<run>/<time> = <runs_dir>/<run>/eval/<time> (read-only)
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

    @app.get("/api/task")
    async def task():
        return JSONResponse(runtime.task(), headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    async def status():
        return runtime.status()

    def _dataset_dir(dataset: str | None) -> Path:
        """An episode folder by name: the recording folder (record.out_dir) by default, else a sibling folder of it
        (e.g. data/episodes_generated, data/mix_parallel), or a policy evaluation's rollouts (eval/<run>/<time>)."""
        if not dataset or dataset == runtime.cfg.out_dir.name:
            return runtime.cfg.out_dir
        if dataset.startswith(EVAL_PREFIX):
            parts = dataset[len(EVAL_PREFIX):].split("/")
            d = runtime.cfg.runs_dir / parts[0] / "eval" / parts[-1]
            if len(parts) != 2 or any(not p or p.startswith(".") for p in parts) or not d.is_dir():
                raise HTTPException(404, "unknown dataset")
            return d
        name = Path(dataset).name
        d = runtime.cfg.out_dir.parent / name
        if not name or name.startswith(".") or not d.is_dir():
            raise HTTPException(404, "unknown dataset")
        return d

    def _episode_file(file: str, dataset: str | None = None) -> Path:
        d = _dataset_dir(dataset)
        p = d / Path(file).name
        if p.suffix != ".hdf5" or not p.exists():
            raise HTTPException(404, "unknown episode")
        return p

    @app.get("/api/datasets")
    async def datasets():
        """Every episode folder next to the recording folder, recording folder first. Title and description come
        from an optional dataset.json in the folder."""
        def run():
            rec = runtime.cfg.out_dir
            folders = [rec] + sorted(d for d in rec.parent.iterdir()
                                     if d.is_dir() and d != rec and not d.name.startswith("."))
            out = []
            for d in folders:
                files = episode_files(d)
                if not files and d != rec:
                    continue
                info = {}
                if (d / "dataset.json").exists():
                    try:
                        info = json.loads((d / "dataset.json").read_text())
                    except ValueError:
                        pass
                out.append({"name": d.name, "recording": d == rec, "count": len(files), "readonly": False,
                            "title": info.get("title", "Recorded demos" if d == rec else d.name),
                            "description": info.get("description", "")})
            return out + _eval_folders()
        return await asyncio.to_thread(run)

    def _eval_folders() -> list[dict]:
        """Saved policy rollouts (qc eval), newest first: replay a failure and take over where it went wrong."""
        runs = runtime.cfg.runs_dir
        if not runs.is_dir():
            return []
        out = []
        for d in sorted(runs.glob("*/eval/*"), key=lambda d: d.name, reverse=True):
            files = episode_files(d) if d.is_dir() else []
            if not files:
                continue
            run = d.parent.parent.name
            desc = ""
            try:
                s = json.loads((d / "metrics.json").read_text())["summary"]
                stages = " · ".join(f"{k.replace('_', ' ')} {100 * v:.0f}%" for k, v in s.get("stage_rates", {}).items())
                desc = f"Policy rollouts (read-only) · success {100 * s['success_rate']:.0f}% · {stages}"
            except (OSError, ValueError, KeyError, TypeError):
                desc = "Policy rollouts (read-only)"
            out.append({"name": f"{EVAL_PREFIX}{run}/{d.name}", "recording": False, "count": len(files),
                        "readonly": True, "title": f"Rollouts · {run} · {d.name}", "description": desc})
        return out

    def _readonly(dataset: str | None) -> bool:
        return bool(dataset) and dataset.startswith(EVAL_PREFIX)

    @app.get("/api/episodes")
    async def episodes(dataset: str | None = None):
        """Saved episodes with their manual labels (curation.json): success = the label, recorded_success = as saved."""
        folder = _dataset_dir(dataset)
        rows = await asyncio.to_thread(list_episodes, folder)
        curation = load_curation(folder)
        for r in rows:
            c = curation.get(r["file"], {})
            r.update(recorded_success=r["success"], success=bool(c.get("success", r["success"])),
                     tags=c.get("tags", []), note=c.get("note", ""), excluded=bool(c.get("exclude")))
        return rows

    @app.patch("/api/episodes/{file}")
    async def edit_episode(file: str, body: dict, dataset: str | None = None):
        """Change an episode's manual labels: any of exclude, note, tags, success (null = as recorded)."""
        if _readonly(dataset):
            raise HTTPException(403, "policy rollouts are read-only")
        p = _episode_file(file, dataset)
        fields = {k: v for k, v in body.items() if k in CURATION_FIELDS}
        if "note" in fields:
            fields["note"] = str(fields["note"] or "")[:500]
        return update_curation(p.parent, p.name, **fields)

    @app.delete("/api/episodes/{file}")
    async def delete_episode(file: str, dataset: str | None = None):
        if _dataset_dir(dataset) != runtime.cfg.out_dir:  # recorded takes only: derived sets are rebuilt by scripts
            raise HTTPException(403, "episodes can only be deleted from the recording folder")
        p = _episode_file(file)
        if Path((runtime.status().get("replay") or {}).get("path", "")).name == p.name:
            runtime.command("replay_stop")
        p.unlink()
        forget_curation(runtime.cfg.out_dir, p.name)
        return {"deleted": p.name}

    @app.get("/api/audit")
    async def audit(dataset: str | None = None):
        """Per episode: contact shares, idle share, level times, and why a training dataset would leave it out (or
        null)."""
        def run():
            folder = _dataset_dir(dataset)
            audits = audit_dir(folder, cache=not _readonly(dataset))
            curation = load_curation(folder)
            rules = runtime.cfg.curate

            def why(a):  # the same reasons as audit.select
                return verdict(a, rules, curation) or (None if kept_spans(a, rules) else "nothing left after trimming")
            return {name: {"contact_frac": a.contact_frac, "contact_events": a.contact_events,
                           "idle_frac": a.idle_frac, "level_times": a.level_times, "fps": a.fps,
                           "success": labelled_success(a, curation),
                           "verdict": why(a)} for name, a in audits.items()}
        return await asyncio.to_thread(run)

    @app.get("/api/policies")
    async def policies():
        """Trained runs in learn.runs_dir for the Policy tab: checkpoints, inputs, evaluations (no torch needed)."""
        return await asyncio.to_thread(list_runs, runtime.cfg.runs_dir)

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
