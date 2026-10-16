"""Real-time sim thread: input -> teleop -> IK -> physics -> recording -> publish.

The network layer talks to this only through thread-safe methods (submit_input, command,
latest_pose_frame, status, previews) and event listeners.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable

import numpy as np

from questcollect.config import Config
from questcollect.ik.solver import BimanualIK
from questcollect.record.hdf5_io import list_episodes, read_episode
from questcollect.record.recorder import Recorder, RecState
from questcollect.record.render import CameraRenderer
from questcollect.server.protocol import (
    FLAG_ENGAGED_L, FLAG_ENGAGED_R, FLAG_RECORDING, InputFrame, pack_pose_frame,
)
from questcollect.sim.env import BimanualEnv
from questcollect.sim.scene import ARMS
from questcollect.sim.scene_export import body_poses, export_scene
from questcollect.sim.tasks import TASKS
from questcollect.teleop.frames import mat_to_quat_wxyz, quat_wxyz_to_mat, wxyz_to_xyzw
from questcollect.teleop.teleop import Teleop

log = logging.getLogger(__name__)

STATUS_EVERY = 5          # ticks (10 Hz at 50 Hz control)
PREVIEW_EVERY = 6         # ticks (~8 Hz)
PREVIEW_SIZE = (240, 320)
IK_WARN_M = 0.03


def _ms() -> float:
    return time.monotonic() * 1000.0


class Runtime:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dt = 1.0 / cfg.sim.control_hz
        self.teleop = Teleop(cfg.teleop)
        self.placement = self.teleop.placement
        self.ik = BimanualIK(cfg.sim, cfg.ik, self.dt)
        self.recorder = Recorder(cfg, cfg.out_dir)

        self._input: InputFrame | None = None
        self._input_lock = threading.Lock()
        self._input_rate = 0.0
        self._last_input_t: float | None = None
        self._rtt_ms = 0.0
        self._cmds: queue.Queue = queue.Queue()
        self._out_lock = threading.Lock()
        self._pose_frame = b""
        self._pose_seq = 0
        self._status: dict = {}
        self._previews: dict[str, bytes] = {}
        self._previews_wanted_until = 0.0
        self._listeners: list[Callable[[dict], None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tick_count = 0
        self._loop_hz = float(cfg.sim.control_hz)
        self._replay: dict | None = None
        self._preview_renderer: CameraRenderer | None = None
        self._record_renderer: CameraRenderer | None = None
        self._warn = {"arm_collision": False, "table_collision": False, "ik_saturated": False}
        self.scene_version = 0
        self._load_task(cfg.task)

    # -- public, thread-safe ---------------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="sim", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.recorder.close()

    def add_listener(self, fn: Callable[[dict], None]) -> None:
        self._listeners.append(fn)

    def submit_input(self, frame: InputFrame) -> None:
        now = time.monotonic()
        with self._input_lock:
            if self._last_input_t is not None:
                inst = 1.0 / max(now - self._last_input_t, 1e-4)
                self._input_rate = 0.9 * self._input_rate + 0.1 * inst
            self._last_input_t = now
            frame.recv_time = now
            if frame.t_echo > 0:
                self._rtt_ms = 0.8 * self._rtt_ms + 0.2 * (_ms() - frame.t_echo)
            self._input = frame

    def command(self, name: str, **kwargs) -> None:
        self._cmds.put((name, kwargs))

    def latest_pose_frame(self) -> tuple[int, bytes]:
        with self._out_lock:
            return self._pose_seq, self._pose_frame

    def status(self) -> dict:
        with self._out_lock:
            return dict(self._status)

    def previews(self) -> dict[str, bytes]:
        self._previews_wanted_until = time.monotonic() + 2.0
        with self._out_lock:
            return dict(self._previews)

    def scene(self) -> tuple[dict, bytes]:
        return self._scene_json, self._scene_blob

    def episodes(self) -> list[dict]:
        return list_episodes(self.cfg.out_dir)

    # -- internals ---------------------------------------------------------------------------------
    def _emit(self, event: dict) -> None:
        for fn in self._listeners:
            try:
                fn(event)
            except Exception:  # noqa: BLE001 - a broken listener must not kill the sim loop
                log.exception("listener failed")

    def _load_task(self, name: str) -> None:
        self.env = BimanualEnv(self.cfg, name)
        self._scene_json, self._scene_blob = export_scene(self.env.model)
        self.scene_version += 1
        self._scene_json["version"] = self.scene_version
        m = self.env.model
        self._arm_bodies = {
            a: {b for b in range(m.nbody) if m.body(b).name.startswith(f"{a}_")} for a in ARMS
        }
        self._finger_bodies = {b for b in range(m.nbody) if "finger" in m.body(b).name}
        self._table_body = m.body("table").id
        for r in (self._preview_renderer, self._record_renderer):
            if r is not None:
                r.close()
        self._preview_renderer = self._record_renderer = None
        self._reset_scene()

    def _reset_scene(self, seed: int | None = None) -> None:
        obs = self.env.reset(seed)
        self.ik.reset(obs["joint_pos"])
        self.teleop.reset({a: self._ik_pose(a) for a in ARMS})

    def _ik_pose(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        p, q = self.ik.tcp_pose(arm)
        return p, quat_wxyz_to_mat(q)

    def _ensure_preview_renderer(self) -> None:
        # GL context + mesh upload takes ~0.5 s, so do it up front rather than mid-teleop.
        if self._preview_renderer is None:
            try:
                self._preview_renderer = CameraRenderer(self.env.model, *PREVIEW_SIZE, list(self.cfg.record.cameras))
            except Exception:  # noqa: BLE001 - previews are optional (no GL available)
                log.exception("camera previews disabled: offscreen rendering failed")

    def _run(self) -> None:
        self._ensure_preview_renderer()
        next_t = time.perf_counter()
        window_start, window_ticks = next_t, 0
        while not self._stop.is_set():
            now_pc = time.perf_counter()
            if now_pc < next_t:
                time.sleep(min(next_t - now_pc, 0.005))
                continue
            window_ticks += 1
            if now_pc - window_start >= 1.0:
                self._loop_hz = window_ticks / (now_pc - window_start)
                window_start, window_ticks = now_pc, 0
            try:
                self._tick(time.monotonic())
            except Exception:  # noqa: BLE001
                log.exception("sim tick failed")
            next_t += self.dt
            if time.perf_counter() - next_t > 0.25:  # fell far behind (e.g. debugger): resync
                next_t = time.perf_counter()

    def _process_commands(self, now: float) -> None:
        while True:
            try:
                name, kw = self._cmds.get_nowait()
            except queue.Empty:
                return
            log.info("command %s %s", name, kw)
            self._handle(name, kw, now)

    def _handle(self, name: str, kw: dict, now: float) -> None:
        rec = self.recorder
        if name == "record_toggle":
            was = rec.state
            rec.toggle(now)
            if was == RecState.IDLE and rec.state == RecState.COUNTDOWN:
                self._emit({"type": "toast", "text": "Get ready…", "kind": "info"})
        elif name == "save":
            path = rec.save(self.env, self.teleop.mode, kw.get("success"))
            if path:
                self._emit({"type": "saved", "path": path})
                self._emit({"type": "toast", "text": f"Saved {path.rsplit('/', 1)[-1]}", "kind": "success"})
                self._haptic_both(0.8, 80)
                self._reset_scene()
        elif name == "discard":
            if rec.discard():
                self._emit({"type": "toast", "text": "Take discarded", "kind": "warn"})
                self._reset_scene()
        elif name == "reset":
            rec.discard()
            self._replay = None
            self._reset_scene(kw.get("seed"))
        elif name == "set_task":
            task = kw.get("task")
            if task in TASKS and task != self.env.task.name:
                rec.discard()
                self._replay = None
                self._load_task(task)
                self._ensure_preview_renderer()
                self._emit({"type": "scene", "version": self.scene_version})
                self._emit({"type": "toast", "text": f"Task: {task}", "kind": "info"})
        elif name == "set_mode":
            self.teleop.set_mode(kw.get("mode", "controllers"))
        elif name == "set_scale":
            self.teleop.set_scale(float(kw.get("scale", 1.0)))
        elif name == "recenter":
            with self._input_lock:
                frame = self._input
            if frame is not None:
                self.placement.recenter(frame.head[:3], frame.head[3:])
                for st in self.teleop.arms.values():
                    st.clutch.release()
        elif name == "set_operator":
            self.cfg.record.operator = str(kw.get("operator", "operator"))[:64]
        elif name == "set_auto_stop":
            rec.auto_stop = bool(kw.get("enabled", True))
        elif name == "replay":
            self._start_replay(kw["path"])
        elif name == "replay_stop":
            self._replay = None
            self._reset_scene()

    def _start_replay(self, path: str) -> None:
        if self.recorder.state != RecState.IDLE:
            return
        meta, data = read_episode(path, keys=["sim/state"])
        if meta.get("task") != self.env.task.name:
            self._load_task(meta["task"])
            self._emit({"type": "scene", "version": self.scene_version})
        self._replay = {"path": path, "states": data["sim/state"], "i": 0, "success": meta.get("success")}
        self._emit({"type": "toast", "text": f"Replaying {path.rsplit('/', 1)[-1]}", "kind": "info"})

    def _haptic_both(self, intensity: float, ms: int) -> None:
        for a in ARMS:
            self._emit({"type": "haptic", "arm": a, "intensity": intensity, "ms": ms})

    def _tick(self, now: float) -> None:
        self._process_commands(now)
        self._tick_count += 1
        env = self.env

        if self._replay is not None:
            r = self._replay
            env.set_state(r["states"][r["i"]])
            r["i"] += 1
            if r["i"] >= len(r["states"]):
                self._replay = None
                self._emit({"type": "toast", "text": "Replay finished", "kind": "info"})
                self._reset_scene()
            self._publish(now, None)
            return

        with self._input_lock:
            frame = self._input
        out = self.teleop.update(frame, now, self.dt)
        for cmd in out.commands:
            self._handle(cmd, {}, now)
        for arm, inten, ms in out.haptics:
            self._emit({"type": "haptic", "arm": arm, "intensity": inten, "ms": ms})

        for a in ARMS:
            p, R = out.targets[a]
            self.ik.set_target(a, p, mat_to_quat_wxyz(R))
        q_target = self.ik.solve()

        prev_state = self.recorder.state
        self.recorder.tick(now, env)
        if prev_state == RecState.COUNTDOWN and self.recorder.recording:
            self._haptic_both(1.0, 120)
            self._emit({"type": "toast", "text": "Recording", "kind": "rec"})

        recording = self.recorder.recording
        if recording:
            obs = env.observe()
            state = env.get_state()
        env.apply_action(q_target, out.grippers)
        success = env.success()

        if recording:
            step = self._build_step(obs, state, out, frame, q_target, success)
            if self.recorder.append(now, step, success):
                self._emit({"type": "toast", "text": "Success! Save (X) or discard (B)", "kind": "success"})
                self._haptic_both(0.5, 60)

        env.step()
        self._check_warnings()
        self._publish(now, out, success)

    def _build_step(self, obs: dict, state: np.ndarray, out, frame: InputFrame | None, q_target,
                    success: bool) -> dict:
        env = self.env
        ee_act = []
        for a in ARMS:
            p, R = out.targets[a]
            ee_act.append(np.concatenate([p, mat_to_quat_wxyz(R)]))
        g = out.grippers
        teleop = {"head": np.zeros(7) if frame is None else frame.head,
                  "engaged": np.array([out.engaged[a] for a in ARMS]),
                  "tracked": np.array([out.tracked[a] for a in ARMS]),
                  "rtt_ms": self._rtt_ms}
        for side, a in (("left", "l"), ("right", "r")):
            h = frame.hand(a) if frame is not None else None
            teleop[side] = {
                "source": 0 if h is None else h.source,
                "pose": np.zeros(7) if h is None else h.pose,
                "buttons": np.zeros(8) if h is None else h.buttons,
                "joints": np.zeros((25, 7)) if h is None else h.joints,
                "pinch": np.zeros(2) if h is None else h.pinch,
            }
        step = {
            "time": env.data.time,
            "wall_time": time.monotonic(),
            "obs": {k: obs[k] for k in ("joint_pos", "joint_vel", "gripper_width", "ee_pose_left",
                                        "ee_pose_right", "objects")},
            "action": {
                "joint_pos": np.concatenate([q_target[:7], g[:1], q_target[7:], g[1:]]),
                "ee_pose": np.concatenate([ee_act[0], ee_act[1], g]),
                "ctrl": env.data.ctrl.copy(),
            },
            "sim": {"state": state},
            "success": success,
            "teleop": teleop,
        }
        if self.cfg.record.live_images:
            if self._record_renderer is None:
                self._record_renderer = CameraRenderer(env.model, self.cfg.record.image_height,
                                                       self.cfg.record.image_width, self.cfg.record.cameras)
            step["images"] = self._record_renderer.render(env.data)
        return step

    def _check_warnings(self) -> None:
        d = self.env.data
        left, right = self._arm_bodies["l"], self._arm_bodies["r"]
        geom_body = self.env.model.geom_bodyid
        arm_hit = table_hit = False
        for x, y in zip(geom_body[d.contact.geom1], geom_body[d.contact.geom2]):
            if (x in left and y in right) or (x in right and y in left):
                arm_hit |= not (x in self._finger_bodies and y in self._finger_bodies)
            other = y if x == self._table_body else x if y == self._table_body else None
            table_hit |= other in left or other in right
        ik_sat = any(e[0] > IK_WARN_M for e in self.ik.errors().values())
        new = {"arm_collision": arm_hit, "table_collision": table_hit, "ik_saturated": ik_sat}
        for k, v in new.items():
            if v and not self._warn[k]:
                self._haptic_both(0.3, 30)
        self._warn = new

    def _publish(self, now: float, out, success: bool = False) -> None:
        env = self.env
        ghosts = np.zeros((2, 7))
        grippers = np.ones(2)
        flags = 0
        if out is not None:
            for i, a in enumerate(ARMS):
                p, R = out.targets[a]
                ghosts[i, :3] = p
                ghosts[i, 3:] = wxyz_to_xyzw(mat_to_quat_wxyz(R))
            grippers = out.grippers
            flags |= FLAG_ENGAGED_L if out.engaged["l"] else 0
            flags |= FLAG_ENGAGED_R if out.engaged["r"] else 0
        if self.recorder.recording:
            flags |= FLAG_RECORDING
        with self._input_lock:
            t_echo = self._input.t_client if self._input is not None else 0.0
        frame = pack_pose_frame(self._pose_seq + 1, _ms(), t_echo, body_poses(env.data), ghosts, grippers, flags)
        with self._out_lock:
            self._pose_seq += 1
            self._pose_frame = frame

        if self._tick_count % STATUS_EVERY == 0:
            self._update_status(now, out, success)
        if (self._tick_count % PREVIEW_EVERY == 0 and now < self._previews_wanted_until
                and self._preview_renderer is not None):
            jpegs = self._preview_renderer.render_jpeg(env.data)
            with self._out_lock:
                self._previews = jpegs

    def _update_status(self, now: float, out, success: bool) -> None:
        rec = self.recorder
        with self._input_lock:
            frame = self._input
            input_age = (now - frame.recv_time) if frame is not None else None
            rate = self._input_rate if input_age is not None and input_age < 1.0 else 0.0
        errs = self.ik.errors()
        status = {
            "type": "status",
            "task": self.env.task.name,
            "instruction": self.env.task.instruction,
            "tasks": sorted(TASKS),
            "mode": self.teleop.mode,
            "scale": self.teleop.scale,
            "rec_state": rec.state.value,
            "countdown": round(rec.countdown_left(now), 2),
            "elapsed": round(rec.elapsed(now), 2),
            "steps": len(rec.steps),
            "saved_count": rec.saved_count,
            "last_saved": rec.last_saved,
            "auto_stop": rec.auto_stop,
            "success": bool(success),
            "success_seen": rec.success_seen,
            "operator": self.cfg.record.operator,
            "engaged": {a: bool(out.engaged[a]) for a in ARMS} if out else {a: False for a in ARMS},
            "tracked": {a: bool(out.tracked[a]) for a in ARMS} if out else {a: False for a in ARMS},
            "mode_mismatch": bool(out.mode_mismatch) if out else False,
            "sources": {a: (frame.hand(a).source if frame is not None and input_age < 1.0 else 0) for a in ARMS},
            "grippers": out.grippers.round(3).tolist() if out else [1.0, 1.0],
            "ik_error_mm": {a: round(e[0] * 1000, 1) for a, e in errs.items()},
            "ee_pos": {a: self.env.data.site_xpos[sid].round(4).tolist() for a, sid in zip(ARMS, self.env.tcp_sites)},
            "warnings": dict(self._warn),
            "input_hz": round(rate, 1),
            "rtt_ms": round(self._rtt_ms, 1) if rate > 0 else None,
            "loop_hz": round(self._loop_hz, 1),
            "connected": rate > 0,
            "sim_time": round(self.env.data.time, 2),
            "scene_version": self.scene_version,
            "placement": self.placement.to_dict()["matrix"],
            "replay": None if self._replay is None else {
                "path": self._replay["path"], "i": self._replay["i"], "n": len(self._replay["states"])},
        }
        with self._out_lock:
            self._status = status

