"""Real-time sim thread: input -> teleop -> IK (or policy) -> safety filter -> physics -> recording
-> publish.

The network layer talks to this only through thread-safe methods (submit_input, command,
latest_pose_frame, status, previews) and event listeners.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from questcollect.audit import CONTACT_KINDS, audit_arrays, kept_spans, verdict
from questcollect.config import Config
from questcollect.frames import mat_to_quat_wxyz, quat_wxyz_to_mat
from questcollect.ik import BimanualIK
from questcollect.protocol import FLAG_ENGAGED_L, FLAG_ENGAGED_R, FLAG_RECORDING, InputFrame, pack_pose_frame
from questcollect.recorder import (SOURCE_OPERATOR, SOURCE_POLICY, Recorder, RecState, make_step,
                                   read_episode, teleop_fields)
from questcollect.safety import SafetyFilter
from questcollect.scene_export import body_poses, export_scene
from questcollect.sim import ARMS, BRICKS, INSTRUCTION, LEVELS, TASK, BimanualEnv, CameraRenderer, task_spec
from questcollect.teleop import Teleop

log = logging.getLogger(__name__)

STATUS_EVERY = 5          # ticks (10 Hz at 50 Hz control)
PREVIEW_EVERY = 6         # ticks (~8 Hz)
PREVIEW_SIZE = (240, 320)
IK_WARN_M = 0.03
DART_TAU_S = 0.5           # DART drift correlation time: slow drifts the operator sees and corrects
DART_SMOOTH_S = 0.2        # ... low-passed once more, so the arm drifts instead of trembling
GRIP_CATCH = 0.15          # a latched gripper follows its Trigger again once the Trigger is this close to it
BLOCKED_RAD = 0.05          # the safety filter changed a joint target this much: warn the operator
CONTROL_MODES = ("teleop", "policy", "shared")  # shared: the policy drives until the operator takes control
REPLAY_SPEED_RANGE = (0.1, 8.0)  # slowest / fastest dashboard replay speed
# Task haptics: (intensity, ms, repeat, gap_ms). Distinct patterns so the operator can tell them apart.
# Most come with a sound cue of the same name (web/src/xr/sounds.ts), panned to that arm's side.
HAPTIC_GRASP = (0.5, 40, 1, 0)        # fingers closed on a brick
HAPTIC_HOLD_TICK = (0.15, 20, 1, 0)   # every HOLD_TICK_S while holding
HAPTIC_RELEASE = (0.2, 20, 1, 0)
HAPTIC_CLICK = (0.8, 15, 1, 0)        # a brick clicked onto studs (anywhere but its tower spot)
HAPTIC_UNCLICK = (0.4, 30, 1, 0)      # a clicked brick was pulled off
HAPTIC_PLACED = (0.9, 60, 2, 70)      # a brick clicked into its correct spot on the tower
HAPTIC_COMPLETE = (1.0, 120, 3, 80)   # tower finished and standing
HOLD_TICK_S = 0.25
HAPTIC_QUIET_S = 0.4                  # no hold ticks right after a pattern (a new pulse cuts off the old one)
WARN_SOUND_GAP_S = 1.0                # at most one sound per warning kind and arm per second


def _ms() -> float:
    return time.monotonic() * 1000.0


class Runtime:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dt = 1.0 / cfg.sim.control_hz
        self.env = BimanualEnv(cfg)
        self.teleop = Teleop(cfg.teleop)
        self.placement = self.teleop.placement
        self.ik = BimanualIK(cfg.sim, cfg.ik, self.dt)
        self.recorder = Recorder(cfg, cfg.out_dir)
        self._scene_json, self._scene_blob = export_scene(self.env.model)
        # Who drives: the operator (teleop), a trained policy, or the policy until the operator takes over (shared).
        # Either way the safety filter keeps the arms off each other and the table.
        self.safety = SafetyFilter(cfg.sim, cfg.ik, cfg.safety, self.dt)
        self.policy = None  # learn.policy.PolicyRunner, see load_policy
        self.control_mode = "teleop"
        self._driver = "operator"
        self._safety_rad = 0.0
        self._noise = np.random.default_rng()
        self._dart = np.zeros(14)  # current DART offset on the executed joint targets (rad)
        self._dart_ou = np.zeros(14)  # ... before smoothing
        # The smoothing shrinks the drift; scaled so its std is still record.action_noise.
        self._dart_gain = cfg.record.action_noise * np.sqrt(1 + DART_SMOOTH_S / DART_TAU_S)
        if cfg.control.policy:
            self.load_policy(cfg.control.policy)
            self.control_mode = cfg.control.mode
        elif cfg.control.mode != "teleop":
            log.warning("control.mode %s needs a policy (qc serve --policy outputs/<run>); using teleop",
                        cfg.control.mode)
        _ = self.env.scene_hash  # compiles the scene once more (~0.2 s): now, not on the first save mid-session

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
        self._preview_renderer: CameraRenderer | None = None
        self._listeners: list[Callable[[dict], None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tick_count = 0
        self._loop_hz = float(cfg.sim.control_hz)
        self._replay: dict | None = None
        self._replay_speed, self._replay_loop = 1.0, False  # kept across replays (dashboard settings)
        self._warn = {"arm_collision": False, "table_collision": False, "ik_saturated": False, "blocked": False}
        self._reach = {a: False for a in ARMS}
        self._engaged = {a: False for a in ARMS}
        self._policy_grip = np.ones(len(ARMS))      # the policy's last gripper command
        self._grip_latch: list[float | None] = [None] * len(ARMS)  # held over a takeover until the Trigger matches
        self._warn_sound_at: dict[tuple[str, str | None], float] = {}
        self._base_in_area = False
        self._held: dict[str, int | None] = {a: None for a in ARMS}
        self._last_holder: list[str | None] = [None] * len(BRICKS)
        self._progress = 0
        self._complete = False
        self._hold_tick_at = {a: 0.0 for a in ARMS}
        self._quiet_until = {a: 0.0 for a in ARMS}
        self._review: tuple[int, dict] | None = None  # (steps, summary) of the take under review
        self._reset_scene()

    def load_policy(self, run: str | Path) -> None:
        """Load a trained policy (`qc serve --policy outputs/<run>`). Imports torch, so only when asked."""
        from questcollect.learn.policy import PolicyRunner

        self.policy = PolicyRunner(Path(run), self.cfg, n_action_steps=self.cfg.control.n_action_steps)

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

    def task(self) -> dict:
        return task_spec()

    # -- internals ---------------------------------------------------------------------------------
    def _emit(self, event: dict) -> None:
        for fn in self._listeners:
            try:
                fn(event)
            except Exception:  # noqa: BLE001 - a broken listener must not kill the sim loop
                log.exception("listener failed")

    def _toast(self, text: str, kind: str = "info") -> None:
        self._emit({"type": "toast", "text": text, "kind": kind})

    def _haptic(self, arm: str, intensity: float, ms: int, repeat: int = 1, gap_ms: int = 0) -> None:
        self._emit({"type": "haptic", "arm": arm, "intensity": intensity, "ms": ms, "repeat": repeat, "gap_ms": gap_ms})

    def _haptic_both(self, intensity: float, ms: int, repeat: int = 1, gap_ms: int = 0) -> None:
        for a in ARMS:
            self._haptic(a, intensity, ms, repeat, gap_ms)

    def _sound(self, name: str, arm: str | None = None) -> None:
        """Sound cue for the headset (arm: which side to pan it to; None = centre)."""
        self._emit({"type": "sound", "name": name, "arm": arm})

    def _warn_sound(self, name: str, arm: str | None, now: float) -> None:
        if now >= self._warn_sound_at.get((name, arm), 0.0):
            self._sound(name, arm)
            self._warn_sound_at[(name, arm)] = now + WARN_SOUND_GAP_S

    def _reset_scene(self, seed: int | None = None) -> None:
        obs = self.env.reset(seed)
        self._held = {a: None for a in ARMS}
        self._last_holder = [None] * len(BRICKS)
        self._progress = 0
        self._complete = False
        self._base_in_area = False
        self.ik.reset(obs["joint_pos"])
        self.teleop.reset({a: self._ik_pose(a) for a in ARMS})
        self.safety.reset(obs["joint_pos"])
        self._driver = "operator"  # a policy starts a fresh chunk on its first tick

    def _ik_pose(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        p, q = self.ik.tcp_pose(arm)
        return p, quat_wxyz_to_mat(q)

    def _run(self) -> None:
        # GL context + mesh upload takes ~0.5 s, so do it up front rather than mid-teleop.
        try:
            self._preview_renderer = CameraRenderer(self.env.model, *PREVIEW_SIZE, list(self.cfg.record.cameras))
        except Exception:  # noqa: BLE001 - previews are optional (no GL available)
            log.exception("camera previews disabled: offscreen rendering failed")
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
            if self._replay is not None and rec.state == RecState.IDLE:
                self._replay = None
                self._reset_scene()
            was = rec.state
            rec.toggle(now)
            if was == RecState.IDLE and rec.state == RecState.COUNTDOWN:
                self._toast("Get ready…")
        elif name == "save":
            path = rec.save(self.env, self.teleop.mode, kw.get("success"), source=self.control_mode)
            if path:
                self._emit({"type": "saved", "path": path})
                self._toast(f"Saved {path.rsplit('/', 1)[-1]}", "success")
                self._haptic_both(0.8, 80)
                self._reset_scene()
        elif name == "discard":
            if rec.discard():
                self._toast("Take discarded", "warn")
                self._reset_scene()
        elif name == "reset":
            rec.discard()
            self._replay = None
            self._reset_scene(kw.get("seed"))
        elif name == "set_mode":
            self.teleop.set_mode(kw.get("mode", "controllers"))
        elif name == "set_scale":
            self.teleop.set_scale(float(kw.get("scale", 1.0)))
        elif name == "recenter":
            with self._input_lock:
                frame = self._input
            if frame is not None:
                self.placement.recenter(frame.head[:3], frame.head[3:])
                self.teleop.release_all()
        elif name == "set_operator":
            self.cfg.record.operator = str(kw.get("operator", "operator"))[:64]
        elif name == "set_auto_stop":
            rec.auto_stop = bool(kw.get("enabled", True))
        elif name == "replay":
            if rec.state == RecState.IDLE:
                _, data = read_episode(kw["path"], keys=["sim/state"])
                states = data["sim/state"]
                self._replay = {"path": kw["path"], "states": states, "paused": bool(kw.get("paused", False)),
                                "i": int(np.clip(kw.get("i", 0), 0, len(states) - 1)), "acc": 0.0}
        elif name == "replay_pause":
            if self._replay is not None:
                r = self._replay
                r["paused"] = bool(kw.get("paused", not r["paused"]))
                if not r["paused"] and r["i"] >= len(r["states"]) - 1:
                    r["i"] = 0  # play again from the start
        elif name == "replay_options":
            if "speed" in kw:
                self._replay_speed = float(np.clip(kw["speed"], REPLAY_SPEED_RANGE[0], REPLAY_SPEED_RANGE[1]))
            if "loop" in kw:
                self._replay_loop = bool(kw["loop"])
        elif name == "replay_seek":
            if self._replay is not None:
                self._replay["i"] = int(np.clip(kw.get("i", 0), 0, len(self._replay["states"]) - 1))
        elif name == "replay_stop":
            self._replay = None
            self._reset_scene()
        elif name == "set_control_mode":
            mode = kw.get("mode", "teleop")
            if mode != "teleop" and self.policy is None:
                self._toast("No policy loaded: start with qc serve --policy outputs/<run>", "warn")
            elif mode in CONTROL_MODES and mode != self.control_mode:
                self.control_mode = mode
                self.teleop.release_all()
                self._toast({"teleop": "You drive", "policy": "Policy drives",
                             "shared": "Policy drives: take over any time with Grip"}[mode], "info")

    def _tick(self, now: float) -> None:
        self._process_commands(now)
        self._tick_count += 1
        env = self.env

        if self._replay is not None:
            r = self._replay
            env.set_state(r["states"][r["i"]])
            if not r["paused"]:
                r["acc"] += self._replay_speed  # recorded steps per tick: speed 1 = real time
                steps, r["acc"] = int(r["acc"]), r["acc"] % 1.0
                last = len(r["states"]) - 1
                if r["i"] + steps <= last:
                    r["i"] += steps
                elif self._replay_loop:
                    r["i"] = 0
                else:
                    r["i"], r["paused"] = last, True  # hold the last frame until replay_stop
            self._publish(now, None)
            return

        with self._input_lock:
            frame = self._input
        out = self.teleop.update(frame, now, self.dt)
        for cmd in out.commands:
            self._handle(cmd, {}, now)
        for arm, inten, ms in out.haptics:
            self._haptic(arm, inten, ms)
        for a in ARMS:
            if out.engaged[a] != self._engaged[a]:
                self._sound("engage" if out.engaged[a] else "disengage", a)
        self._engaged = dict(out.engaged)

        q_measured = env.data.qpos[env.arm_qadr].copy()
        driver = self._who_drives(out)
        if driver == "policy":
            if self._driver != "policy":  # (re)starting: a fresh chunk from the current state
                self.policy.reset(env)
            q_wanted, grippers = self.policy.act(env)
            self._policy_grip = np.array(grippers, float)
            self._grip_latch = [None, None]
        else:
            for a in ARMS:
                p, R = out.targets[a]
                self.ik.set_target(a, p, mat_to_quat_wxyz(R))
            self.ik.follow(q_measured)
            q_wanted, grippers = self.ik.solve(), np.array(out.grippers, float)
            if self._driver == "policy":  # takeover: each gripper keeps the policy's command (a held brick stays held)
                self._grip_latch = [float(g) for g in self._policy_grip]
            for i in range(len(ARMS)):  # ... until that controller's Trigger is brought to the same opening
                if self._grip_latch[i] is not None:
                    if abs(grippers[i] - self._grip_latch[i]) < GRIP_CATCH:
                        self._grip_latch[i] = None
                    else:
                        grippers[i] = self._grip_latch[i]

        prev_state = self.recorder.state
        self.recorder.tick(now)
        if prev_state == RecState.COUNTDOWN and self.recorder.recording:
            self._haptic_both(1.0, 120)
            self._toast("Recording", "rec")

        recording = self.recorder.recording
        if recording:
            obs = env.observe()
            state = env.get_state()
            contacts = env.contact_flags()
        # DART: execute a slightly perturbed target but record the operator's clean one, so the data shows how the
        # operator recovers from small errors (Laskey et al., 2017). The offset is a slow random drift: an
        # Ornstein-Uhlenbeck process (correlation DART_TAU_S) smoothed once more (DART_SMOOTH_S), because a raw OU
        # path is rough at the 20 ms scale and the arm visibly trembles (measured 0.4 mm per tick). Only on arms the
        # operator is moving, while recording; otherwise it decays to zero, never jumps.
        a, b = self.dt / DART_TAU_S, self.dt / DART_SMOOTH_S
        on = np.repeat([recording and driver == "operator" and out.engaged[arm] for arm in ARMS], 7)
        self._dart_ou += -a * self._dart_ou + on * self._dart_gain * np.sqrt(2 * a) * self._noise.standard_normal(14)
        self._dart += b * (self._dart_ou - self._dart)
        # The drift goes in before the safety filter, so it can't push a hand onto the table or the other arm;
        # the recorded action is what was executed minus the drift.
        q_exec, self._safety_rad = self.safety.filter(q_wanted + self._dart, q_measured, env.data.qpos[env.finger_qadr])
        q_target = q_exec - self._dart
        ee = self.safety.tcp_poses(q_target)
        if driver == "policy":
            # Keep the operator side where the arms are, so taking control (Grip) never makes them jump.
            self.ik.reset(q_target)
            self.teleop.sync_hold({a: self._ik_pose(a) for a in ARMS})
        self._driver = driver
        env.apply_action(q_exec, grippers)
        success = env.success()

        if recording:
            teleop = teleop_fields(frame, [out.engaged[a] for a in ARMS], [out.tracked[a] for a in ARMS],
                                   self._rtt_ms)
            source = SOURCE_POLICY if driver == "policy" else SOURCE_OPERATOR
            step = make_step(env, obs, state, q_target, grippers, ee, success, self._progress, contacts, teleop,
                             source)
            if self.recorder.append(now, step, success):
                self._toast("Success! Save (X) or discard (B)", "success")

        env.step()
        self._task_feedback(now, success)
        self._check_warnings(now)
        self._publish(now, out, success)

    def _who_drives(self, out) -> str:
        if self.policy is None or self.control_mode == "teleop":
            return "operator"
        if self.control_mode == "shared" and any(out.engaged.values()):
            return "operator"  # DAgger: the operator took over; the policy resumes when every arm is released
        return "policy"

    def _pattern(self, arm: str | None, pattern: tuple, now: float, sound: str | None = None) -> None:
        intensity, ms, repeat, gap = pattern
        for a in ARMS if arm is None else (arm,):
            self._haptic(a, intensity, ms, repeat, gap)
            self._quiet_until[a] = now + HAPTIC_QUIET_S
        if sound:
            self._sound(sound, arm)

    def _task_feedback(self, now: float, success: bool) -> None:
        """Haptics for grasp / hold / release, clicks, each correctly placed brick, and the finished tower."""
        held = self.env.held()
        for a in ARMS:
            b, was = held[a], self._held[a]
            if b is not None:
                self._last_holder[b] = a
            if b is not None and was is None:
                self._pattern(a, HAPTIC_GRASP, now, "grasp")
                self._hold_tick_at[a] = now + HOLD_TICK_S
            elif b is None and was is not None:
                self._pattern(a, HAPTIC_RELEASE, now, "release")
            elif b is not None and now >= self._hold_tick_at[a]:
                if now >= self._quiet_until[a]:
                    self._haptic(a, *HAPTIC_HOLD_TICK)
                self._hold_tick_at[a] = now + HOLD_TICK_S
        self._held = held

        progress = self.env.progress()
        placed = range(self._progress + 1, progress + 1)  # a knocked-down level can fire again later
        for level in placed:
            self._pattern(self._last_holder[level], HAPTIC_PLACED, now, "placed")
            self._toast(f"{BRICKS[level].label} placed ({level + 1}/{len(BRICKS)})", "success")  # base = 1
        self._progress = progress
        for kind, _lower, upper in self.env.clutch_events:
            if kind == "lock" and upper not in placed:
                self._pattern(self._last_holder[upper], HAPTIC_CLICK, now, "click")
            elif kind == "unlock":
                self._pattern(self._last_holder[upper], HAPTIC_UNCLICK, now, "unclick")

        # Base brick set down (not held) inside the build area: a soft confirmation, once per placement.
        base_in = self.env.in_area() and 0 not in held.values()
        if base_in and not self._base_in_area:
            self._sound("area", self._last_holder[0])
            self._toast(f"{BRICKS[0].label} is in the build area", "info")
        self._base_in_area = base_in

        if success and not self._complete:
            self._pattern(None, HAPTIC_COMPLETE, now, "complete")
        self._complete = success

    def _check_warnings(self, now: float) -> None:
        link_table, arm_hit, tip_table = (bool(f) for f in self.env.contact_flags())
        table_hit = link_table or tip_table
        reach = {a: e[0] > IK_WARN_M for a, e in self.ik.errors().items()}
        blocked = self._safety_rad > BLOCKED_RAD  # the operator pushes into the table or the other arm
        new = {"arm_collision": arm_hit, "table_collision": table_hit, "ik_saturated": any(reach.values()),
               "blocked": blocked}
        for k, v in new.items():
            if v and not self._warn[k]:
                self._haptic_both(0.3, 30)
        if arm_hit and not self._warn["arm_collision"]:
            self._warn_sound("collision", None, now)
        if table_hit and not self._warn["table_collision"]:
            self._warn_sound("table", None, now)
        if blocked and not self._warn["blocked"]:
            self._warn_sound("collision", None, now)
        for a in ARMS:
            if reach[a] and not self._reach[a]:
                self._warn_sound("out_of_reach", a, now)
        self._warn, self._reach = new, reach

    def _publish(self, now: float, out, success: bool = False) -> None:
        env = self.env
        flags = 0
        if out is not None:
            flags |= FLAG_ENGAGED_L if out.engaged["l"] else 0
            flags |= FLAG_ENGAGED_R if out.engaged["r"] else 0
        if self.recorder.recording:
            flags |= FLAG_RECORDING
        frame = pack_pose_frame(self._pose_seq + 1, _ms(), body_poses(env.data), flags)
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

    def _review_summary(self) -> dict | None:
        """Quality of the take under review, judged by the same rules `qc audit` and export use, so the operator
        can discard a take that training would not use anyway (computed once per take)."""
        rec = self.recorder
        if rec.state != RecState.REVIEW or not rec.steps:
            self._review = None
            return None
        if self._review is None or self._review[0] != len(rec.steps):
            steps = rec.steps
            data = {"obs/joint_vel": np.array([s["obs"]["joint_vel"] for s in steps]),
                    "teleop/engaged": np.array([s["teleop"]["engaged"] for s in steps]),
                    "progress": np.array([s["progress"] for s in steps]),
                    "contacts": np.array([s["contacts"] for s in steps]),
                    "control/source": np.array([s["control"]["source"] for s in steps])}
            a = audit_arrays("take", self.cfg.sim.control_hz, rec.success_seen, data)
            why = verdict(a, self.cfg.curate, {})
            used = 0 if why else sum(e - s for s, e in kept_spans(a, self.cfg.curate))
            self._review = (len(steps), {
                "duration_s": round(a.duration_s, 1), "success": a.success, "idle_pct": round(100 * a.idle_frac),
                "events": {k: a.contact_events[k] for k in CONTACT_KINDS},
                "rejected": why, "used_s": round(used / a.fps, 1)})
        return self._review[1]

    def _update_status(self, now: float, out, success: bool) -> None:
        rec = self.recorder
        with self._input_lock:
            frame = self._input
            input_age = (now - frame.recv_time) if frame is not None else None
            rate = self._input_rate if input_age is not None and input_age < 1.0 else 0.0
        errs = self.ik.errors()
        status = {
            "type": "status",
            "task": TASK,
            "instruction": INSTRUCTION,
            "mode": self.teleop.mode,
            "scale": self.teleop.scale,
            "rec_state": rec.state.value,
            "countdown": round(rec.countdown_left(now), 2),
            "elapsed": round(rec.elapsed(now), 2),
            "steps": len(rec.steps),
            "saved_count": rec.saved_count,
            "auto_stop": rec.auto_stop,
            "success": bool(success),
            "success_seen": rec.success_seen,
            "progress": self._progress,
            "levels": LEVELS,
            "in_area": self.env.in_area(),
            "clicked": len(self.env.locked_pairs()),
            "held": {a: (None if b is None else BRICKS[b].name) for a, b in self._held.items()},
            "operator": self.cfg.record.operator,
            "engaged": {a: bool(out.engaged[a]) for a in ARMS} if out else {a: False for a in ARMS},
            "tracked": {a: bool(out.tracked[a]) for a in ARMS} if out else {a: False for a in ARMS},
            "mode_mismatch": bool(out.mode_mismatch) if out else False,
            "sources": {a: (frame.hand(a).source if frame is not None and input_age < 1.0 else 0) for a in ARMS},
            "grippers": out.grippers.round(3).tolist() if out else [1.0, 1.0],
            "ik_error_mm": {a: round(e[0] * 1000, 1) for a, e in errs.items()},
            "out_of_reach": {a: bool(e[0] > IK_WARN_M) for a, e in errs.items()},
            "ee_pos": {a: self.env.data.site_xpos[sid].round(4).tolist() for a, sid in zip(ARMS, self.env.tcp_sites)},
            "warnings": dict(self._warn),
            "input_hz": round(rate, 1),
            "rtt_ms": round(self._rtt_ms, 1) if rate > 0 else None,
            "loop_hz": round(self._loop_hz, 1),
            "connected": rate > 0,
            "placement": self.placement.column_major(),
            "control": {"mode": self.control_mode, "driver": self._driver,
                        "policy": None if self.policy is None else self.policy.name,
                        "safety_rad": round(self._safety_rad, 3)},
            "review": self._review_summary(),
            "replay": None if self._replay is None else {
                "path": self._replay["path"], "i": self._replay["i"], "n": len(self._replay["states"]),
                "paused": self._replay["paused"], "speed": self._replay_speed, "loop": self._replay_loop},
        }
        with self._out_lock:
            self._status = status
