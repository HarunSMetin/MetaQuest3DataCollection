"""Episode recording: state machine + HDF5 episode files.

States: IDLE -> COUNTDOWN -> RECORDING -> REVIEW -> (save | discard) -> IDLE

One HDF5 file per episode (data/episodes/episode_XXXXXX.hdf5). Keys are dataset paths, time is axis 0.

meta (file attrs): schema_version, task, instruction, operator, input_mode, success, fps, seed,
                   created_at, duration_s, num_steps, model_xml, config_yaml, camera_names, objects,
                   tower (JSON: bricks bottom-up with studs/height/color/mass, per-level offsets, clutch
                   pairs and limits), scene_hash (sim.scene_hash; v5+), source (teleop | policy | shared; v5+)

/time                         (T,)        sim time [s]
/wall_time                    (T,)        host monotonic time [s]
/obs/joint_pos                (T, 14)     [left 7, right 7] rad
/obs/joint_vel                (T, 14)
/obs/gripper_width            (T, 2)      finger opening [m] (0 .. 0.08)
/obs/ee_pose_left             (T, 7)      tcp [x y z qw qx qy qz] world
/obs/ee_pose_right            (T, 7)
/obs/objects                  (T, 4, 7)   brick poses, bottom-centre of each brick, order = meta objects
                                          (privileged, sim only)
/action/joint_pos             (T, 16)     [7 L joints, grip L, 7 R joints, grip R]; grip in [0 closed, 1 open]
/action/ee_pose               (T, 16)     [L xyz qwxyz, R xyz qwxyz, grip L, grip R] IK targets
/sim/state                    (T, S)      mj_getState(mjSTATE_INTEGRATION) before the step, then the 6 clutch
                                          welds' relative poses (6 x 7); BimanualEnv.set_state restores both
/success                      (T,)        task success predicate per step (tower complete and released)
/progress                     (T,)        bricks correctly stacked and clicked in on the base so far (0..3)
/clutch                       (T, 6)      which brick pairs are clicked together (order = meta tower.clutch_pairs)
/contacts                     (T, 3)      [arm link on table, arms touching, hand/fingertip on table] (v5+;
                                          BimanualEnv.contact_flags; older files: audit.py replays sim/state)
/control/source               (T,)        who drove this step: 0 operator, 1 policy (v5+)
/teleop/head                  (T, 7)      XR head pose [x y z qx qy qz qw]
/teleop/{left,right}/source   (T,)        0 none, 1 controller, 2 hand
/teleop/{left,right}/pose     (T, 7)      raw XR control pose
/teleop/{left,right}/buttons  (T, 8)
/teleop/{left,right}/joints   (T, 25, 7)  WebXR hand joints (zeros for controllers)
/teleop/{left,right}/pinch    (T, 2)
/teleop/engaged               (T, 2)
/teleop/tracked               (T, 2)
/teleop/rtt_ms                (T,)

Camera images are not stored: sim/state replays the episode exactly, and the exporter re-renders them.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from pathlib import Path

import h5py
import numpy as np
import yaml

from questcollect.protocol import InputFrame
from questcollect.sim import (BRICK_H, BRICKS, CLUTCH_PAIRS, CLUTCH_PRESS_N, CLUTCH_PULL_N, INSTRUCTION, OBJECTS, PITCH,
                              TASK, TOWER)

SCHEMA_VERSION = 5
AUTO_STOP_TAIL_S = 1.0
SOURCE_OPERATOR, SOURCE_POLICY = 0, 1


# -- one step --------------------------------------------------------------------------------------
def teleop_fields(frame: InputFrame | None, engaged=(False, False), tracked=(False, False),
                  rtt_ms: float = 0.0) -> dict:
    """The raw operator input of one step (all zeros without a headset, e.g. a policy rollout)."""
    out = {"head": np.zeros(7) if frame is None else frame.head, "engaged": np.array(engaged, bool),
           "tracked": np.array(tracked, bool), "rtt_ms": rtt_ms}
    for side, a in (("left", "l"), ("right", "r")):
        h = None if frame is None else frame.hand(a)
        out[side] = {
            "source": 0 if h is None else h.source,
            "pose": np.zeros(7) if h is None else h.pose,
            "buttons": np.zeros(8) if h is None else h.buttons,
            "joints": np.zeros((25, 7)) if h is None else h.joints,
            "pinch": np.zeros(2) if h is None else h.pinch,
        }
    return out


def make_step(env, obs: dict, state: np.ndarray, q_target: np.ndarray, grippers: np.ndarray, ee_targets,
              success: bool, progress: int, contacts: np.ndarray, teleop: dict,
              source: int = SOURCE_OPERATOR) -> dict:
    """One recorded step in the layout above. Used by the live recorder and by policy rollouts, so every
    episode file has the same fields whoever drove the arms.

    q_target: (14,) joint targets [left 7, right 7]; grippers: (2,) in [0 closed, 1 open];
    ee_targets: the two TCP targets, each [x y z qw qx qy qz].
    """
    return {
        "time": env.data.time,
        "wall_time": time.monotonic(),
        "obs": obs,
        "action": {
            "joint_pos": np.concatenate([q_target[:7], grippers[:1], q_target[7:], grippers[1:]]),
            "ee_pose": np.concatenate([ee_targets[0], ee_targets[1], grippers]),
        },
        "sim": {"state": state},
        "success": success,
        "progress": progress,
        "clutch": env.data.eq_active[env.clutch_eqs].copy(),
        "contacts": np.asarray(contacts, bool),
        "control": {"source": source},
        "teleop": teleop,
    }


# -- episode files ---------------------------------------------------------------------------------
def _flatten(d: dict, prefix: str = "") -> dict[str, object]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def write_episode(path: Path, steps: list[dict], meta: dict) -> Path:
    """steps: per-step nested dicts with identical structure; nesting becomes HDF5 groups."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    flat = [_flatten(s) for s in steps]
    with h5py.File(tmp, "w") as f:
        for k, v in meta.items():
            f.attrs[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
        for key in flat[0]:
            arr = np.stack([np.asarray(s[key]) for s in flat])
            f.create_dataset(key, data=arr, compression="lzf" if arr.size > 4096 else None)
    tmp.rename(path)
    return path


def _decode(v):
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    if isinstance(v, np.generic):
        return v.item()
    return v


def read_meta(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        return {k: _decode(v) for k, v in f.attrs.items()}


def read_episode(path: Path, keys: list[str] | None = None) -> tuple[dict, dict]:
    """Returns (meta, data) with data as a flat dict path -> ndarray."""
    data = {}
    with h5py.File(path, "r") as f:
        meta = {k: _decode(v) for k, v in f.attrs.items()}

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and (keys is None or name in keys):
                data[name] = obj[()]

        f.visititems(visit)
    return meta, data


def episode_files(out_dir: Path) -> list[Path]:
    return sorted(Path(out_dir).glob("episode_*.hdf5"))


def next_episode_path(out_dir: Path) -> Path:
    nums = [int(p.stem.split("_")[-1]) for p in episode_files(out_dir) if p.stem.split("_")[-1].isdigit()]
    return Path(out_dir) / f"episode_{(max(nums) + 1 if nums else 0):06d}.hdf5"


def list_episodes(out_dir: Path) -> list[dict]:
    rows = []
    for p in episode_files(out_dir):
        try:
            m = read_meta(p)
        except OSError:
            continue
        rows.append({
            "path": str(p), "file": p.name, "success": bool(m.get("success")),
            "duration_s": m.get("duration_s"), "num_steps": m.get("num_steps"),
            "input_mode": m.get("input_mode"), "operator": m.get("operator"), "created_at": m.get("created_at"),
            "source": m.get("source", "teleop"),
        })
    return rows


def episode_meta(cfg, env, num_steps: int, success: bool, input_mode: str, operator: str,
                 source: str = "teleop") -> dict:
    """File attributes of an episode (see the module docstring)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "instruction": INSTRUCTION,
        "operator": operator,
        "input_mode": input_mode,
        "source": source,
        "success": bool(success),
        "fps": cfg.sim.control_hz,
        "seed": env.seed if env.seed is not None else -1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "duration_s": num_steps / cfg.sim.control_hz,
        "num_steps": num_steps,
        "model_xml": env.xml,
        "scene_hash": env.scene_hash,
        "config_yaml": yaml.safe_dump(cfg.model_dump()),
        "camera_names": list(cfg.record.cameras),
        "objects": list(OBJECTS),
        "tower": {"pitch": PITCH, "brick_height": BRICK_H,
                  "bricks": [{"name": b.name, "studs": [b.nx, b.ny], "height": b.nz, "rgba": list(b.rgba),
                              "mass": b.mass} for b in BRICKS],
                  "levels": [{"dx": dx, "dy": dy, "yaw": dyaw} for dx, dy, dyaw in TOWER],
                  "clutch_pairs": [list(pair) for pair in CLUTCH_PAIRS],
                  "clutch_press_n": CLUTCH_PRESS_N, "clutch_pull_n": CLUTCH_PULL_N},
    }


# -- recorder state machine ------------------------------------------------------------------------
class RecState(str, Enum):
    IDLE = "idle"
    COUNTDOWN = "countdown"
    RECORDING = "recording"
    REVIEW = "review"


class Recorder:
    def __init__(self, cfg, out_dir: Path):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.state = RecState.IDLE
        self.steps: list[dict] = []
        self.countdown_end = 0.0
        self.t_start = 0.0
        self.success_since: float | None = None
        self.success_seen = False
        self.auto_stop = True
        self.saved_count = 0
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hdf5-writer")
        self._pending: Future | None = None

    # -- transitions ---------------------------------------------------------------------
    def start(self, now: float) -> bool:
        if self.state != RecState.IDLE:
            return False
        self.state = RecState.COUNTDOWN
        self.countdown_end = now + self.cfg.record.countdown_s
        return True

    def stop(self) -> bool:
        if self.state == RecState.COUNTDOWN:
            self.state = RecState.IDLE
            return True
        if self.state != RecState.RECORDING:
            return False
        self.state = RecState.REVIEW if self.steps else RecState.IDLE
        return True

    def toggle(self, now: float) -> None:
        if self.state == RecState.IDLE:
            self.start(now)
        else:
            self.stop()

    def discard(self) -> bool:
        """Drop the current take. Returns True if a take was dropped (caller resets the scene)."""
        if self.state == RecState.IDLE:
            return False
        self._clear()
        return True

    def save(self, env, teleop_mode: str, success: bool | None = None, source: str = "teleop") -> str | None:
        """Write the take in the background. success=None uses whether the task check ever passed."""
        if self.state == RecState.RECORDING:
            self.stop()
        if self.state != RecState.REVIEW or not self.steps:
            return None
        path = next_episode_path(self.out_dir)
        meta = episode_meta(self.cfg, env, len(self.steps), self.success_seen if success is None else bool(success),
                            input_mode=teleop_mode, operator=self.cfg.record.operator, source=source)
        self._pending = self._writer.submit(write_episode, path, self.steps, meta)
        self.saved_count += 1
        self._clear()
        return str(path)

    def _clear(self) -> None:
        self.state = RecState.IDLE
        self.steps = []
        self.success_since = None
        self.success_seen = False

    # -- per control tick ------------------------------------------------------------------
    def tick(self, now: float) -> None:
        """Advance the countdown; call once per control step before `append`."""
        if self.state == RecState.COUNTDOWN and now >= self.countdown_end:
            self.state = RecState.RECORDING
            self.t_start = now
            self.steps = []
            self.success_since = None
            self.success_seen = False

    def append(self, now: float, step: dict, success: bool) -> bool:
        """Store one step while recording. Returns True if auto-stop fired (task succeeded)."""
        if self.state != RecState.RECORDING:
            return False
        self.steps.append(step)
        if success:
            self.success_seen = True
            self.success_since = self.success_since or now
            if self.auto_stop and now - self.success_since >= AUTO_STOP_TAIL_S:
                self.stop()
                return True
        else:
            self.success_since = None
        return False

    @property
    def recording(self) -> bool:
        return self.state == RecState.RECORDING

    def elapsed(self, now: float) -> float:
        if self.state == RecState.RECORDING:
            return now - self.t_start
        if self.state == RecState.REVIEW:
            return len(self.steps) / self.cfg.sim.control_hz
        return 0.0

    def countdown_left(self, now: float) -> float:
        return max(0.0, self.countdown_end - now) if self.state == RecState.COUNTDOWN else 0.0

    def flush(self, timeout: float = 30.0) -> None:
        if self._pending is not None:
            self._pending.result(timeout=timeout)

    def close(self) -> None:
        self.flush()
        self._writer.shutdown(wait=True)
