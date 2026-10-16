"""Episode recording state machine: IDLE -> COUNTDOWN -> RECORDING -> REVIEW -> (save | discard) -> IDLE."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import yaml

from questcollect.record.hdf5_io import next_episode_path, write_episode
from questcollect.record.schema import SCHEMA_VERSION

AUTO_STOP_TAIL_S = 1.0


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
        self.initial_state: np.ndarray | None = None
        self.countdown_end = 0.0
        self.t_start = 0.0
        self.success_since: float | None = None
        self.success_seen = False
        self.auto_stop = True
        self.last_saved: str | None = None
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

    def save(self, env, teleop_mode: str, success: bool | None = None) -> str | None:
        if self.state == RecState.RECORDING:
            self.stop()
        if self.state != RecState.REVIEW or not self.steps:
            return None
        ok = self.success_seen if success is None else bool(success)
        path = next_episode_path(self.out_dir, env.task.name)
        steps, init = self.steps, self.initial_state
        duration = len(steps) / self.cfg.sim.control_hz
        meta = {
            "schema_version": SCHEMA_VERSION,
            "task": env.task.name,
            "instruction": env.task.instruction,
            "operator": self.cfg.record.operator,
            "input_mode": teleop_mode,
            "success": ok,
            "fps": self.cfg.sim.control_hz,
            "seed": env.seed if env.seed is not None else -1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "duration_s": duration,
            "num_steps": len(steps),
            "model_xml": env.xml,
            "config_yaml": yaml.safe_dump(self.cfg.model_dump()),
            "camera_names": list(self.cfg.record.cameras),
            "objects": list(env.task.objects),
        }
        self._pending = self._writer.submit(write_episode, path, steps, meta, {"sim": {"initial_state": init}})
        self.last_saved = str(path)
        self.saved_count += 1
        self._clear()
        return str(path)

    def _clear(self) -> None:
        self.state = RecState.IDLE
        self.steps = []
        self.initial_state = None
        self.success_since = None
        self.success_seen = False

    # -- per control tick ------------------------------------------------------------------
    def tick(self, now: float, env) -> None:
        """Advance countdown; call once per control step before `append`."""
        if self.state == RecState.COUNTDOWN and now >= self.countdown_end:
            self.state = RecState.RECORDING
            self.t_start = now
            self.steps = []
            self.initial_state = env.get_state()
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

