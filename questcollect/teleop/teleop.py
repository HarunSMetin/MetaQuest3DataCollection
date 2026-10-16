"""Combines input, mode selection, filtering and clutch mapping into per-arm EE targets."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from questcollect.config import TeleopConfig
from questcollect.server.protocol import SRC_CONTROLLER, SRC_HAND, InputFrame
from questcollect.sim.scene import ARMS
from questcollect.teleop.arm_mapper import ArmClutch
from questcollect.teleop.controller_mapper import ControllerMapper
from questcollect.teleop.filters import PoseFilter
from questcollect.teleop.frames import Placement
from questcollect.teleop.hand_mapper import HandMapper

MODES = ("controllers", "hands")
INPUT_TIMEOUT_S = 0.25
MAX_JUMP_M = 0.12  # hand/controller jump between control ticks treated as a tracking glitch


@dataclass
class TeleopOutput:
    targets: dict[str, tuple[np.ndarray, np.ndarray]]
    grippers: np.ndarray
    engaged: dict[str, bool]
    tracked: dict[str, bool]
    commands: list[str] = field(default_factory=list)
    haptics: list[tuple[str, float, int]] = field(default_factory=list)  # (arm, intensity, ms)
    mode_mismatch: bool = False


class _ArmState:
    def __init__(self, arm: str, cfg: TeleopConfig):
        self.clutch = ArmClutch()
        self.filter = PoseFilter(cfg.filter_min_cutoff, cfg.filter_beta)
        self.ctrl = ControllerMapper(arm)
        self.hand = HandMapper(cfg)
        self.gripper = 1.0
        self.hold_p = np.zeros(3)
        self.hold_R = np.eye(3)
        self.last_raw: np.ndarray | None = None


class Teleop:
    def __init__(self, cfg: TeleopConfig):
        self.cfg = cfg
        self.mode = cfg.mode if cfg.mode in MODES else "controllers"
        self.placement = Placement()
        self.scale = cfg.scale
        self.ws_min = np.array(cfg.workspace_min)
        self.ws_max = np.array(cfg.workspace_max)
        self.arms = {a: _ArmState(a, cfg) for a in ARMS}

    def reset(self, ee_poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        """Disengage everything and hold the given EE poses."""
        for a, st in self.arms.items():
            st.clutch.release()
            st.ctrl.reset()
            st.hand.reset()
            st.filter.reset()
            st.last_raw = None
            st.gripper = 1.0
            st.hold_p, st.hold_R = ee_poses[a][0].copy(), ee_poses[a][1].copy()

    def set_mode(self, mode: str) -> None:
        if mode in MODES and mode != self.mode:
            self.mode = mode
            for st in self.arms.values():
                st.clutch.release()
                st.ctrl.reset()
                st.hand.reset()

    def set_scale(self, scale: float) -> None:
        self.scale = float(np.clip(scale, 0.2, 3.0))
        for st in self.arms.values():
            st.clutch.release()  # re-engage at the new scale without a jump

    def update(self, frame: InputFrame | None, now: float, dt: float) -> TeleopOutput:
        out = TeleopOutput(targets={}, grippers=np.ones(2), engaged={}, tracked={})
        stale = frame is None or (now - frame.recv_time) > INPUT_TIMEOUT_S
        want_src = SRC_CONTROLLER if self.mode == "controllers" else SRC_HAND
        for i, (a, st) in enumerate(self.arms.items()):
            h = None if stale else frame.hand(a)
            tracked = h is not None and h.source == want_src
            if h is not None and h.source not in (want_src, 0):
                out.mode_mismatch = True

            engage = False
            if tracked:
                if self.mode == "controllers":
                    engage, st.gripper, cmds = st.ctrl.update(h)
                    out.commands += cmds
                    pose = h.pose
                else:
                    engage, st.gripper = st.hand.update(h)
                    pose = st.hand.control_pose(h)
                p_w, R_w = self.placement.world_from_xr(pose[:3], pose[3:])
                jumped = st.last_raw is not None and np.linalg.norm(p_w - st.last_raw) > MAX_JUMP_M
                st.last_raw = p_w
                if jumped:
                    # Tracking glitch: drop the clutch rather than yank the arm across the table.
                    if st.clutch.engaged:
                        out.haptics.append((a, 0.8, 60))
                    st.clutch.release()
                    st.hand.reset()
                    st.filter.reset()
                    engage = False
                p_f, R_f = st.filter(p_w, R_w, dt)

                if engage and not st.clutch.engaged:
                    st.clutch.engage(p_f, R_f, st.hold_p, st.hold_R)
                    out.haptics.append((a, 0.6, 40))
                elif not engage and st.clutch.engaged:
                    st.clutch.release()
                    out.haptics.append((a, 0.25, 25))

                if st.clutch.engaged:
                    p, R = st.clutch.target(p_f, R_f, self.scale)
                    st.hold_p = np.clip(p, self.ws_min, self.ws_max)
                    st.hold_R = R
            else:
                # Lost tracking: freeze the arm where it is commanded to be.
                st.clutch.release()
                st.hand.reset()
                st.filter.reset()
                st.last_raw = None

            out.targets[a] = (st.hold_p.copy(), st.hold_R.copy())
            out.grippers[i] = st.gripper
            out.engaged[a] = st.clutch.engaged
            out.tracked[a] = tracked
        return out
