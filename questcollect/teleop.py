"""Operator input -> per-arm end-effector targets.

Pipeline per arm and control tick:
  XR pose -> MuJoCo world (frames.Placement) -> jump rejection -> One-Euro filter -> clutch -> workspace clamp

Controllers: a Grip press toggles the clutch, Trigger closes the gripper, A/B/X/stick send recorder commands.
Hands:       thumb-middle pinch toggles the clutch, thumb-index distance sets the gripper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from questcollect.config import TeleopConfig
from questcollect.frames import Placement
from questcollect.protocol import (
    B_PRIMARY, B_SECONDARY, B_SQUEEZE, B_STICK_PRESS, B_TRIGGER, J_MIDDLE_META, J_WRIST, SRC_CONTROLLER, SRC_HAND,
    HandInput, InputFrame,
)
from questcollect.sim import ARMS

MODES = ("controllers", "hands")
INPUT_TIMEOUT_S = 0.25
MAX_JUMP_M = 0.12  # hand/controller jump between control ticks treated as a tracking glitch


# -- One-Euro filtering for positions and orientations (Casiez et al., 2012) --------------------------
def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.5, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = None

    def reset(self) -> None:
        self.x = self.dx = None

    def __call__(self, x, dt: float) -> np.ndarray:
        x = np.asarray(x, float)
        if self.x is None or dt <= 0:
            self.x, self.dx = x.copy(), np.zeros_like(x)
            return self.x.copy()
        dx = (x - self.x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx)
        a = _alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        return self.x.copy()


def _slerp_mat(Ra: np.ndarray, Rb: np.ndarray, Rrel: np.ndarray, angle: float, a: float) -> np.ndarray:
    """Interpolate from Ra towards Rb (Rrel = Ra^T Rb, rotating by `angle`) by fraction a."""
    if angle < 1e-9:
        return Rb.copy()
    axis = np.array([Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0], Rrel[1, 0] - Rrel[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-9:  # angle ~ pi: no unique axis, snap
        return Rb.copy()
    axis /= n
    th = a * angle
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return Ra @ (np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K)


class OneEuroRotation:
    """One-Euro style adaptive smoothing on SO(3) using angular speed."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.3, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.R = None
        self.speed = 0.0

    def reset(self) -> None:
        self.R = None
        self.speed = 0.0

    def __call__(self, R, dt: float) -> np.ndarray:
        R = np.asarray(R, float)
        if self.R is None or dt <= 0:
            self.R = R.copy()
            return self.R.copy()
        Rrel = self.R.T @ R
        angle = float(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1.0, 1.0)))
        a_d = _alpha(self.d_cutoff, dt)
        self.speed = a_d * angle / dt + (1 - a_d) * self.speed
        a = _alpha(self.min_cutoff + self.beta * self.speed, dt)
        self.R = _slerp_mat(self.R, R, Rrel, angle, a)
        return self.R.copy()


# -- clutch ------------------------------------------------------------------------------------------
class ArmClutch:
    """Relative mapping: while engaged, EE target = EE pose at engage time + scaled hand motion since engage.

    Position:    p = p_E0 + s * (p_H - p_H0)
    Orientation: R = (R_H @ R_H0^T) @ R_E0   (hand rotation applied in the world frame)
    Releasing and re-engaging never makes the target jump.
    """

    def __init__(self):
        self.engaged = False
        self._hp0 = self._hR0 = self._ep0 = self._eR0 = None

    def engage(self, hand_p, hand_R, ee_p, ee_R) -> None:
        self._hp0, self._hR0 = np.array(hand_p, float), np.array(hand_R, float)
        self._ep0, self._eR0 = np.array(ee_p, float), np.array(ee_R, float)
        self.engaged = True

    def release(self) -> None:
        self.engaged = False

    def target(self, hand_p, hand_R, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        assert self.engaged, "target() requires an engaged clutch"
        p = self._ep0 + scale * (np.asarray(hand_p) - self._hp0)
        R = np.asarray(hand_R) @ self._hR0.T @ self._eR0
        return p, R


# -- input mappers ---------------------------------------------------------------------------------
# Rising-edge button -> command, per arm. Left "Y" (secondary) toggles the wrist menu client-side.
CONTROLLER_COMMANDS = {
    ("r", B_PRIMARY): "record_toggle",   # A
    ("r", B_SECONDARY): "discard",       # B
    ("l", B_PRIMARY): "save",            # X
    ("l", B_STICK_PRESS): "recenter",    # left stick click
}


class ControllerMapper:
    """Grip press toggles the clutch (no need to keep squeezing), Trigger sets the gripper."""

    GRIP_ON, GRIP_OFF = 0.6, 0.4  # press hysteresis, so one press = one toggle

    def __init__(self, arm: str):
        self.arm = arm
        self.engaged = False
        self.grip_down = False
        self.prev_buttons: dict[int, bool] = {}

    def reset(self) -> None:
        """Switch the clutch off. A grip still held down doesn't count as a new press."""
        self.engaged = False

    def update(self, h: HandInput) -> tuple[bool, float, list[str]]:
        """Returns (engaged, gripper open fraction, commands)."""
        down = h.buttons[B_SQUEEZE] > (self.GRIP_OFF if self.grip_down else self.GRIP_ON)
        if down and not self.grip_down:
            self.engaged = not self.engaged
        self.grip_down = down
        gripper = 1.0 - float(h.buttons[B_TRIGGER])
        commands = []
        for (arm, b), cmd in CONTROLLER_COMMANDS.items():
            if arm != self.arm:
                continue
            pressed = h.buttons[b] > 0.5
            if pressed and not self.prev_buttons.get(b, False):
                commands.append(cmd)
            self.prev_buttons[b] = pressed
        return self.engaged, gripper, commands


class HandMapper:
    """Control pose = middle-metacarpal position + wrist orientation (stable while pinching).

    Gripper:  thumb-index distance, mapped linearly between closed and open distances.
    Clutch:   thumb-middle pinch toggles engagement (with hysteresis so one pinch = one toggle).
    """

    CLUTCH_ON, CLUTCH_OFF = 0.018, 0.035
    MIDDLE_MARGIN = 0.01

    def __init__(self, cfg: TeleopConfig):
        self.cfg = cfg
        self.engaged = False
        self._latched = False

    def reset(self) -> None:
        self.engaged = False
        self._latched = False

    def control_pose(self, h: HandInput) -> np.ndarray:
        pose = h.joints[J_WRIST].copy()
        pose[:3] = h.joints[J_MIDDLE_META, :3]
        return pose

    def gripper(self, h: HandInput) -> float:
        lo, hi = self.cfg.pinch_closed_m, self.cfg.pinch_open_m
        return float(np.clip((h.pinch[0] - lo) / (hi - lo), 0.0, 1.0))

    def update(self, h: HandInput) -> tuple[bool, float]:
        """Returns (engaged, gripper open fraction)."""
        d, d_index = h.pinch[1], h.pinch[0]
        # An index pinch (closing the gripper) also brings the thumb near the middle fingertip;
        # only count it as a clutch gesture when the thumb is clearly closer to the middle tip.
        is_middle_pinch = d < self.CLUTCH_ON and d < d_index - self.MIDDLE_MARGIN
        if not self._latched and is_middle_pinch:
            self._latched = True
            self.engaged = not self.engaged
        elif self._latched and d > self.CLUTCH_OFF:
            self._latched = False
        return self.engaged, self.gripper(h)


# -- teleop ----------------------------------------------------------------------------------------
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
        self.pos_filter = OneEuro(cfg.filter_min_cutoff, cfg.filter_beta)
        self.rot_filter = OneEuroRotation(cfg.filter_min_cutoff, cfg.filter_beta * 0.6)
        self.ctrl = ControllerMapper(arm)
        self.hand = HandMapper(cfg)
        self.gripper = 1.0
        self.hold_p = np.zeros(3)
        self.hold_R = np.eye(3)
        self.last_raw: np.ndarray | None = None

    def drop_tracking(self) -> None:
        """Tracking lost or glitched: disengage until the operator engages again."""
        self.clutch.release()
        self.ctrl.reset()
        self.hand.reset()
        self.pos_filter.reset()
        self.rot_filter.reset()


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
            st.drop_tracking()
            st.ctrl.reset()
            st.last_raw = None
            st.gripper = 1.0
            st.hold_p, st.hold_R = ee_poses[a][0].copy(), ee_poses[a][1].copy()

    def sync_hold(self, ee_poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        """Something else (a policy) moved the arms: hold where they are now, so taking control starts there."""
        for a, st in self.arms.items():
            if not st.clutch.engaged:
                st.hold_p, st.hold_R = ee_poses[a][0].copy(), ee_poses[a][1].copy()

    def release_all(self) -> None:
        """Drop every clutch, e.g. after the mapping changed, so re-engaging starts from the held target."""
        for st in self.arms.values():
            st.clutch.release()

    def set_mode(self, mode: str) -> None:
        if mode in MODES and mode != self.mode:
            self.mode = mode
            for st in self.arms.values():
                st.clutch.release()
                st.ctrl.reset()
                st.hand.reset()

    def set_scale(self, scale: float) -> None:
        self.scale = float(np.clip(scale, 0.2, 3.0))
        self.release_all()

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
                    st.drop_tracking()
                    engage = False
                p_f, R_f = st.pos_filter(p_w, dt), st.rot_filter(R_w, dt)

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
                st.drop_tracking()
                st.last_raw = None

            out.targets[a] = (st.hold_p.copy(), st.hold_R.copy())
            out.grippers[i] = st.gripper
            out.engaged[a] = st.clutch.engaged
            out.tracked[a] = tracked
        return out
