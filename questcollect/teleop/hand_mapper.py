"""Tracked hand -> control pose / clutch gesture / gripper."""

from __future__ import annotations

import numpy as np

from questcollect.config import TeleopConfig
from questcollect.server.protocol import J_MIDDLE_META, J_WRIST, HandInput


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
