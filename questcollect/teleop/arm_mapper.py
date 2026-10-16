"""Relative ("clutch") mapping from an operator hand pose to an end-effector target."""

from __future__ import annotations

import numpy as np


class ArmClutch:
    """While engaged, EE target = EE pose at engage time + scaled hand motion since engage.

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
