"""One-Euro filtering for positions and orientations (Casiez et al., 2012)."""

from __future__ import annotations

import numpy as np


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


def _rot_angle(Rrel: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1.0, 1.0)))


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
        angle = _rot_angle(Rrel)
        a_d = _alpha(self.d_cutoff, dt)
        self.speed = a_d * angle / dt + (1 - a_d) * self.speed
        a = _alpha(self.min_cutoff + self.beta * self.speed, dt)
        self.R = _slerp_mat(self.R, R, Rrel, angle, a)
        return self.R.copy()


class PoseFilter:
    def __init__(self, min_cutoff: float, beta: float):
        self.pos = OneEuro(min_cutoff, beta)
        self.rot = OneEuroRotation(min_cutoff, beta * 0.6)

    def reset(self) -> None:
        self.pos.reset()
        self.rot.reset()

    def __call__(self, p, R, dt):
        return self.pos(p, dt), self.rot(R, dt)
