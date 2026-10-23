"""Coordinate frames between WebXR and MuJoCo.

WebXR (local-floor): y up, -z forward, x right, quaternions as [x y z w].
MuJoCo world:        z up, +x forward (away from operator), +y left, quaternions as [w x y z].

The placement T_xr_from_world = [Ry(theta) @ R_MJ2XR | t] positions the simulated world in the
room. It is used both to render the scene in the headset and to map hand poses back to the world,
so what the operator sees is what they control.
"""

from __future__ import annotations

import numpy as np

# Columns are world basis vectors expressed in XR: world +x -> xr -z, world +y -> xr -x, world +z -> xr +y.
R_MJ2XR = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])


def quat_wxyz_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat_wxyz(R) -> np.ndarray:
    R = np.asarray(R)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def xyzw_to_wxyz(q) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]])


def wxyz_to_xyzw(q) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]])


def rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class Placement:
    """Rigid transform taking MuJoCo world coordinates into the XR room."""

    def __init__(self, theta: float = 0.0, t=(0.0, 0.0, 0.0), operator_x: float = -0.5):
        self.theta = float(theta)
        self.t = np.asarray(t, float)
        self.operator_x = operator_x
        if not np.any(self.t):
            # Default: operator standing at the XR origin.
            self.t = -(rot_y(self.theta) @ R_MJ2XR @ np.array([operator_x, 0.0, 0.0]))

    @property
    def R(self) -> np.ndarray:
        """Rotation world -> xr."""
        return rot_y(self.theta) @ R_MJ2XR

    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def column_major(self) -> list[float]:
        """4x4 world -> xr matrix in three.js (column-major) order."""
        return self.matrix().T.flatten().tolist()

    def recenter(self, head_pos, head_quat_xyzw) -> None:
        """Put the operator's head behind the arm bases, looking along world +x."""
        R_head = quat_wxyz_to_mat(xyzw_to_wxyz(head_quat_xyzw))
        fwd = -R_head[:, 2]
        fwd[1] = 0.0
        if np.linalg.norm(fwd) < 1e-6:
            return
        fwd /= np.linalg.norm(fwd)
        self.theta = float(np.arctan2(-fwd[0], -fwd[2]))
        head = np.asarray(head_pos, float)
        op_world = np.array([self.operator_x, 0.0, head[1]])
        self.t = head - self.R @ op_world
        self.t[1] = 0.0  # keep the simulated floor on the real floor

    def world_from_xr(self, pos, quat_xyzw) -> tuple[np.ndarray, np.ndarray]:
        """XR pose -> (world position, world rotation matrix)."""
        R = self.R
        p_w = R.T @ (np.asarray(pos, float) - self.t)
        R_w = R.T @ quat_wxyz_to_mat(xyzw_to_wxyz(quat_xyzw))
        return p_w, R_w

    def world_point_from_xr(self, pos) -> np.ndarray:
        return self.R.T @ (np.asarray(pos, float) - self.t)
