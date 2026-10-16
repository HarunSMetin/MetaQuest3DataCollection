"""Binary wire protocol between the WebXR client and the server.

Mirrors web/src/protocol.ts — change both together.

Input (client -> server), little endian:
  u8 type=1 | u8 src_left | u8 src_right | u8 reserved | u32 seq | f64 t_client_ms | f64 t_echo_ms
  f32[7]  head pose            (pos xyz, quat xyzw; XR local-floor frame)
  per hand (left, right):
    f32[7]   control pose      (controller grip pose, or hand wrist pose)
    f32[8]   buttons           (trigger, squeeze, stick_x, stick_y, primary, secondary, stick_press, menu)
    f32[175] 25 hand joints    (pos xyz, quat xyzw each; zeros for controllers)
    f32[2]   pinch distances   (thumb-index, thumb-middle) in metres

Pose frame (server -> client):
  u8 type=2 | u8 flags | u16 nbody | u32 seq | f64 t_server_ms | f64 t_echo_ms
  f32[nbody*7] body poses (MuJoCo world, pos xyz + quat xyzw)
  f32[14] target ghost poses (left, right) | f32[2] gripper commands
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field

import numpy as np

MSG_INPUT = 1
MSG_POSE = 2

SRC_NONE, SRC_CONTROLLER, SRC_HAND = 0, 1, 2
N_HAND_JOINTS = 25
HAND_FLOATS = 7 + 8 + N_HAND_JOINTS * 7 + 2
HEADER = struct.Struct("<BBBBIdd")
INPUT_FLOATS = 7 + 2 * HAND_FLOATS
INPUT_SIZE = HEADER.size + 4 * INPUT_FLOATS

POSE_HEADER = struct.Struct("<BBHIdd")
FLAG_ENGAGED_L, FLAG_ENGAGED_R, FLAG_RECORDING = 1, 2, 4

# WebXR hand joint indices used by the mappers.
J_WRIST, J_THUMB_TIP, J_INDEX_TIP, J_MIDDLE_META, J_MIDDLE_TIP = 0, 4, 9, 10, 14

# Button slots.
B_TRIGGER, B_SQUEEZE, B_STICK_X, B_STICK_Y, B_PRIMARY, B_SECONDARY, B_STICK_PRESS, B_MENU = range(8)


@dataclass
class HandInput:
    source: int = SRC_NONE
    pose: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 0, 0, 0, 1.0]))
    buttons: np.ndarray = field(default_factory=lambda: np.zeros(8))
    joints: np.ndarray = field(default_factory=lambda: np.zeros((N_HAND_JOINTS, 7)))
    pinch: np.ndarray = field(default_factory=lambda: np.full(2, 0.1))


@dataclass
class InputFrame:
    seq: int = 0
    t_client: float = 0.0
    t_echo: float = 0.0
    head: np.ndarray = field(default_factory=lambda: np.array([0, 1.6, 0, 0, 0, 0, 1.0]))
    left: HandInput = field(default_factory=HandInput)
    right: HandInput = field(default_factory=HandInput)
    recv_time: float = field(default_factory=time.monotonic)

    def hand(self, arm: str) -> HandInput:
        return self.left if arm == "l" else self.right


def _parse_hand(src: int, f: np.ndarray) -> HandInput:
    return HandInput(
        source=src,
        pose=f[0:7].astype(float),
        buttons=f[7:15].astype(float),
        joints=f[15:15 + N_HAND_JOINTS * 7].reshape(N_HAND_JOINTS, 7).astype(float),
        pinch=f[15 + N_HAND_JOINTS * 7:HAND_FLOATS].astype(float),
    )


def parse_input(buf: bytes) -> InputFrame:
    if len(buf) != INPUT_SIZE:
        raise ValueError(f"input message has {len(buf)} bytes, expected {INPUT_SIZE}")
    mtype, src_l, src_r, _, seq, t_client, t_echo = HEADER.unpack_from(buf, 0)
    if mtype != MSG_INPUT:
        raise ValueError(f"unexpected message type {mtype}")
    f = np.frombuffer(buf, dtype="<f4", offset=HEADER.size)
    o = 7
    left = _parse_hand(src_l, f[o:o + HAND_FLOATS])
    right = _parse_hand(src_r, f[o + HAND_FLOATS:o + 2 * HAND_FLOATS])
    return InputFrame(seq, t_client, t_echo, f[0:7].astype(float), left, right)


def pack_input(frame: InputFrame) -> bytes:
    """Used by tests and the scripted mock client."""
    parts = [frame.head]
    for h in (frame.left, frame.right):
        parts += [h.pose, h.buttons, h.joints.reshape(-1), h.pinch]
    body = np.concatenate(parts).astype("<f4").tobytes()
    head = HEADER.pack(MSG_INPUT, frame.left.source, frame.right.source, 0, frame.seq, frame.t_client, frame.t_echo)
    return head + body


def pack_pose_frame(seq: int, t_server: float, t_echo: float, body_pose: np.ndarray,
                    ghosts: np.ndarray, grippers: np.ndarray, flags: int) -> bytes:
    nbody = body_pose.shape[0]
    head = POSE_HEADER.pack(MSG_POSE, flags, nbody, seq, t_server, t_echo)
    payload = np.concatenate([body_pose.reshape(-1), ghosts.reshape(-1), grippers]).astype("<f4")
    return head + payload.tobytes()
