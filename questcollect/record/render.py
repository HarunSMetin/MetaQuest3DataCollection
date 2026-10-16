"""Offscreen camera rendering (live previews, optional live recording, offline export)."""

from __future__ import annotations

import io

import mujoco
import numpy as np
from PIL import Image


class CameraRenderer:
    def __init__(self, model: mujoco.MjModel, height: int, width: int, cameras: list[str]):
        self.cameras = [c for c in cameras if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c) >= 0]
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def render(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        out = {}
        for cam in self.cameras:
            self.renderer.update_scene(data, camera=cam)
            out[cam] = self.renderer.render().copy()
        return out

    def render_jpeg(self, data: mujoco.MjData, quality: int = 70) -> dict[str, bytes]:
        out = {}
        for cam, img in self.render(data).items():
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=quality)
            out[cam] = buf.getvalue()
        return out

    def close(self) -> None:
        self.renderer.close()
