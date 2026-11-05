"""Render recorded episodes (demos or `qc eval` rollouts) to MP4: an overview camera plus the policy cameras.

    MUJOCO_GL=egl uv run python scripts/render_video.py <episode.hdf5> [...] [--out DIR] [--speed 2]

Frames are re-rendered from sim/state in the episode's own scene. Overlay: time, tower progress, grippers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import av
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from questcollect.recorder import read_episode
from questcollect.sim import LEVELS, CameraRenderer, env_from_episode

W, H = 960, 540              # overview size; the camera strip below is 3 x (320 x 240) scaled to the width
STRIP_CAMS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
FPS = 25


def overview_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.42, 0.0, 0.80)
    cam.distance, cam.azimuth, cam.elevation = 1.35, 180.0, -38.0
    return cam


def overlay(img: Image.Image, lines: list[str]) -> None:
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=22)
    y = 10
    for text in lines:
        box = d.textbbox((0, 0), text, font=font)
        d.rounded_rectangle((10, y, 30 + box[2], y + box[3] + 14), radius=8, fill=(10, 14, 24))
        d.text((20, y + 6), text, font=font, fill=(232, 238, 252))
        y += box[3] + 20


def render(path: Path, out: Path, speed: float) -> Path:
    meta, d = read_episode(path, keys=["sim/state", "progress", "action/joint_pos", "obs/gripper_width"])
    env = env_from_episode(meta)
    env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, W)
    env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, H)
    main = mujoco.Renderer(env.model, H, W)
    strip = CameraRenderer(env.model, 240, 320, STRIP_CAMS)
    cam = overview_camera()
    fps_rec = int(meta["fps"])
    stride = max(1, round(speed * fps_rec / FPS))
    strip_h = round(240 * W / (3 * 320))
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{path.parent.name}_{path.stem}.mp4"
    with av.open(str(dst), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = W, H + strip_h, "yuv420p"
        for t in range(0, len(d["sim/state"]), stride):
            env.set_state(d["sim/state"][t])
            main.update_scene(env.data, camera=cam)
            top = Image.fromarray(main.render())
            grip = d["action/joint_pos"][t, [7, 15]]
            overlay(top, [f"{path.parent.name}/{path.name}   t {t / fps_rec:5.1f} s   x{speed:g}",
                          f"tower {int(d['progress'][t])}/{LEVELS}   grip cmd L {'open' if grip[0] > 0.5 else 'CLOSE'}"
                          f"  R {'open' if grip[1] > 0.5 else 'CLOSE'}"])
            cams = np.concatenate(list(strip.render(env.data).values()), axis=1)
            bottom = Image.fromarray(cams).resize((W, strip_h))
            frame = Image.new("RGB", (W, H + strip_h))
            frame.paste(top, (0, 0))
            frame.paste(bottom, (0, H))
            for packet in stream.encode(av.VideoFrame.from_image(frame)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    main.close()
    strip.close()
    return dst


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("episodes", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=None, help="Output folder (default: <episode folder>/videos)")
    p.add_argument("--speed", type=float, default=2.0, help="Playback speed (2 = twice real time)")
    a = p.parse_args()
    for ep in a.episodes:
        print(render(ep, a.out or ep.parent / "videos", a.speed))


if __name__ == "__main__":
    main()
