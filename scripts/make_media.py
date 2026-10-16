"""Renders the README media in docs/media/ from the simulator (no headset needed).

    MUJOCO_GL=egl uv run python scripts/make_media.py

- demo.gif      scripted transfer_cube demo (IK-driven, same control stack as teleop)
- tasks.png     the three tasks side by side
- cameras.png   the four recorded camera views (what goes into the LeRobot dataset)
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from questcollect.config import ROOT, load_config
from questcollect.ik.solver import BimanualIK
from questcollect.record.render import CameraRenderer
from questcollect.sim.env import BimanualEnv

OUT = ROOT / "docs" / "media"
DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # TCP pointing at the table (wxyz)


def free_camera(distance=1.75, azimuth=180.0, elevation=-28.0, lookat=(0.35, 0.0, 0.92)) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def label(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(14, img.height // 18))
    pad = img.height // 40
    box = d.textbbox((0, 0), text, font=font)
    d.rounded_rectangle((pad, pad, pad * 3 + box[2], pad * 3 + box[3]), radius=pad, fill=(10, 14, 24))
    d.text((pad * 2, pad * 2), text, font=font, fill=(232, 238, 252))
    return img


def demo_gif(cfg, path: Path, size=(640, 400), every=4) -> None:
    env = BimanualEnv(cfg, "transfer_cube")
    env.reset(0)
    th = cfg.sim.table_height
    # Deterministic layout for the demo.
    env.task.set_free_pose(env.data, "cube", [0.48, 0.2, th + 0.02])
    env.data.mocap_pos[env.model.body("target").mocapid[0]] = [0.42, -0.25, th]
    mujoco.mj_forward(env.model, env.data)
    ik = BimanualIK(cfg.sim, cfg.ik, env.control_dt)

    home = {a: ik.tcp_pose(a)[0] for a in "lr"}
    z_grasp, z_up = th + 0.02, th + 0.2
    cube, mid, tgt = np.array([0.48, 0.2]), np.array([0.45, 0.0]), np.array([0.42, -0.25])
    at = lambda xy, z: np.array([*xy, z])  # noqa: E731
    # (seconds, left target, right target, left grip, right grip); grip 1 = open.
    keys = [
        (0.0, home["l"], home["r"], 1, 1),
        (1.2, at(cube, z_up), home["r"], 1, 1),
        (2.2, at(cube, z_grasp), home["r"], 1, 1),
        (2.8, at(cube, z_grasp), home["r"], 0, 1),
        (3.8, at(cube, z_up), home["r"], 0, 1),
        (5.0, at(mid, z_grasp + 0.005), at(mid + [0, -0.22], z_up), 0, 1),
        (5.5, at(mid, z_grasp + 0.005), at(mid + [0, -0.22], z_up), 1, 1),
        (6.5, at(mid + [0, 0.22], z_up), at(mid, z_up), 1, 1),
        (7.4, home["l"], at(mid, z_grasp), 1, 1),
        (8.0, home["l"], at(mid, z_grasp), 1, 0),
        (9.0, home["l"], at(mid, z_up), 1, 0),
        (10.4, home["l"], at(tgt, z_grasp + 0.01), 1, 0),
        (11.0, home["l"], at(tgt, z_grasp + 0.01), 1, 1),
        (12.2, home["l"], home["r"], 1, 1),
        (13.0, home["l"], home["r"], 1, 1),
    ]
    times = np.array([k[0] for k in keys])

    def interp(t, i):
        j = min(np.searchsorted(times, t, side="right"), len(keys) - 1)
        k0, k1 = keys[j - 1], keys[j]
        s = np.clip((t - k0[0]) / max(k1[0] - k0[0], 1e-6), 0, 1)
        s = s * s * (3 - 2 * s)  # smoothstep
        return np.asarray(k0[i]) + s * (np.asarray(k1[i]) - np.asarray(k0[i]))

    renderer = mujoco.Renderer(env.model, height=size[1], width=size[0])
    cam = free_camera()
    frames = []
    n = int(times[-1] / env.control_dt)
    for step in range(n):
        t = step * env.control_dt
        for i, a in ((1, "l"), (2, "r")):
            ik.set_target(a, interp(t, i), DOWN)
        env.apply_action(ik.solve(), np.array([interp(t, 3), interp(t, 4)]))
        env.step()
        if step % every == 0:
            renderer.update_scene(env.data, camera=cam)
            frames.append(Image.fromarray(renderer.render()))
    renderer.close()
    print(f"demo success: {env.success()}")
    # One shared palette + a static camera lets the GIF encoder store only what changed per frame.
    palette = frames[len(frames) // 2].quantize(colors=128)
    pal = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=int(1000 * env.control_dt * every),
                loop=0, optimize=True)


def tasks_png(cfg, path: Path, size=(480, 400)) -> None:
    tiles = []
    for name in ("transfer_cube", "insertion", "lift_box"):
        env = BimanualEnv(cfg, name)
        env.reset(1)
        r = mujoco.Renderer(env.model, height=size[1], width=size[0])
        r.update_scene(env.data, camera=free_camera(distance=1.9, elevation=-30, lookat=(0.3, 0.0, 1.0)))
        tiles.append(label(Image.fromarray(r.render()), name))
        r.close()
    sheet = Image.new("RGB", (size[0] * len(tiles), size[1]))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * size[0], 0))
    sheet.save(path, optimize=True)


def cameras_png(cfg, path: Path, size=(320, 240)) -> None:
    env = BimanualEnv(cfg, "transfer_cube")
    env.reset(2)
    imgs = CameraRenderer(env.model, size[1], size[0], cfg.record.cameras).render(env.data)
    sheet = Image.new("RGB", (size[0] * 2, size[1] * 2))
    for i, (name, img) in enumerate(imgs.items()):
        sheet.paste(label(Image.fromarray(img), name), ((i % 2) * size[0], (i // 2) * size[1]))
    sheet.save(path, optimize=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    demo_gif(cfg, OUT / "demo.gif")
    tasks_png(cfg, OUT / "tasks.png")
    cameras_png(cfg, OUT / "cameras.png")
    for p in sorted(OUT.iterdir()):
        print(f"{p.name:<16} {p.stat().st_size / 1e6:.2f} MB")
