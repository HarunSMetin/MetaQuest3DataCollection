"""Renders the README media in docs/media/ from the simulator (no headset needed).

    MUJOCO_GL=egl uv run python scripts/make_media.py

- demo.gif      scripted build_tower demo (IK-driven, same control stack as teleop)
- tower.png     the four bricks, a brick's underside, and the goal tower (close-ups)
- cameras.png   the four recorded camera views (what goes into the LeRobot dataset)
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from questcollect.config import ROOT, load_config
from questcollect.ik import BimanualIK
from questcollect.sim import (BRICK_H, BRICKS, LEVELS, STUD_H, BimanualEnv, CameraRenderer, set_brick_pose,
                              tower_poses)

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


# Deterministic demo layout: brick -> (x, y, yaw); the tower goes in the middle of the table.
DEMO_LAYOUT = {"red_2x4": (0.40, 0.22, 0.4), "blue_2x3": (0.55, -0.22, -0.3),
               "yellow_2x2": (0.36, -0.25, 0.8), "green_1x2x2": (0.56, 0.24, -0.6)}
DEMO_TOWER = ((0.47, 0.0), 0.0)


def grasp_quat(yaw: float) -> np.ndarray:
    """TCP pointing down with the fingers closing across the brick's y axis (shortest wrist turn)."""
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
    qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    out = np.empty(4)
    mujoco.mju_mulQuat(out, qz, DOWN)
    return out


def tower_keys(env, ik) -> list[tuple]:
    """Keyframes (seconds, left pose, right pose, left grip, right grip) that build the tower brick by brick."""
    th = env.table_height
    lift = 0.12
    home = {a: (ik.tcp_pose(a)[0], DOWN) for a in "lr"}
    goals = tower_poses([*DEMO_TOWER[0], th], DEMO_TOWER[1])
    pose = dict(home)
    grip = {"l": 1, "r": 1}
    t = 0.0
    keys = [(t, pose["l"], pose["r"], 1, 1)]

    def key(dt, arm, p=None, q=None, g=None):
        nonlocal t
        t += dt
        if p is not None:
            pose[arm] = (np.asarray(p, float), pose[arm][1] if q is None else q)
        if g is not None:
            grip[arm] = g
        keys.append((t, pose["l"], pose["r"], grip["l"], grip["r"]))

    for b, (gp, gyaw) in zip(BRICKS, goals):
        x, y, yaw = DEMO_LAYOUT[b.name]
        a = "l" if y > 0 else "r"
        q_pick, q_place = grasp_quat(yaw), grasp_quat(gyaw)
        grasp_z = b.h / 2 + 0.002
        key(1.2, a, [x, y, th + lift], q_pick)
        key(0.8, a, [x, y, th + grasp_z])
        key(0.5, a, g=0)
        key(0.8, a, [x, y, th + lift])
        key(1.2, a, [gp[0], gp[1], gp[2] + lift], q_place)
        key(0.9, a, [gp[0], gp[1], gp[2] + grasp_z + 0.004])
        key(0.4, a, [gp[0], gp[1], gp[2] + grasp_z])
        key(0.4, a, [gp[0], gp[1], gp[2] + grasp_z - 0.004])  # press it down until it clicks
        key(0.5, a, g=1)
        key(0.8, a, [gp[0], gp[1], gp[2] + lift])
        key(1.0, a, *home[a])
    key(1.0, "l")
    return keys


def gif_palette(frames: list[Image.Image], colors: int = 160) -> Image.Image:
    """Palette from frames across the whole clip plus each brick colour from shadow to highlight: the bricks
    cover a few hundred pixels, so a palette from the frames alone renders them grey."""
    w, h = frames[0].size
    picks = frames[:: max(1, len(frames) // 6)][:6]
    shades = np.linspace(0.25, 1.15, 24)
    swatch = np.array([[np.clip(np.array(b.rgba[:3]) * s, 0, 1) for s in shades] for b in BRICKS])
    swatch = Image.fromarray((swatch * 255).astype(np.uint8)).resize((w, h // 2), Image.Resampling.NEAREST)
    src = Image.new("RGB", (w, h * len(picks) + swatch.height))
    for i, f in enumerate(picks):
        src.paste(f, (0, i * h))
    src.paste(swatch, (0, h * len(picks)))
    return src.quantize(colors=colors)


def demo_gif(cfg, path: Path | None, size=(640, 400), every=5) -> bool:
    env = BimanualEnv(cfg)
    env.reset(0)
    th = env.table_height
    for name, (x, y, yaw) in DEMO_LAYOUT.items():
        set_brick_pose(env.data, name, [x, y, th], yaw)
    mujoco.mj_forward(env.model, env.data)
    ik = BimanualIK(cfg.sim, cfg.ik, env.control_dt)
    keys = tower_keys(env, ik)
    times = np.array([k[0] for k in keys])

    def interp(t, i):
        j = min(np.searchsorted(times, t, side="right"), len(keys) - 1)
        k0, k1 = keys[j - 1], keys[j]
        s = np.clip((t - k0[0]) / max(k1[0] - k0[0], 1e-6), 0, 1)
        s = s * s * (3 - 2 * s)  # smoothstep
        if i >= 3:
            return k0[i] + s * (k1[i] - k0[i])
        (p0, q0), (p1, q1) = k0[i], k1[i]
        q = (1 - s) * (q0 if np.dot(q0, q1) >= 0 else -q0) + s * q1  # nlerp; keyframes are close
        return p0 + s * (p1 - p0), q / np.linalg.norm(q)

    renderer = mujoco.Renderer(env.model, height=size[1], width=size[0]) if path else None
    cam = free_camera(distance=1.0, elevation=-38.0, lookat=(0.4, 0.0, 0.85))
    frames = []
    best = 0
    for step in range(int(times[-1] / env.control_dt)):
        t = step * env.control_dt
        for i, a in ((1, "l"), (2, "r")):
            ik.set_target(a, *interp(t, i))
        ik.follow(env.data.qpos[env.arm_qadr])
        env.apply_action(ik.solve(), np.array([interp(t, 3), interp(t, 4)]))
        env.step()
        best = max(best, env.progress())
        if renderer and step % every == 0:
            renderer.update_scene(env.data, camera=cam)
            frames.append(Image.fromarray(renderer.render()))
    print(f"demo progress: {env.progress()}/{LEVELS} (best {best}), success: {env.success()}")
    if renderer:
        renderer.close()
        # One shared palette + a static camera lets the GIF encoder store only what changed per frame.
        palette = gif_palette(frames)
        pal = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
        pal[0].save(path, save_all=True, append_images=pal[1:], duration=int(1000 * env.control_dt * every),
                    loop=0, optimize=True)
    return env.success()


def tower_png(cfg, path: Path, size=(480, 360)) -> None:
    """Three close-ups: the bricks, a brick's underside, and the finished tower."""
    env = BimanualEnv(cfg)
    env.reset(0)
    th, d = env.table_height, env.data
    x0 = 0.75  # near the far table edge, away from the arms
    renderer = mujoco.Renderer(env.model, height=size[1], width=size[0])

    def shot(poses: dict, cam: mujoco.MjvCamera, flip: str | None = None) -> Image.Image:
        for i, b in enumerate(BRICKS):  # unused bricks go out of view
            pos, yaw = poses.get(b.name, ([0.25, 0.6 - 0.1 * i, th], 0.0))
            set_brick_pose(d, b.name, pos, yaw)
        if flip:
            d.joint(f"{flip}_free").qpos[3:] = [0, 1, 0, 0]  # upside down (180 deg about x)
        mujoco.mj_forward(env.model, d)
        renderer.update_scene(d, camera=cam)
        return Image.fromarray(renderer.render())

    ys = {"red_2x4": 0.075, "blue_2x3": 0.03, "yellow_2x2": -0.015, "green_1x2x2": -0.05}
    panels = [
        label(shot({n: ([x0, y, th], 0.0) for n, y in ys.items()},
                   free_camera(0.24, 160, -32, (x0, 0.012, th + 0.01))), "The 4 bricks (real DUPLO size)"),
        label(shot({"red_2x4": ([x0, 0.02, th + BRICK_H + STUD_H], 0.0), "yellow_2x2": ([x0, -0.045, th], 0.0)},
                   free_camera(0.2, 160, -55, (x0, -0.005, th + 0.01)), flip="red_2x4"),
              "Underside: walls + tubes hold the studs"),
        label(shot({b.name: (p, yaw) for b, (p, yaw) in zip(BRICKS, tower_poses([x0, 0.0, th], 0.5))},
                   free_camera(0.26, 150, -22, (x0, 0.0, th + 0.035))), "Goal: the same tower every time"),
    ]
    renderer.close()
    sheet = Image.new("RGB", (size[0] * len(panels), size[1]))
    for i, img in enumerate(panels):
        sheet.paste(img, (i * size[0], 0))
    sheet.save(path, optimize=True)


def cameras_png(cfg, path: Path, size=(320, 240)) -> None:
    env = BimanualEnv(cfg)
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
    tower_png(cfg, OUT / "tower.png")
    cameras_png(cfg, OUT / "cameras.png")
    for p in sorted(OUT.iterdir()):
        print(f"{p.name:<16} {p.stat().st_size / 1e6:.2f} MB")
