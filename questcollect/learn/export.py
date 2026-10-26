"""Selected episode segments -> a LeRobot dataset (one folder per dataset, with a manifest).

Camera frames are rendered now from the recorded sim states, in the scene each episode was recorded in, so any
resolution works and recording stays light. All LeRobot dataset API usage lives in this file.
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from questcollect.audit import Segment
from questcollect.config import ROOT
from questcollect.learn.features import ROBOT_TYPE, FeatureSpec, episode_obs
from questcollect.recorder import read_episode
from questcollect.sim import AppearanceRandomizer, CameraRenderer, env_from_episode

log = logging.getLogger(__name__)
MANIFEST = "questcollect_manifest.json"


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_manifest(dataset_root: Path) -> dict:
    return json.loads((Path(dataset_root) / MANIFEST).read_text())


def export(segments: list[Segment], out: Path, spec: FeatureSpec, repo_id: str, randomize_visuals: bool = False,
           extra_manifest: dict | None = None, progress=None) -> int:
    """Write the segments as LeRobot episodes into `out` (must not exist yet). Returns the number of episodes."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} already exists; pick another name or remove it first")
    if not segments:
        raise ValueError("no episodes selected (see `qc audit`)")
    scenes: dict[str, tuple] = {}  # config -> (env, renderer, randomizer): one scene per recording setup
    ds = None
    written = []
    loaded: tuple[Path, dict, dict] | None = None
    for k, seg in enumerate(segments):
        if loaded is None or loaded[0] != seg.path:  # DAgger takes give several segments per file, in order
            loaded = (seg.path, *read_episode(seg.path))
        _, meta, data = loaded
        key = meta.get("config_yaml", "")
        if key not in scenes:
            env = env_from_episode(meta)
            scenes[key] = (env, CameraRenderer(env.model, spec.height, spec.width, list(spec.cameras)),
                           AppearanceRandomizer(env.model) if randomize_visuals else None)
        env, renderer, looks = scenes[key]
        if ds is None:
            ds = LeRobotDataset.create(repo_id=repo_id, fps=int(meta["fps"]), features=spec.lerobot_features(),
                                       root=out, robot_type=ROBOT_TYPE, use_videos=True, image_writer_threads=4)
        if looks is not None:
            looks.apply(np.random.default_rng([k, int(meta.get("seed", 0))]))
        states = data["sim/state"]
        if states.shape[1] != env.state_size:
            raise RuntimeError(f"{seg.path.name}: recorded state size {states.shape[1]} != scene {env.state_size}")
        obs = episode_obs(data)
        state, action = spec.state(obs), spec.recorded_action(data)
        env_state = spec.environment_state(obs) if spec.env_state else None
        for t in range(seg.start, seg.stop):
            env.set_state(states[t])
            frame = {"observation.state": state[t].astype(np.float32), "action": action[t].astype(np.float32),
                     "task": meta["instruction"]}
            if env_state is not None:
                frame["observation.environment_state"] = env_state[t].astype(np.float32)
            for cam, img in renderer.render(env.data).items():
                frame[f"observation.images.{cam}"] = img
            ds.add_frame(frame)
        ds.save_episode()
        written.append({"file": seg.path.name, "start": seg.start, "stop": seg.stop,
                        "scene_hash": env.scene_hash, "schema_version": meta.get("schema_version")})
        if progress:
            progress(k + 1, len(segments), seg)
    ds.finalize()
    for _, renderer, _ in scenes.values():
        renderer.close()
    manifest = {"created_at": datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit(),
                "repo_id": repo_id, "features": spec.to_dict(), "randomize_visuals": randomize_visuals,
                "source_dir": str(segments[0].path.parent), "segments": written, **(extra_manifest or {})}
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1))
    return len(written)
