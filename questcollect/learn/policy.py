"""A trained LeRobot policy driving the simulated arms: BimanualEnv -> observation -> policy -> joint targets.

The observation is built with the FeatureSpec of the dataset the policy was trained on (from the run manifest),
so cameras, image size and state layout match training exactly. Used by `qc eval` (evaluate.py) and live by
the runtime (`qc serve --policy`).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from questcollect.config import Config
from questcollect.ik import BimanualIK
from questcollect.learn.features import FeatureSpec
from questcollect.learn.train import checkpoint_dir, run_dir, run_manifest
from questcollect.sim import ARMS, INSTRUCTION, BimanualEnv, CameraRenderer

log = logging.getLogger(__name__)


def spec_of(checkpoint: Path) -> FeatureSpec:
    """The FeatureSpec a checkpoint was trained with (from its `qc train` run)."""
    return FeatureSpec.from_dict(run_manifest(checkpoint)["dataset_manifest"]["features"])


class PolicyRunner:
    def __init__(self, run: Path, cfg: Config, n_action_steps: int | None = None,
                 temporal_ensemble: float | None = None, device: str | None = None):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.path = checkpoint_dir(run)
        self.run = run_dir(self.path)
        self.name = self.run.name
        self.spec = spec_of(self.path)
        self._last_batch = None
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        pcfg = PreTrainedConfig.from_pretrained(self.path)
        pcfg.device = device
        if temporal_ensemble is not None:  # ACT: query every step, average the overlapping chunks
            pcfg.temporal_ensemble_coeff, pcfg.n_action_steps = temporal_ensemble, 1
        elif n_action_steps is not None:
            pcfg.n_action_steps = min(n_action_steps, getattr(pcfg, "chunk_size", n_action_steps))
        self.policy = get_policy_class(pcfg.type).from_pretrained(self.path, config=pcfg).to(device).eval()
        self.pre, self.post = make_pre_post_processors(pcfg, pretrained_path=str(self.path),
                                                       preprocessor_overrides={"device_processor": {"device": device}})
        self.device = device
        self.cfg = cfg
        self._renderer: CameraRenderer | None = None
        self._renderer_model = None
        self.ik = BimanualIK(cfg.sim, cfg.ik, 1.0 / cfg.sim.control_hz) if self.spec.action == "ee" else None
        log.info("policy %s (%s) on %s, cameras %s at %dx%d", self.path, pcfg.type, device,
                 list(self.spec.cameras), self.spec.height, self.spec.width)

    def reset(self, env: BimanualEnv | None = None) -> None:
        self.policy.reset()
        self._last_batch = None
        if self.ik is not None and env is not None:
            self.ik.reset(env.data.qpos[env.arm_qadr])

    def observation(self, env: BimanualEnv) -> dict:
        if self._renderer_model is not env.model:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = CameraRenderer(env.model, self.spec.height, self.spec.width, list(self.spec.cameras))
            self._renderer_model = env.model
        obs = env.observe()
        batch = {"observation.state": torch.from_numpy(self.spec.state(obs).astype(np.float32))[None],
                 "task": [INSTRUCTION]}
        if self.spec.env_state:
            batch["observation.environment_state"] = torch.from_numpy(
                self.spec.environment_state(obs).astype(np.float32))[None]
        for cam, img in self._renderer.render(env.data).items():
            batch[f"observation.images.{cam}"] = torch.from_numpy(img).permute(2, 0, 1).float()[None] / 255.0
        return batch

    @torch.inference_mode()
    def raw_action(self, env: BimanualEnv) -> np.ndarray:
        """The policy's next action, in the dataset's action layout (FeatureSpec.names)."""
        if getattr(self.policy, "_action_queue", None) and self._last_batch is not None:
            batch = self._last_batch  # still executing the last predicted chunk: no need to render the cameras
        else:
            batch = self._last_batch = self.pre(self.observation(env))
        action = self.post(self.policy.select_action(batch))
        return action[0].float().cpu().numpy()

    def act(self, env: BimanualEnv) -> tuple[np.ndarray, np.ndarray]:
        """(14,) joint targets [left 7, right 7] and (2,) grippers in [0 closed, 1 open] for this control step."""
        arm, grippers = self.spec.split_action(self.raw_action(env))
        if self.ik is not None:  # end-effector policy: TCP targets -> joints with the teleop IK
            for a, pose in zip(ARMS, arm):
                q = pose[3:] / np.linalg.norm(pose[3:])
                self.ik.set_target(a, pose[:3], q)
            self.ik.follow(env.data.qpos[env.arm_qadr])
            arm = self.ik.solve()
        return arm, np.clip(grippers, 0.0, 1.0)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
