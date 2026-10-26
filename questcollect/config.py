"""Typed configuration loaded from config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


class SimConfig(BaseModel):
    physics_dt: float = 0.002
    control_hz: int = 50
    table_height: float = 0.75
    arm_spacing: float = 0.6
    gripper_force: float = 40.0  # N squeeze when closed on an object (real Franka Hand: <= 70 N)


class TeleopConfig(BaseModel):
    mode: str = "controllers"
    scale: float = 1.0
    pinch_open_m: float = 0.08
    pinch_closed_m: float = 0.015
    filter_min_cutoff: float = 1.5
    filter_beta: float = 0.5
    workspace_min: list[float] = Field(default_factory=lambda: [0.05, -0.65, 0.755])
    workspace_max: list[float] = Field(default_factory=lambda: [0.95, 0.65, 1.45])


class IKConfig(BaseModel):
    position_cost: float = 1.0
    orientation_cost: float = 0.6
    posture_cost: float = 0.01
    max_joint_vel: float = 2.5
    max_lead: float = 0.15
    avoid_arm_collisions: bool = False
    collision_margin: float = 0.03
    iterations: int = 2


class RecordConfig(BaseModel):
    out_dir: str = "data/episodes"
    countdown_s: float = 3
    cameras: list[str] = Field(
        default_factory=lambda: ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    )
    operator: str = "operator"
    action_noise: float = 0.0  # DART: std (rad) of noise on the executed joint targets; the clean target is recorded


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    https_port: int = 8443
    cert_dir: str = "certs"


class CurateConfig(BaseModel):
    """Which recorded steps go into a training dataset (audit.select)."""
    include_failures: bool = False
    max_link_table_frac: float = 0.01  # arm links (not the hand) resting on the table
    max_arm_arm_frac: float = 0.0
    max_tip_table_frac: float = 0.3    # hand/fingertips on the table: some is normal, bricks are 19 mm tall
    trim_idle: bool = True             # drop the still start and end of each episode
    dagger_frames: Literal["operator", "all"] = "operator"  # shared-control takes: only the operator's steps


class LearnConfig(BaseModel):
    """Policy observations/actions (learn/features.py), and where datasets and training runs go."""
    cameras: list[str] = Field(default_factory=lambda: ["cam_high", "cam_left_wrist", "cam_right_wrist"])
    image_size: list[int] = Field(default_factory=lambda: [240, 320])  # height, width
    action: Literal["joint", "ee"] = "joint"
    env_state: bool = False            # also export the brick poses (privileged: sim only)
    datasets_dir: str = "data/lerobot"
    runs_dir: str = "outputs"
    presets: dict[str, dict[str, Any]] = Field(default_factory=lambda: {
        # Measured on an RTX 4050 6 GB: ~3.6 GB, ~54 samples/s, ~8 h for 100k steps.
        "act_240": {"policy.type": "act", "policy.device": "cuda", "policy.use_amp": True,
                    "policy.chunk_size": 100, "policy.n_action_steps": 100, "policy.push_to_hub": False,
                    "batch_size": 16, "num_workers": 8, "steps": 100_000, "save_freq": 20_000},
    })


class SafetyConfig(BaseModel):
    """Filter on policy joint targets before they reach the arms (safety.py)."""
    enabled: bool = True
    arm_margin: float = 0.02           # min distance between the two arms (m)
    table_margin: float = 0.01         # min distance between arm links and the table top (m)


class ControlConfig(BaseModel):
    """Who drives the arms in `qc serve`: the operator, a trained policy, or both (DAgger)."""
    mode: Literal["teleop", "policy", "shared"] = "teleop"
    policy: str | None = None          # training run folder (outputs/<run>) or checkpoint folder
    n_action_steps: int = 10           # live: steps executed from each predicted chunk before asking again


class Config(BaseModel):
    sim: SimConfig = Field(default_factory=SimConfig)
    teleop: TeleopConfig = Field(default_factory=TeleopConfig)
    ik: IKConfig = Field(default_factory=IKConfig)
    record: RecordConfig = Field(default_factory=RecordConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    curate: CurateConfig = Field(default_factory=CurateConfig)
    learn: LearnConfig = Field(default_factory=LearnConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)

    @property
    def out_dir(self) -> Path:
        return _path(self.record.out_dir)

    @property
    def datasets_dir(self) -> Path:
        return _path(self.learn.datasets_dir)

    @property
    def runs_dir(self) -> Path:
        return _path(self.learn.runs_dir)


def _path(p: str) -> Path:
    """Config paths are relative to the repository root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        return Config.model_validate(yaml.safe_load(f) or {})
