"""Typed configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


class SimConfig(BaseModel):
    physics_dt: float = 0.002
    control_hz: int = 50
    publish_hz: int = 60
    table_height: float = 0.75
    arm_spacing: float = 0.6


class TeleopConfig(BaseModel):
    mode: str = "controllers"
    scale: float = 1.0
    pinch_open_m: float = 0.08
    pinch_closed_m: float = 0.015
    filter_min_cutoff: float = 1.5
    filter_beta: float = 0.5
    workspace_min: list[float] = Field(default_factory=lambda: [0.05, -0.65, 0.76])
    workspace_max: list[float] = Field(default_factory=lambda: [0.95, 0.65, 1.45])


class IKConfig(BaseModel):
    position_cost: float = 1.0
    orientation_cost: float = 0.6
    posture_cost: float = 0.01
    max_joint_vel: float = 2.5
    collision_margin: float = 0.03
    iterations: int = 2


class RecordConfig(BaseModel):
    out_dir: str = "data/raw"
    countdown_s: float = 3
    cameras: list[str] = Field(
        default_factory=lambda: ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    )
    live_images: bool = False
    image_height: int = 480
    image_width: int = 640
    operator: str = "operator"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    https_port: int = 8443
    cert_dir: str = "certs"


class Config(BaseModel):
    sim: SimConfig = Field(default_factory=SimConfig)
    task: str = "transfer_cube"
    teleop: TeleopConfig = Field(default_factory=TeleopConfig)
    ik: IKConfig = Field(default_factory=IKConfig)
    record: RecordConfig = Field(default_factory=RecordConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @property
    def out_dir(self) -> Path:
        p = Path(self.record.out_dir)
        return p if p.is_absolute() else ROOT / p


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        return Config.model_validate(yaml.safe_load(f) or {})
