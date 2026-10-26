"""Training: a named preset + a dataset -> one `lerobot-train` run, tied to the dataset it used.

Run folder layout (outputs/<run>/):
  questcollect_run.json        preset, dataset + its manifest, git commit, the exact command
  train/                       lerobot-train's output_dir (checkpoints/<step>/pretrained_model, checkpoints/last)
  eval/<timestamp>/            `qc eval` results (evaluate.py)
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from questcollect.config import Config
from questcollect.learn.export import git_commit, read_manifest

RUN_MANIFEST = "questcollect_run.json"


def _arg(key: str, value) -> str:
    if isinstance(value, bool):
        value = str(value).lower()
    elif value is None:
        value = "null"
    return f"--{key}={value}"


def checkpoint_dir(run: Path, step: str = "last") -> Path:
    """The pretrained_model folder of a run (or a checkpoint folder given directly)."""
    run = Path(run)
    if (run / "config.json").exists():
        return run
    path = run / "train" / "checkpoints" / step / "pretrained_model"
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint {step!r} in {run} (expected {path})")
    return path


def run_dir(path: Path) -> Path:
    """The `qc train` run folder that contains `path` (a run folder or any checkpoint inside it)."""
    for p in (Path(path), *Path(path).parents):
        if (p / RUN_MANIFEST).exists():
            return p
    raise FileNotFoundError(f"{path} is not inside a `qc train` run (no {RUN_MANIFEST})")


def run_manifest(path: Path) -> dict:
    return json.loads((run_dir(path) / RUN_MANIFEST).read_text())


def train(cfg: Config, preset: str, dataset: str, name: str | None = None, extra: list[str] | None = None,
          dry_run: bool = False) -> Path:
    """Start lerobot-train for `dataset` (a folder name in learn.datasets_dir) with a preset; returns the run."""
    if preset not in cfg.learn.presets:
        raise KeyError(f"unknown preset {preset!r}; config learn.presets has {sorted(cfg.learn.presets)}")
    data_root = cfg.datasets_dir / dataset
    manifest = read_manifest(data_root)
    name = name or f"{datetime.now():%Y%m%d-%H%M}_{preset}_{dataset}"
    run = cfg.runs_dir / name
    if run.exists():
        raise FileExistsError(f"run {run} already exists")
    exe = Path(sys.executable).parent / "lerobot-train"  # installed with the `learn` extra, next to python
    argv = [str(exe), _arg("dataset.repo_id", manifest["repo_id"]), _arg("dataset.root", data_root),
            _arg("output_dir", run / "train"), _arg("job_name", name),
            *(_arg(k, v) for k, v in cfg.learn.presets[preset].items()), *(extra or [])]
    if dry_run:
        print(" ".join(argv))
        return run
    run.mkdir(parents=True)
    (run / RUN_MANIFEST).write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit(),
        "preset": preset, "dataset": dataset, "dataset_root": str(data_root), "dataset_manifest": manifest,
        "command": argv}, indent=1))
    subprocess.run(argv, check=True)
    return run
