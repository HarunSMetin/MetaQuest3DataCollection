"""Training: a named preset + a dataset -> one `lerobot-train` run, tied to the dataset it used.

Run folder layout (outputs/<run>/):
  questcollect_run.json        preset, dataset + its manifest, git commit, the exact command
  train/                       lerobot-train's output_dir (checkpoints/<step>/pretrained_model, checkpoints/last)
  eval/<timestamp>/            `qc eval` results (evaluate.py)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from questcollect.config import Config
from questcollect.learn.export import git_commit, read_manifest

RUN_MANIFEST = "questcollect_run.json"
CONSTANT_RANGE = 1e-4  # an input whose whole range in the dataset is below this is treated as constant


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


def list_runs(runs_dir: Path) -> list[dict]:
    """Every `qc train` run in a folder, newest first, for the dashboard's Policy tab: its checkpoints (those with a
    pretrained_model, oldest first, `last` = which step it points to), what it was trained on, and its evaluations
    (newest first, from eval/*/metrics.json). Reads only small JSON files: no torch."""
    out = []
    for run in Path(runs_dir).glob(f"*/{RUN_MANIFEST}"):
        run = run.parent
        try:
            man = json.loads((run / RUN_MANIFEST).read_text())
        except (OSError, ValueError):
            continue
        ck = run / "train" / "checkpoints"
        steps = sorted(d.name for d in ck.iterdir() if d.name != "last" and (d / "pretrained_model").is_dir()) \
            if ck.is_dir() else []
        last = (ck / "last").resolve().name if (ck / "last" / "pretrained_model").is_dir() else None
        kind = None
        if steps:
            try:
                kind = json.loads((ck / steps[-1] / "pretrained_model" / "config.json").read_text()).get("type")
            except (OSError, ValueError):
                pass
        evals = []
        for m in sorted(run.glob("eval/*/metrics.json"), reverse=True):
            try:
                e = json.loads(m.read_text())
            except (OSError, ValueError):
                continue
            s = e.get("summary", {})
            evals.append({"time": m.parent.name, "checkpoint": Path(e.get("checkpoint", "")).parent.name or None,
                          "n": s.get("rollouts"), "success_rate": s.get("success_rate"),
                          "stage_rates": s.get("stage_rates", {}), "n_action_steps": e.get("settings", {}).get("n_action_steps")})
        feats = man.get("dataset_manifest", {}).get("features", {})
        out.append({"name": run.name, "path": str(run), "created_at": man.get("created_at"),
                    "preset": man.get("preset"), "dataset": man.get("dataset"), "type": kind,
                    "features": feats, "steps": steps, "last": last, "evals": evals})
    return sorted(out, key=lambda r: r["created_at"] or "", reverse=True)


def constant_inputs(data_root: Path) -> dict[str, list[int]]:
    """Per non-image policy feature, the indices whose range over the whole dataset is (almost) zero.

    LeRobot normalizes with (x - min) / (max - min + eps) or (x - mean) / (std + eps), so such an input is divided by
    ~0: whatever it does outside the training frames comes out huge. Seen with the brick poses of the state oracles
    (2026-11-13, experiments/REVIEW_2026-11-12.md V9): bricks the takes never touch have a constant z in the data,
    but settle by a few mm at the start of an eval episode, which put 11 and 23 into the network instead of -1.
    """
    stats = json.loads((Path(data_root) / "meta" / "stats.json").read_text())
    out = {}
    for key in ("observation.state", "observation.environment_state", "action"):
        if key in stats:
            lo, hi = stats[key]["min"], stats[key]["max"]
            idx = [i for i, (a, b) in enumerate(zip(lo, hi)) if b - a < CONSTANT_RANGE]
            if idx:
                out[key] = idx
    return out


def train(cfg: Config, preset: str, dataset: str, name: str | None = None, extra: list[str] | None = None,
          dry_run: bool = False, allow_constant_inputs: bool = False) -> Path:
    """Start lerobot-train for `dataset` (a folder name in learn.datasets_dir) with a preset; returns the run.

    Refuses a dataset with constant inputs (`constant_inputs`) unless `allow_constant_inputs`."""
    if preset not in cfg.learn.presets:
        raise KeyError(f"unknown preset {preset!r}; config learn.presets has {sorted(cfg.learn.presets)}")
    data_root = cfg.datasets_dir / dataset
    constant = constant_inputs(data_root)
    if constant and not allow_constant_inputs:
        raise ValueError(f"{dataset}: inputs constant over the whole dataset (range < {CONSTANT_RANGE}): {constant}. "
                         "Normalization divides them by ~0, so any change at eval time explodes. Export without "
                         "them, or pass allow_constant_inputs (qc train --allow-constant-inputs) and keep them "
                         "constant at eval time (experiments/obs_ablation/diagnostics/e8_eval_pinned.py).")
    manifest = read_manifest(data_root)
    name = name or f"{datetime.now():%Y%m%d-%H%M}_{preset}_{dataset}"
    run = cfg.runs_dir / name
    if run.exists():
        raise FileExistsError(f"run {run} already exists")
    args = dict(cfg.learn.presets[preset])
    ema = args.pop("ema", None)  # ours, not lerobot-train's: max decay of a weight EMA (lerobot_train_ema.py)
    exe = Path(sys.executable).parent / "lerobot-train"  # installed with the `learn` extra, next to python
    cmd = [sys.executable, "-m", "questcollect.learn.lerobot_train_ema"] if ema else [str(exe)]
    argv = [*cmd, _arg("dataset.repo_id", manifest["repo_id"]), _arg("dataset.root", data_root),
            _arg("output_dir", run / "train"), _arg("job_name", name),
            *(_arg(k, v) for k, v in args.items()), *(extra or [])]
    if dry_run:
        print(" ".join(argv))
        return run
    run.mkdir(parents=True)
    (run / RUN_MANIFEST).write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit(),
        "preset": preset, "dataset": dataset, "dataset_root": str(data_root), "dataset_manifest": manifest,
        "command": argv}, indent=1))
    env = {**os.environ, "QC_EMA_MAX_DECAY": str(ema)} if ema else None
    subprocess.run(argv, check=True, env=env)
    return run
