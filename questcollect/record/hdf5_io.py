"""Read/write raw episodes. Steps are nested dicts of arrays; nesting becomes HDF5 groups."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def _flatten(d: dict, prefix: str = "") -> dict[str, object]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def write_episode(path: Path, steps: list[dict], meta: dict, static: dict | None = None) -> Path:
    """steps: list of per-step nested dicts with identical structure. static: non-time arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    flat = [_flatten(s) for s in steps]
    with h5py.File(tmp, "w") as f:
        for k, v in meta.items():
            f.attrs[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
        for key in flat[0]:
            arr = np.stack([np.asarray(s[key]) for s in flat])
            if key.startswith("images/"):
                f.create_dataset(key, data=arr, chunks=(1, *arr.shape[1:]), compression="gzip", compression_opts=4)
            else:
                f.create_dataset(key, data=arr, compression="lzf" if arr.size > 4096 else None)
        for key, v in _flatten(static or {}).items():
            f.create_dataset(key, data=np.asarray(v))
    tmp.rename(path)
    return path


def read_meta(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        return {k: _decode(v) for k, v in f.attrs.items()}


def read_episode(path: Path, keys: list[str] | None = None) -> tuple[dict, dict]:
    """Returns (meta, data) with data as a flat dict path -> ndarray."""
    data = {}
    with h5py.File(path, "r") as f:
        meta = {k: _decode(v) for k, v in f.attrs.items()}

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and (keys is None or name in keys):
                data[name] = obj[()]

        f.visititems(visit)
    return meta, data


def _decode(v):
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    if isinstance(v, np.generic):
        return v.item()
    return v


def next_episode_path(out_dir: Path, task: str) -> Path:
    d = Path(out_dir) / task
    d.mkdir(parents=True, exist_ok=True)
    nums = [int(p.stem.split("_")[-1]) for p in d.glob("episode_*.hdf5") if p.stem.split("_")[-1].isdigit()]
    return d / f"episode_{(max(nums) + 1 if nums else 0):06d}.hdf5"


def list_episodes(out_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(Path(out_dir).glob("*/episode_*.hdf5")):
        try:
            m = read_meta(p)
        except OSError:
            continue
        rows.append({
            "path": str(p), "file": p.name, "task": m.get("task"), "success": bool(m.get("success")),
            "duration_s": m.get("duration_s"), "num_steps": m.get("num_steps"),
            "input_mode": m.get("input_mode"), "operator": m.get("operator"), "created_at": m.get("created_at"),
        })
    return rows
