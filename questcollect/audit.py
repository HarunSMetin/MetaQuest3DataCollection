"""Episode quality audit and dataset selection (the stage between recording and export).

Every episode gets an `EpisodeAudit`: how often the arms touched the table or each other, how much of it is
still, when each tower level clicked in, and who drove (operator or policy). `select` turns the audits, the
`curate` rules from config.yaml and the manual excludes in curation.json into the segments a training dataset
is built from, with a reason for every rejected episode.

Raw episodes are never modified: audits are cached in <episodes>/audit.json, manual decisions live in
<episodes>/curation.json ({"episode_000001.hdf5": {"exclude": true, "note": "..."}}).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from questcollect.config import CurateConfig
from questcollect.recorder import SOURCE_OPERATOR, episode_files, read_episode
from questcollect.sim import LEVELS

AUDIT_VERSION = 2
CACHE_NAME = "audit.json"
CURATION_NAME = "curation.json"
CONTACT_KINDS = ("link_table", "arm_arm", "tip_table")  # order of BimanualEnv.contact_flags
IDLE_SPEED = 0.02          # rad/s: every joint slower than this, and nobody engaged = still
MIN_SEGMENT_S = 1.0        # shorter pieces are not worth a training episode
AUDIT_KEYS = ["obs/joint_vel", "teleop/engaged", "progress", "contacts", "control/source"]


@dataclass
class EpisodeAudit:
    file: str
    num_steps: int
    fps: int
    success: bool
    contact_frac: dict[str, float]           # share of steps with each contact kind
    contact_events: dict[str, int]           # separate times each contact kind started
    idle_frac: float
    active: tuple[int, int]                  # [start, stop): first to last step with motion
    level_times: list[float | None]          # seconds until each tower level (1..LEVELS) first clicked in
    operator_runs: list[tuple[int, int]] = field(default_factory=list)  # [start, stop) driven by the operator

    @property
    def duration_s(self) -> float:
        return self.num_steps / self.fps


@dataclass
class Segment:
    """A contiguous piece of an episode that becomes one training episode."""
    path: Path
    start: int
    stop: int

    @property
    def num_steps(self) -> int:
        return self.stop - self.start


# -- audit ------------------------------------------------------------------------------------------
def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) of every run of True."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _replay_contacts(path: Path, meta: dict, envs: dict) -> np.ndarray:
    """Contacts of an episode recorded before schema v5, by replaying its states."""
    from questcollect.sim import env_from_episode  # builds a MuJoCo scene: only when needed

    states = read_episode(path, keys=["sim/state"])[1]["sim/state"]
    key = meta.get("config_yaml", "")
    if key not in envs:
        envs[key] = env_from_episode(meta)
    env = envs[key]
    out = np.empty((len(states), len(CONTACT_KINDS)), bool)
    for t, s in enumerate(states):
        env.set_state(s)
        out[t] = env.contact_flags()
    return out


def audit_episode(path: Path, envs: dict | None = None) -> EpisodeAudit:
    """Audit one episode file (envs: scenes reused between episodes recorded with the same config)."""
    meta, data = read_episode(path, keys=AUDIT_KEYS)
    contacts = data.get("contacts")
    if contacts is None:
        contacts = _replay_contacts(path, meta, {} if envs is None else envs)
    n = len(data["progress"])
    moving = (np.abs(data["obs/joint_vel"]).max(axis=1) > IDLE_SPEED) | data["teleop/engaged"].any(axis=1)
    active = (int(np.argmax(moving)), int(n - np.argmax(moving[::-1]))) if moving.any() else (0, 0)
    source = data.get("control/source", np.full(n, SOURCE_OPERATOR))
    fps = int(meta["fps"])
    progress = data["progress"]
    return EpisodeAudit(
        file=Path(path).name, num_steps=n, fps=fps, success=bool(meta.get("success")),
        contact_frac={k: float(contacts[:, i].mean()) for i, k in enumerate(CONTACT_KINDS)},
        contact_events={k: len(true_runs(contacts[:, i])) for i, k in enumerate(CONTACT_KINDS)},
        idle_frac=float(1.0 - moving.mean()), active=active,
        level_times=[float(np.argmax(progress >= k)) / fps if (progress >= k).any() else None
                     for k in range(1, LEVELS + 1)],
        operator_runs=true_runs(source == SOURCE_OPERATOR),
    )


def audit_dir(episodes_dir: Path, progress=None) -> dict[str, EpisodeAudit]:
    """Audits of every episode in a folder, from the cache where the file hasn't changed since."""
    episodes_dir = Path(episodes_dir)
    cache_path = episodes_dir / CACHE_NAME
    cache = {}
    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
        if raw.get("version") == AUDIT_VERSION:
            cache = raw["episodes"]
    envs: dict = {}
    out, fresh = {}, {}
    for path in episode_files(episodes_dir):
        st = path.stat()
        key = [st.st_size, st.st_mtime]
        entry = cache.get(path.name)
        if entry and entry["key"] == key:
            a = EpisodeAudit(**entry["audit"])
            a.active = tuple(a.active)
            a.operator_runs = [tuple(r) for r in a.operator_runs]
        else:
            if progress:
                progress(path.name)
            a = audit_episode(path, envs)
        out[path.name] = a
        fresh[path.name] = {"key": key, "audit": asdict(a)}
    if json.loads(json.dumps(fresh)) != cache:  # compared as JSON: tuples come back from the file as lists
        cache_path.write_text(json.dumps({"version": AUDIT_VERSION, "episodes": fresh}, indent=1))
    return out


# -- curation ---------------------------------------------------------------------------------------
def load_curation(episodes_dir: Path) -> dict[str, dict]:
    p = Path(episodes_dir) / CURATION_NAME
    return json.loads(p.read_text()) if p.exists() else {}


def set_curation(episodes_dir: Path, file: str, exclude: bool, note: str = "") -> None:
    """Exclude an episode from training datasets by hand (or include it again)."""
    cur = load_curation(episodes_dir)
    if exclude:
        cur[file] = {"exclude": True, "note": note}
    else:
        cur.pop(file, None)
    (Path(episodes_dir) / CURATION_NAME).write_text(json.dumps(cur, indent=1, sort_keys=True))


def verdict(a: EpisodeAudit, rules: CurateConfig, curation: dict) -> str | None:
    """Why an episode is left out of training datasets, or None if it is used."""
    if curation.get(a.file, {}).get("exclude"):
        return "excluded by hand" + (f": {curation[a.file]['note']}" if curation[a.file].get("note") else "")
    if not a.success and not rules.include_failures:
        return "failed take"
    limits = {"link_table": (rules.max_link_table_frac, "arm link on the table"),
              "arm_arm": (rules.max_arm_arm_frac, "arms touching"),
              "tip_table": (rules.max_tip_table_frac, "hand dragging on the table")}
    for kind, (limit, what) in limits.items():
        if a.contact_frac[kind] > limit:
            return f"{what} {a.contact_frac[kind]:.1%} of the time (limit {limit:.1%})"
    return None


def select(episodes_dir: Path, audits: dict[str, EpisodeAudit], rules: CurateConfig,
           curation: dict | None = None) -> tuple[list[Segment], dict[str, str]]:
    """Segments to export, and the reason each rejected episode was left out."""
    curation = load_curation(episodes_dir) if curation is None else curation
    segments, rejected = [], {}
    for name, a in sorted(audits.items()):
        why = verdict(a, rules, curation)
        if why:
            rejected[name] = why
            continue
        lo, hi = a.active if rules.trim_idle else (0, a.num_steps)
        runs = a.operator_runs if rules.dagger_frames == "operator" else [(0, a.num_steps)]
        pieces = [(max(s, lo), min(e, hi)) for s, e in runs]
        pieces = [Segment(Path(episodes_dir) / name, s, e) for s, e in pieces if e - s >= MIN_SEGMENT_S * a.fps]
        if not pieces:
            rejected[name] = "nothing left after trimming"
            continue
        segments += pieces
    return segments, rejected
