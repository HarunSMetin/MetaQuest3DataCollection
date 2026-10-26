"""`qc` command line interface."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

import typer

from questcollect.config import ROOT, load_config

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Quest 3 -> MuJoCo bimanual teleop & data collection")


def _headless_gl() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no packets are sent; selects the outbound interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


@app.command()
def serve(
    config: Optional[Path] = typer.Option(None, help="YAML config (default: config.yaml)"),
    mode: Optional[str] = typer.Option(None, help="controllers | hands"),
    https: bool = typer.Option(True, help="Also serve HTTPS for Wi-Fi (needs `qc certs`)"),
    policy: Optional[Path] = typer.Option(None, help="Trained policy run (outputs/<run>) to drive the arms; "
                                                     "needs `uv sync --extra learn`"),
    control: Optional[str] = typer.Option(None, help="teleop | policy | shared (default with --policy: shared)"),
):
    """Run the sim, teleop server, and web clients."""
    _headless_gl()
    import uvicorn

    from questcollect.runtime import Runtime
    from questcollect.server import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(config)
    if policy:
        cfg.control.policy = str(policy)
        cfg.control.mode = control or ("shared" if cfg.control.mode == "teleop" else cfg.control.mode)
    elif control:
        cfg.control.mode = control
    if mode:
        cfg.teleop.mode = mode
    runtime = Runtime(cfg)
    runtime.start()
    web = create_app(runtime)

    servers = [uvicorn.Server(uvicorn.Config(web, host=cfg.server.host, port=cfg.server.port, log_level="warning",
                                             ws_max_size=1 << 20))]
    cert_dir = ROOT / cfg.server.cert_dir
    cert, key = cert_dir / "cert.pem", cert_dir / "key.pem"
    if https and cert.exists() and key.exists():
        servers.append(uvicorn.Server(uvicorn.Config(web, host=cfg.server.host, port=cfg.server.https_port,
                                                     ssl_certfile=str(cert), ssl_keyfile=str(key),
                                                     log_level="warning", ws_max_size=1 << 20)))
    ip = lan_ip()
    typer.secho("\n  QuestCollect is running", fg="cyan", bold=True)
    typer.echo(f"  Dashboard      http://localhost:{cfg.server.port}/dashboard/")
    typer.echo(f"  Quest (USB)    http://localhost:{cfg.server.port}/xr/     (after `qc setup-quest`)")
    if len(servers) > 1:
        typer.echo(f"  Quest (Wi-Fi)  https://{ip}:{cfg.server.https_port}/xr/")
    elif https:
        typer.echo("  Wi-Fi HTTPS disabled: run `qc certs` first")
    typer.echo("")

    async def main():
        await asyncio.gather(*(s.serve() for s in servers))

    try:
        asyncio.run(main())
    finally:
        runtime.stop()


@app.command()
def certs(ip: Optional[str] = typer.Option(None, help="LAN IP to put in the certificate")):
    """Create a self-signed TLS certificate for Wi-Fi WebXR (Quest needs HTTPS off-localhost)."""
    if not shutil.which("openssl"):
        raise typer.Exit("openssl not found (sudo apt install openssl)")
    cfg = load_config()
    ip = ip or lan_ip()
    d = ROOT / cfg.server.cert_dir
    d.mkdir(exist_ok=True)
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "825",
        "-keyout", str(d / "key.pem"), "-out", str(d / "cert.pem"), "-subj", "/CN=questcollect",
        "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost",
    ], check=True, capture_output=True)
    typer.secho(f"certificate for {ip} written to {d}", fg="green")
    typer.echo(f"Open https://{ip}:{cfg.server.https_port}/xr/ on the Quest and accept the warning once.")


@app.command("setup-quest")
def setup_quest():
    """Forward the server port to a USB-connected Quest (adb reverse)."""
    adb = shutil.which("adb")
    if not adb:
        typer.secho("adb not found. Install it: sudo apt install android-tools-adb", fg="red")
        raise typer.Exit(1)
    cfg = load_config()
    out = subprocess.run([adb, "devices"], capture_output=True, text=True).stdout
    devices = [line for line in out.splitlines()[1:] if line.strip()]
    if not devices:
        typer.secho("No device. Plug in the Quest, enable developer mode, accept 'Allow USB debugging'.", fg="red")
        raise typer.Exit(1)
    if any("unauthorized" in d for d in devices):
        typer.secho("Device unauthorized: put the headset on and accept the USB debugging prompt.", fg="yellow")
        raise typer.Exit(1)
    port = cfg.server.port
    subprocess.run([adb, "reverse", f"tcp:{port}", f"tcp:{port}"], check=True)
    typer.secho(f"OK: Quest localhost:{port} -> this PC:{port}", fg="green")
    typer.echo(f"In Quest Browser open: http://localhost:{port}/xr/")


@app.command()
def view():
    """Open the scene in MuJoCo's native viewer (sanity check, no headset)."""
    import mujoco.viewer

    from questcollect.sim import BimanualEnv

    env = BimanualEnv(load_config())
    env.reset()
    mujoco.viewer.launch(env.model, env.data)


@app.command()
def replay(path: Path, speed: float = typer.Option(1.0)):
    """Play back a raw episode's recorded states in MuJoCo's native viewer."""
    import mujoco.viewer

    from questcollect.recorder import read_episode
    from questcollect.sim import BimanualEnv

    meta, data = read_episode(path, keys=["sim/state"])
    env = BimanualEnv(load_config())
    states = data["sim/state"]
    dt = 1.0 / meta["fps"] / speed
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        for s in states:
            if not v.is_running():
                break
            env.set_state(s)
            v.sync()
            time.sleep(dt)


@app.command()
def inspect(path: Path):
    """Print an episode's metadata and dataset shapes."""
    from questcollect.recorder import read_episode

    meta, data = read_episode(path)
    for k, v in meta.items():
        if k in ("model_xml", "config_yaml"):
            v = f"<{len(v)} chars>"
        typer.echo(f"{k:>16}: {v}")
    typer.echo("")
    for k in sorted(data):
        typer.echo(f"{k:<32} {str(data[k].shape):<18} {data[k].dtype}")


@app.command()
def audit(
    raw: Optional[Path] = typer.Option(None, help="Episode folder (default: record.out_dir from the config)"),
    config: Optional[Path] = typer.Option(None, help="YAML config (default: config.yaml)"),
    exclude: Optional[str] = typer.Option(None, help="Exclude this episode file from training datasets"),
    include: Optional[str] = typer.Option(None, help="Undo a manual exclude"),
    note: str = typer.Option("", help="Why (with --exclude)"),
):
    """Quality report of every episode (contacts, idle time, timing) and which ones a dataset would use."""
    _headless_gl()
    from questcollect.audit import audit_dir, select, set_curation

    cfg = load_config(config)
    folder = raw or cfg.out_dir
    if exclude or include:
        set_curation(folder, exclude or include, exclude is not None, note)
    audits = audit_dir(folder, progress=lambda name: typer.echo(f"  auditing {name} …"))
    segments, rejected = select(folder, audits, cfg.curate)
    kept = {}
    for s in segments:
        kept[s.path.name] = kept.get(s.path.name, 0) + s.num_steps
    typer.echo(f"\n{'episode':<22}{'length':>8}{'link-table':>12}{'arm-arm':>9}{'hand-table':>12}{'idle':>7}"
               f"   levels at (s)          verdict")
    for name, a in audits.items():
        levels = " ".join("—" if t is None else f"{t:5.1f}" for t in a.level_times)
        c = a.contact_frac
        verdict = "✓ " + f"{kept[name] / a.fps:.0f} s used" if name in kept else "✗ " + rejected[name]
        typer.secho(f"{name:<22}{a.duration_s:7.1f}s{c['link_table']:12.1%}{c['arm_arm']:9.1%}{c['tip_table']:12.1%}"
                    f"{a.idle_frac:7.0%}   {levels:<22} {verdict}", fg=None if name in kept else "yellow")
    fps = next(iter(audits.values())).fps if audits else 50
    total = sum(a.num_steps for a in audits.values()) / fps / 60
    used = sum(kept.values()) / fps / 60
    typer.secho(f"\n{len(kept)} of {len(audits)} episodes used: {used:.1f} of {total:.1f} min "
                f"({len(segments)} segments). Rules: `curate` in the config; manual excludes: --exclude.",
                fg="green")


@app.command()
def export(
    name: str = typer.Argument(..., help="Dataset name: written to <learn.datasets_dir>/<name>"),
    config: Optional[Path] = typer.Option(None, help="YAML config (default: config.yaml)"),
    raw: Optional[Path] = typer.Option(None, help="Episode folder (default: record.out_dir from the config)"),
    repo_id: Optional[str] = typer.Option(None, help="LeRobot repo id (default: local/<name>)"),
    randomize_visuals: bool = typer.Option(False, help="Vary lighting, colours and camera mounts per episode"),
):
    """Selected episodes -> LeRobot dataset. Cameras, image size and action space: `learn` in the config;
    which episodes and steps: `curate` (see `qc audit`). Needs `uv sync --extra learn`."""
    _headless_gl()
    from questcollect.audit import audit_dir, select
    from questcollect.learn.export import export as export_lerobot
    from questcollect.learn.features import FeatureSpec

    cfg = load_config(config)
    folder = raw or cfg.out_dir
    segments, rejected = select(folder, audit_dir(folder), cfg.curate)
    spec = FeatureSpec.from_config(cfg)
    out = cfg.datasets_dir / name
    typer.echo(f"{len(segments)} segments ({sum(s.num_steps for s in segments) / cfg.sim.control_hz / 60:.1f} min), "
               f"{len(rejected)} episodes left out (`qc audit` shows why); cameras {list(spec.cameras)} at "
               f"{spec.height}x{spec.width}, {spec.action} actions")
    n = export_lerobot(segments, out, spec, repo_id or f"local/{name}", randomize_visuals=randomize_visuals,
                       extra_manifest={"curate": cfg.curate.model_dump(), "rejected": rejected},
                       progress=lambda i, total, seg: typer.echo(f"  [{i}/{total}] {seg.path.name} "
                                                                 f"{seg.start}-{seg.stop}"))
    typer.secho(f"exported {n} episodes to {out}", fg="green")


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def train(
    ctx: typer.Context,
    dataset: str = typer.Argument(..., help="Dataset name in learn.datasets_dir (from `qc export`)"),
    preset: str = typer.Option("act_240", help="Preset in learn.presets of the config"),
    name: Optional[str] = typer.Option(None, help="Run name (default: date_preset_dataset)"),
    config: Optional[Path] = typer.Option(None, help="YAML config (default: config.yaml)"),
    dry_run: bool = typer.Option(False, help="Only print the lerobot-train command"),
):
    """Train a policy with lerobot-train. Extra --key=value arguments are passed through to lerobot-train."""
    from questcollect.learn.train import train as train_run

    run = train_run(load_config(config), preset, dataset, name=name, extra=list(ctx.args), dry_run=dry_run)
    if not dry_run:
        typer.secho(f"run finished: {run}  ->  qc eval {run}", fg="green")


@app.command("eval")
def eval_(
    run: Path = typer.Argument(..., help="Run folder (outputs/<run>) or a checkpoint folder"),
    n: int = typer.Option(50, help="Number of rollouts (fixed held-out seeds)"),
    config: Optional[Path] = typer.Option(None, help="YAML config (default: config.yaml)"),
    max_seconds: Optional[float] = typer.Option(None, help="Rollout time limit (default: 1.5 x the 95th "
                                                           "percentile training episode)"),
    safety: bool = typer.Option(True, help="Filter actions with the safety filter (safety in the config)"),
    ensemble: Optional[float] = typer.Option(None, help="ACT temporal ensembling coefficient, e.g. 0.01"),
    n_action_steps: Optional[int] = typer.Option(None, help="Steps executed per predicted chunk"),
    save_rollouts: bool = typer.Option(True, help="Save every rollout as an episode file"),
):
    """Evaluate a trained policy in simulation: success rate, stage reached, contacts, safety interventions."""
    _headless_gl()
    from questcollect.learn.evaluate import evaluate
    from questcollect.sim import LEVELS

    cfg = load_config(config)

    def show(i, total, r):
        typer.echo(f"  [{i}/{total}] seed {r.seed:>10}  {'✓' if r.success else '✗'}  {r.duration_s:6.1f}s  "
                   f"base {'in' if r.base_in_area else 'out'}  levels {r.max_progress}/{LEVELS}  "
                   f"contacts {r.contact_events}  safety max {r.safety_max_rad:.3f} rad")

    max_steps = int(max_seconds * cfg.sim.control_hz) if max_seconds else None
    summary, out = evaluate(run, cfg, n=n, max_steps=max_steps, use_safety=safety, save_rollouts=save_rollouts,
                            temporal_ensemble=ensemble, n_action_steps=n_action_steps, progress=show)
    st = summary["stage_rates"]
    typer.secho(f"\nsuccess {summary['success_rate']:.0%}  |  base in area {st['base_in_area']:.0%}  "
                + "  ".join(f"level {k} {st[f'level_{k}']:.0%}" for k in range(1, LEVELS + 1)), fg="green", bold=True)
    typer.echo(f"rollouts with contact: {summary['rollouts_with_contact']}  safety mean "
               f"{summary['safety_mean_rad']:.4f} rad, max {summary['safety_max_rad']:.3f} rad")
    typer.echo(f"results: {out}/metrics.json" + ("  (rollouts: qc replay <file>)" if save_rollouts else ""))


@app.command()
def doctor():
    """Check the environment: assets, rendering, web build, adb, network."""
    _headless_gl()
    import mujoco

    from questcollect.sim import PANDA_XML, BimanualEnv

    def row(ok: bool, label: str, hint: str = ""):
        typer.secho(f"  {'✓' if ok else '✗'} {label}" + (f"  — {hint}" if hint and not ok else ""),
                    fg="green" if ok else "red")

    def render_ok(model) -> bool:
        try:
            mujoco.Renderer(model, 64, 64).close()
            return True
        except Exception:  # noqa: BLE001
            return False

    cfg = load_config()
    row(PANDA_XML.exists(), "menagerie Panda model", "run scripts/fetch_assets.sh")
    env = BimanualEnv(cfg)
    row(True, f"scene compiles ({env.model.nbody} bodies, {env.model.nu} actuators)")
    row(render_ok(env.model), f"offscreen rendering (MUJOCO_GL={os.environ['MUJOCO_GL']})",
        "try MUJOCO_GL=osmesa or check NVIDIA EGL drivers")
    row((ROOT / "web" / "dist" / "xr" / "index.html").exists(), "web client built", "cd web && npm ci && npm run build")
    adb = shutil.which("adb")
    row(adb is not None, "adb installed", "sudo apt install android-tools-adb")
    if adb:
        out = subprocess.run([adb, "devices"], capture_output=True, text=True).stdout
        row(any("\tdevice" in line for line in out.splitlines()), "Quest connected over USB",
            "plug in + allow USB debugging")
    row((ROOT / cfg.server.cert_dir / "cert.pem").exists(), "TLS cert for Wi-Fi", "qc certs")
    typer.echo(f"  LAN IP: {lan_ip()}")


if __name__ == "__main__":
    app()
