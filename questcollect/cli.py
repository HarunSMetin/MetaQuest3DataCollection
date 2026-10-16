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
export_app = typer.Typer(no_args_is_help=True, help="Convert raw episodes to training formats")
app.add_typer(export_app, name="export")


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
    config: Optional[Path] = typer.Option(None, help="YAML config (default: config/default.yaml)"),
    task: Optional[str] = typer.Option(None, help="Override the task"),
    mode: Optional[str] = typer.Option(None, help="controllers | hands"),
    https: bool = typer.Option(True, help="Also serve HTTPS for Wi-Fi (needs `qc certs`)"),
):
    """Run the sim, teleop server, and web clients."""
    _headless_gl()
    import uvicorn

    from questcollect.runtime import Runtime
    from questcollect.server.app import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(config)
    if task:
        cfg.task = task
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
def view(task: Optional[str] = typer.Option(None)):
    """Open the scene in MuJoCo's native viewer (sanity check, no headset)."""
    import mujoco
    import mujoco.viewer

    from questcollect.sim.env import BimanualEnv

    cfg = load_config()
    env = BimanualEnv(cfg, task)
    env.reset()
    mujoco.viewer.launch(env.model, env.data)


@app.command()
def replay(path: Path, speed: float = typer.Option(1.0)):
    """Play back a raw episode's recorded states in MuJoCo's native viewer."""
    import mujoco.viewer

    from questcollect.record.hdf5_io import read_episode
    from questcollect.sim.env import BimanualEnv

    meta, data = read_episode(path, keys=["sim/state"])
    env = BimanualEnv(load_config(), meta["task"])
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
    from questcollect.record.hdf5_io import read_episode

    meta, data = read_episode(path)
    for k, v in meta.items():
        if k in ("model_xml", "config_yaml"):
            v = f"<{len(v)} chars>"
        typer.echo(f"{k:>16}: {v}")
    typer.echo("")
    for k in sorted(data):
        typer.echo(f"{k:<32} {str(data[k].shape):<18} {data[k].dtype}")


@export_app.command("lerobot")
def export_lerobot(
    raw: Path = typer.Option(ROOT / "data" / "raw", help="Raw episode dir (or a task subdir)"),
    out: Path = typer.Option(..., help="Output LeRobot dataset root"),
    repo_id: str = typer.Option("local/quest_bimanual_panda"),
    task: Optional[str] = typer.Option(None, help="Only this task"),
    action: str = typer.Option("joint", help="joint | ee"),
    height: int = typer.Option(480),
    width: int = typer.Option(640),
    include_failures: bool = typer.Option(False),
):
    """Raw HDF5 -> LeRobotDataset (cameras re-rendered from recorded sim states)."""
    _headless_gl()
    from questcollect.export.to_lerobot import export

    n = export(raw, out, repo_id, task=task, action=action, height=height, width=width,
               include_failures=include_failures)
    typer.secho(f"exported {n} episodes to {out}", fg="green")


@app.command()
def doctor():
    """Check the environment: assets, rendering, web build, adb, network."""
    _headless_gl()
    import mujoco

    from questcollect.sim.env import BimanualEnv
    from questcollect.sim.scene import PANDA_XML

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
