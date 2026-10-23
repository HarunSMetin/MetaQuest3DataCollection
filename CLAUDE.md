# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

QuestCollect: teleoperate two simulated Franka Panda arms (MuJoCo) from a Meta Quest 3 via a WebXR page, record
demonstrations of one task (`build_tower`: stack 4 DUPLO bricks into a fixed tower) to HDF5, export to LeRobot.
Python 3.12 backend in a flat package (`questcollect/`, one module per pipeline stage), TypeScript/three.js/React
frontend (`web/`). The README is the user-facing doc (GitHub); keep it in sync with behaviour changes.

## Commands

```bash
./scripts/fetch_assets.sh               # Franka model from mujoco_menagerie -> assets/third_party/ (gitignored)
uv sync                                 # add --extra export for lerobot
(cd web && npm ci && npm run build)     # server serves web/dist; rebuild after web changes
uv run qc doctor                        # environment checklist

uv run qc serve [--mode hands] [--config my.yaml] [--no-https]
uv run qc view | replay <ep.hdf5> | inspect <ep.hdf5>
uv run qc export [--action ee] [--include-failures] [--out data/lerobot]

uv run --group dev pytest                                   # all tests
uv run --group dev pytest tests/test_teleop.py::test_protocol_roundtrip   # single test
MUJOCO_GL=egl uv run --group dev pytest                     # also runs the LeRobot export test (needs --extra export)

cd web && npm run typecheck             # tsc --noEmit (also part of npm run build)
cd web && npm run dev                   # vite on :5173, proxies /api and /ws to :8000

MUJOCO_GL=egl uv run python scripts/make_media.py   # docs/media: demo.gif, tower.png, cameras.png
```

`scripts/make_media.py` builds the whole tower with the real IK + gripper stack and prints `demo progress: 3/3 ...
success: True`; it's a good end-to-end check after changing brick physics, grasping or IK.

Headset-free end-to-end test (emulated Quest 3 via Meta's IWER + Playwright): setup is at the top of
`web/e2e/emulated-quest.mjs` (`npm install && npx playwright install chromium && npm run build`, then
`uv run qc serve --no-https --config <yaml with a scratch record.out_dir>`, then `npm test`). It writes episodes to
the server's out_dir. `npm run screenshots` overwrites the PNGs in `docs/media/`.

## Architecture

Data flow: Quest Browser → `/ws/xr` binary input frames → `Runtime` sim thread (teleop → IK → physics → recorder) →
binary pose frames back to headset + JSON status/events to the dashboard (`/ws/dash`).

- **`runtime.py`** is the hub. A single real-time thread at `sim.control_hz` (50 Hz) owns the env, IK, teleop and
  recorder. `server.py` only talks to it through thread-safe methods (`submit_input`, `command`,
  `latest_pose_frame`, `status`, `previews`) and event listeners. Dashboard/REST actions go through
  `POST /api/command` → `Runtime.command` queue; don't mutate sim state from the server thread.
- **Wire protocol**: `protocol.py` and `web/src/protocol.ts` are hand-mirrored binary layouts (little endian,
  fixed float counts, 16-byte headers). Change both together and keep `test_protocol_roundtrip` passing.
- **Scene + task**: `sim.py` builds the scene programmatically with `MjSpec` (ALOHA-style side-by-side arms) and
  holds the one task (`BRICKS`/`TOWER` constants, `_add_task`, `_randomize_task`, `tower_progress`, `held_bricks`,
  `_task_success`). Bricks are real DUPLO parts (BrickLink masses) with collision studs; body frame = bottom centre.
  Build area (`BUILD_AREA_*`, tape geoms in group 1, so cameras see it): `tower_progress` is 0 unless
  `in_build_area` (whole base footprint inside); `_randomize_task` never spawns bricks in it. Geom group 5
  (`OPERATOR_GROUP`) = headset/dashboard-only visuals (MuJoCo cameras render groups 0–2).
- **Goal UI is data-driven**: `sim.task_spec()` (bricks, goal placements relative to the base, area) is served at
  `GET /api/task`; `web/src/goal.ts` draws any such goal (isometric canvas) for the headset `xr/GoalPanel.ts` and the
  dashboard Goal card, using `status.progress`/`status.in_area`. Don't hard-code the tower in the client; there is
  no reference tower in the scene.
- **Clutch** (`BimanualEnv._update_clutch`, runs after every control step): one inactive weld per brick pair
  (`CLUTCH_PAIRS`); it switches on (at the bricks' current relative pose) when a pair touches, `stud_seated` finds the
  upper brick on the lower one's stud grid and the vertical contact force exceeds `CLUTCH_PRESS_N`, and off when
  `weld_pull` (the weld's tension + shear, compression excluded) exceeds `CLUTCH_PULL_N`. The weld (`CLUTCH_SOLREF`)
  must stay softer than the `RIGID_SOL*` contacts, or it takes the press/shear load and pops off while pressing.
  Studs have a chamfered convex-mesh top (`STUD_CHAMFER`) that centres a slightly misaligned brick. Weld relposes live in `model.eq_data`, which `mj_getState` doesn't cover, so `get_state` appends
  them (6 × 7) to the MuJoCo state and `set_state` restores them: always go through these, never raw
  `mj_getState`. `tower_progress` only counts clicked levels.
- **Gripper**: stiff tendon servo with `forcerange` = 2 × `sim.gripper_force` (each finger gets half the tendon force)
  and the target rate-limited to 50 mm/s per finger in `apply_action`, like the real Franka Hand. Elliptic friction
  cones + `impratio` 10.
  `BimanualIK` compiles the same spec without the task (`build_spec(..., with_task=False)`). It is open-loop but
  `follow()` clamps it to `ik.max_lead` of the measured joints each tick, so a physically blocked arm doesn't wind
  up; arm-arm avoidance in the IK is off by default (`ik.avoid_arm_collisions`), the physics handles collisions.
  `scene_export.py` serializes visual geometry served at `/api/scene` + `/api/scene.bin`;
  `web/src/MujocoScene.ts` rebuilds it in three.js for both pages, which (re)load it on every websocket connect.
  Pose frames carry only per-body poses (no target ghosts). Each hand has an operator-only `tcp_marker` geom
  (group 5) that the client colours per arm, red while `status.out_of_reach`, with a drop line to the table.
- **Task haptics + sounds**: `Runtime._task_feedback` emits distinct patterns (`HAPTIC_*` constants: grasp, hold tick,
  release, click, unclick, placed, complete); haptic events carry `repeat`/`gap_ms`, scheduled client-side in `web/src/xr/main.ts`.
  `{"type": "sound", name, arm}` events (engage/disengage, those task events, `area`, collision/table/out_of_reach
  warnings, rate-limited by `WARN_SOUND_GAP_S`) are synthesized with WebAudio in `web/src/xr/sounds.ts` (cue names must
  match); audio is unlocked by the Enter VR click, muted via the wrist menu (localStorage).
- **Teleop** (`teleop.py`): One-Euro filters, `ArmClutch` (relative motion, no jump on re-engage), controller (Grip
  press toggles the clutch) and hand (thumb-middle pinch toggles) mappers, and `Teleop` (stale-input disengage, jump rejection, workspace clamp, mode-mismatch detection).
  `frames.py` converts XR local-floor ↔ MuJoCo world and owns table placement/recenter.
- **Recording** (`recorder.py`): state machine (idle → countdown → recording → review → saved/discarded, auto-stop
  after success) plus HDF5 read/write; the file layout is documented in its module docstring. Images are never
  recorded: full `sim/state` per step lets the exporter re-render cameras at any resolution. `action/joint_pos` uses
  the ALOHA/ACT layout `[7 left, left grip, 7 right, right grip]`, grip 0 = closed, 1 = open.
- **Export**: keep all LeRobot API usage in `lerobot_export.py` (lerobot is an optional extra).
  `--randomize-visuals` applies `AppearanceRandomizer` (lights, colours, camera mounts) per episode at render time.
- **Config**: runtime tunables live in `config.yaml`, loaded into pydantic models in `config.py`. Exceptions: the task
  definition (brick sizes/colors/masses, `TOWER`, tolerances, stud `PLAY`, clutch forces) is constants at the top of
  `sim.py`, and
  the haptic patterns are `HAPTIC_*` constants in `runtime.py`.

## Design constraints

- Simulation-first; real robots (FR3) come later. Keep robot-specific details in `sim.py` and keep the HDF5 lossless
  (sim state, raw XR poses/hand joints) so data can be re-retargeted or re-rendered.
- The project was deliberately reduced to a single task; don't reintroduce a task registry or runtime task switching
  unless asked.
- `cli.py` defaults `MUJOCO_GL=egl`; a harmless EGLError can print at interpreter exit.
- `pyproject.toml` overrides away `evdev` (lerobot → pynput → evdev needs a C compiler); don't remove it.
