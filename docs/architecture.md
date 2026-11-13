# How QuestCollect works

This is the developer guide: what each part of the code does, how data moves through it, and where to change
things. For installing and using QuestCollect, see the [README](../README.md).

- [The big picture](#the-big-picture)
- [One control tick](#one-control-tick)
- [Threads and who owns what](#threads-and-who-owns-what)
- [Network: endpoints and the wire protocol](#network-endpoints-and-the-wire-protocol)
- [Coordinate frames](#coordinate-frames)
- [Teleoperation: from hand to arm target](#teleoperation-from-hand-to-arm-target)
- [Inverse kinematics](#inverse-kinematics)
- [The scene and the task](#the-scene-and-the-task)
- [Recording, replay and export](#recording-replay-and-export)
- [Policy learning](#policy-learning)
- [The web client](#the-web-client)
- [Common changes](#common-changes)
- [Tests](#tests)

## The big picture

```mermaid
flowchart LR
    subgraph Quest["Meta Quest 3 · Quest Browser (web/src/xr)"]
        IN["input.ts<br/>head, controllers, hand joints"]
        VIEW["MujocoScene.ts<br/>draws the robot · panels · sounds"]
    end
    subgraph PC["PC · Python (questcollect/)"]
        SRV["server.py<br/>FastAPI + websockets"]
        RT["runtime.py · 50 Hz sim thread<br/>teleop → IK → physics → recorder"]
        H5[("data/episodes/<br/>episode_*.hdf5")]
    end
    DASH["Dashboard<br/>(web/src/dashboard)"]
    LR[("LeRobot dataset")]

    IN -- "binary input frame<br/>(every XR frame)" --> SRV
    SRV -- "submit_input()" --> RT
    RT -- "body poses (50 Hz)<br/>status (10 Hz) · events" --> SRV
    SRV -- "/ws/xr" --> VIEW
    SRV -- "/ws/dash · camera JPEGs" --> DASH
    RT -- "save" --> H5
    H5 -- "qc export<br/>(re-renders cameras)" --> LR
```

The simulation exists only on the PC. The headset sends what it tracks (head, controllers or hands) and draws
the body poses the PC sends back, so the robot the operator sees is the same robot that is recorded. The
headset and the dashboard both rebuild the 3D scene from geometry the server exports (`/api/scene`), so nothing
about the robot or the bricks is hard-coded in the browser.

## One control tick

`Runtime._tick` runs 50 times a second (`sim.control_hz`). Each tick, in order:

| Step | Code | What happens |
| --- | --- | --- |
| 1 | `Runtime._process_commands` | Apply queued commands (record, save, reset, set mode, …) from the dashboard, the headset menu or the buttons. |
| 2 | `Teleop.update` | Latest input frame → one end-effector target and one gripper value per arm (see [Teleoperation](#teleoperation-from-hand-to-arm-target)). |
| 3 | `BimanualIK.follow` + `solve` | Targets → 14 joint angles, with the IK held within `ik.max_lead` of the measured joints. When a policy drives instead (see [Control modes](#policy-learning)): `PolicyRunner.act`. |
| 4 | `SafetyFilter.filter` | Either driver's joint targets, changed as little as possible so the arms keep apart and off the table. These are the recorded action; above `BLOCKED_RAD` of change the operator hears a warning. |
| 5 | `Recorder.tick` | Advance the countdown; switch to recording when it ends. |
| 6 | `recorder.snapshot` | If recording: the observation, the full sim state and the contact flags *before* the action. |
| 7 | `recorder.control_step` | `BimanualEnv.apply_action`: joint targets (plus the DART drift while an operator records, `record.action_noise`) to the arm servos, the gripper target moving at 50 mm/s per finger; if recording, the step built by `recorder.make_step` (obs, action without the drift, state, success, progress, clutch, contacts, who drove, raw teleop); then `BimanualEnv.step`: 10 physics substeps (`physics_dt` = 2 ms) and `_update_clutch`, which clicks or unclicks bricks. Rollouts, the scripted expert and generated takes run the same function. |
| 8 | `Recorder.append` | If recording: store that step. |
| 9 | `Runtime._task_feedback` | Compare grasp / progress / clutch state with the last tick and emit haptic, sound and toast events. |
| 10 | `Runtime._check_warnings` | Arm–arm and arm–table contacts (`BimanualEnv.contact_flags`, the one contact definition also used by recording, audit and evaluation), out-of-reach targets. |
| 11 | `Runtime._publish` | Pack the pose frame; every 5th tick the status dict; every 6th tick camera previews (only while someone is looking). |

During a replay (`replay` command or `qc replay`), steps 2–10 are skipped: the tick just loads the next recorded
state with `BimanualEnv.set_state`. The dashboard's replay can start paused at any step (`replay {path, i, paused}`),
`replay_pause` / `replay_seek {i}` control it, `replay_options {speed, loop}` sets the speed (recorded steps per
tick, `REPLAY_SPEED_RANGE`) and looping for this and later replays; without loop it holds the last frame until
`replay_stop`, and `record_toggle` ends it (back to a fresh live scene).

## Threads and who owns what

| Thread | Owns | Talks to the others through |
| --- | --- | --- |
| **sim** (`Runtime._run`) | the MuJoCo model and data, IK, teleop, recorder | — |
| **asyncio** (uvicorn, `server.py`) | websockets, HTTP | `Runtime.submit_input`, `command`, `latest_pose_frame`, `status`, `previews` (all lock-protected), and events pushed through `Runtime.add_listener` → `Hub` |
| **hdf5-writer** (`Recorder`) | writing one finished episode | a single-worker `ThreadPoolExecutor`, so saving never stalls the 50 Hz loop |

Rule: the server never touches simulation state directly. Anything that changes the sim goes through
`Runtime.command(name, **args)`, which queues it for step 1 of the next tick.

## Network: endpoints and the wire protocol

| Endpoint | Direction | Content |
| --- | --- | --- |
| `GET /xr/`, `/dashboard/` | → browser | the built web pages (`web/dist`) |
| `GET /api/scene`, `/api/scene.bin` | → browser | visual geometry: bodies, geoms, materials (JSON) + mesh vertices/faces (binary) |
| `GET /api/task` | → browser | the goal as data: bricks, where each goes, the build area (`sim.task_spec()`) |
| `GET /api/status` | → browser | the latest status dict (same as on the websockets) |
| `GET /api/datasets` | → browser | episode folders: `record.out_dir` first, then every sibling folder with `episode_*.hdf5` files (`name`, `count`, `recording`, `title`/`description` from an optional `dataset.json`) |
| `GET /api/episodes` | → browser | saved episodes with their manual labels from `curation.json` (`success` = label, `recorded_success` = as saved, `tags`, `note`, `excluded`) and provenance (`origin` = `retimed_from`/`generated_from`, `generation`, `retime_moves`). `?dataset=<folder>` (also on PATCH and audit) picks a sibling folder of `record.out_dir` |
| `PATCH /api/episodes/<file>`, `DELETE /api/episodes/<file>` | → server | change labels (`exclude`, `note`, `tags`, `success`; null = as recorded) via `audit.update_curation` in that folder's `curation.json` / delete the file and its labels (recording folder only: 403 for any other dataset) |
| `GET /api/audit` | → browser | per-episode audit: contact shares and events, idle share, level times, and why a dataset would skip it (same reasons as `audit.select`) |
| `POST /api/command` | → server | `{"name": ..., "args": {...}}`, queued to the sim thread |
| `GET /api/preview/<cam>.jpg` | → browser | latest camera preview (rendered only while requested) |
| `WS /ws/xr` | both | headset: binary input frames in; binary pose frames (every tick), JSON status + events out |
| `WS /ws/dash` | both | dashboard: pose frames at 30 Hz, JSON status + events out; JSON commands in |

**Binary frames** are defined in [questcollect/protocol.py](../questcollect/protocol.py) and mirrored by hand
in [web/src/protocol.ts](../web/src/protocol.ts) (little endian, 16-byte headers):

- *Input* (headset → server, every XR frame): header with both hands' input source (none / controller / hand),
  sequence number and an echoed server timestamp (for the round-trip time), then the head pose and, per hand, the
  control pose, 8 button values, 25 hand-joint poses and 2 pinch distances: always 391 floats (1,580 bytes).
- *Pose* (server → clients): header with flags (left engaged, right engaged, recording), body count and server
  time, then every MuJoCo body's world pose.

**JSON events** (server → clients): `status` (10 Hz), `toast`, `haptic` (`arm`, `intensity`, `ms`, `repeat`,
`gap_ms`), `sound` (`name`, `arm`), `saved`.

## Coordinate frames

[questcollect/frames.py](../questcollect/frames.py) converts between the two worlds:

| | Up | Forward | Quaternion order |
| --- | --- | --- | --- |
| WebXR `local-floor` | +y | −z | x y z w |
| MuJoCo world | +z | +x (away from the operator) | w x y z |

`Placement` is the rigid transform that puts the simulated table in the room. The same transform is used to draw
the scene in the headset and to map hand poses back into the world, so what you see is what you control.
*Recenter* (left stick click, wrist menu, or automatically 1.2 s after entering VR) places the operator's head
behind the arm bases, looking along +x, and keeps the simulated floor on the real floor.

## Teleoperation: from hand to arm target

[questcollect/teleop.py](../questcollect/teleop.py), per arm and tick:

```text
XR pose → world (Placement) → jump rejection → One-Euro filter → clutch → workspace clamp
  → IK → safety filter (runtime)
```

- **Input source.** In *controllers* mode the control pose is the controller's grip pose; in *hands* mode it is
  the middle-finger metacarpal position with the wrist orientation (stable while pinching). If the headset sends
  the other kind of input, `mode_mismatch` is set and the UI says *Pick up controllers* / *Put controllers down*.
- **Clutch** (`ArmClutch`). Toggled by a Grip press (controllers) or a thumb–middle pinch (hands). While engaged,
  target = arm pose at engage time + scaled hand motion since then (`teleop.scale`), and hand rotation is applied
  in the world frame. Releasing and re-engaging never makes the arm jump.
- **Gripper.** Trigger value, or the thumb–index distance mapped between `pinch_closed_m` and `pinch_open_m`.
- **Safety.** Input older than 0.25 s or a jump of more than 12 cm between ticks drops the clutch (the arm holds
  its last target until you engage again). Targets are clamped to `workspace_min/max`.
- **Buttons.** `CONTROLLER_COMMANDS` maps A/B/X/left-stick presses to recorder commands. The headset's button
  labels (`HINTS` in `web/src/xr/visuals.ts`) mirror this table; keep them in sync.

## Inverse kinematics

[questcollect/ik.py](../questcollect/ik.py) uses [mink](https://github.com/kevinzakka/mink) with one QP for
both arms: a frame task per gripper (position + orientation cost), a small posture task, a damping task
(`ik.damping_cost`: prefer the smallest joint motion), joint position and velocity limits, and optionally an
arm–arm collision limit (`ik.avoid_arm_collisions`, off by default: the safety filter after the IK keeps the arms
apart).

Without the damping task, and with the arm bases at the table edge (now `sim.arm_base_x` = −5 cm; episodes recorded before it existed replay at 0), a top-down grasp
of the nearest bricks folds the elbow to within 0.1 rad of its limit. There the IK spins joints 1+3 or 5+7 at the
velocity limit while the hand barely moves: 144 steps in the 11 recorded human demos, replayed kinematically.
Both changes together: 24, tracking error p99 1.5 mm.

The IK keeps its own model (the same scene without the bricks) and is open-loop, except that `follow()` keeps it
within `ik.max_lead` radians of the measured joints. Without that, an arm blocked by the table or the other arm
would let the IK run on through the obstacle and then yank the arm when it came free. If the target stays more
than 3 cm away from what the IK can reach, the arm is flagged *out of reach*.

## The scene and the task

[questcollect/sim.py](../questcollect/sim.py) builds everything in code with `MjSpec`:

- **Robots.** Two Franka Panda arms from MuJoCo Menagerie, side by side on the near table edge, 0.6 m apart
  (`sim.arm_spacing`), gravity-compensated like the real controller. Each hand has a `tcp` site (the point the
  arm is steered by), a wrist camera and an operator-only grasp-point marker.
- **Gripper.** A stiff servo on the finger tendon whose force is capped at `sim.gripper_force` (40 N), with the
  target moving at the real Franka Hand's 50 mm/s. The four small fingertip pads from Menagerie are merged into
  one flat pad so a brick can't wedge between them.
- **Bricks.** Hollow DUPLO bricks built from boxes and cylinders: top plate, walls, chamfered studs and the tubes
  underneath, at real dimensions and BrickLink masses. Each brick also has an invisible solid `<brick>_core` in its
  hollow underside that only the robot collides with, so a finger can't hook into a brick.
- **Contacts.** Stiff contact parameters (`RIGID_SOLREF`, `RIGID_SOLIMP`), elliptic friction cones and
  `impratio` 10, so bricks don't sink into each other and grasps don't creep.
- **Clicking (clutch).** One inactive weld per brick pair (`CLUTCH_PAIRS`). After every step, `_update_clutch`
  turns a weld on when two bricks touch, `stud_seated` finds the upper one on the lower one's stud grid, and the
  vertical contact force exceeds `CLUTCH_PRESS_N`. It turns it off when the weld has to hold more than
  `CLUTCH_PULL_N` of tension plus shear (compression doesn't count, so pressing harder never pops a brick). The
  weld is softer than the contacts, so the studs carry the press load.
- **Progress and success.** `tower_progress` counts levels that are in the right place (within `TOL_XY`,
  `TOL_Z`, `TOL_YAW`) *and* clicked, bottom-up, and is 0 unless the base lies completely inside the build area.
  `_task_success` = all levels placed, nothing held, and every brick slower than 2 cm/s.
- **Cameras.** `cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist` (ALOHA names). Geom group 5
  (`OPERATOR_GROUP`) holds visuals only the operator sees (grasp-point markers); MuJoCo cameras render groups 0–2,
  so these never reach the recorded images. `CameraRenderer` (export, policy input, previews) renders without
  shadows; the headset draws its own shadows. `/api/scene` exports the cameras too, so the headset can draw the
  wrist cameras' view.
- **Start layout.** `SPAWN`: every brick in its own region (±`SPAWN_JITTER_M`, ±`SPAWN_YAW`), alternating sides in
  build order, so each arm has its own bricks and the two can work at once (left: red base, yellow; right: blue,
  green).

## Recording, replay and export

- **State machine** ([questcollect/recorder.py](../questcollect/recorder.py)):
  `idle → countdown (3 s) → recording → review → saved | discarded → idle`. With auto-stop on, recording ends
  1 s after the task first succeeds. Saving resets the scene with a new brick layout (`SPAWN`). In review,
  `status.review` judges the take with the curation rules (`audit.audit_arrays`, `verdict`, `kept_spans`: the same
  code as `qc audit`), shown on the headset HUD.
- **File format.** One HDF5 file per episode; the full field list is the module docstring of `recorder.py`.
  Every step is built by `recorder.make_step` inside `recorder.control_step` (snapshot before the action → apply
  → record → physics step) and every file's attributes by `recorder.episode_meta`, whoever drove (the live
  recorder, `qc eval` rollouts, the scripted expert, generated takes), so all episodes look the same. Images are never recorded.
- **Exact replay.** `sim/state` is MuJoCo's full integration state *plus* the six clutch welds' relative poses.
  The welds live in `model.eq_data`, which `mj_getState` doesn't cover, so always use
  `BimanualEnv.get_state` / `set_state`, never raw `mj_getState`.
- **Replaying in the right scene.** Each episode stores its config and `scene_hash` (a fingerprint of geoms,
  masses, materials, cameras and state layout, computed from the saved MJCF). `sim.env_from_episode(meta)` builds
  the scene from the episode's own config and warns if the fingerprint differs from today's scene.
  `sim.load_recorded_model` compiles a saved `model_xml` (its meshes are found in the Menagerie folder).

## Policy learning

The learning side is a pipeline of stages; each one writes a folder with a manifest, and raw episodes are never
modified:

```text
data/episodes/*.hdf5 ──qc audit──▶ audit.json (+ curation.json) ──qc export──▶ data/lerobot/<name>/
      ▲                                                                              │
      │                                                                         qc train
      │                                                                              ▼
      └──── qc serve --policy (shared control: your corrections) ◀──── outputs/<run>/train/checkpoints/…
                                                                        └── qc eval ──▶ outputs/<run>/eval/<time>/
```

**The dependency rule:** nothing outside [questcollect/learn/](../questcollect/learn/) imports torch or lerobot
(optional extra `learn`), so recording works without them. The runtime imports `learn.policy` only inside
`Runtime.load_policy`.

| Stage | Code | What it does |
| --- | --- | --- |
| audit | [audit.py](../questcollect/audit.py) | `audit_episode` (→ `audit_arrays`, also used for the take under review): contact shares and events (from the recorded `contacts`, or by replaying `sim/state` for episodes older than schema v5), idle share, active range, level times, operator-driven runs. `audit_dir` caches by file size + mtime. `select` applies the `curate` rules and `curation.json` and returns `Segment`s plus a reason for every rejected episode. |
| generate | [generate.py](../questcollect/generate.py) | experimental: `replay` plays a take's TCP targets (`action/ee_pose`) through `BimanualIK` from `sim/state[0]`: pick windows move with the brick's spawn shift, place windows follow the brick it goes onto (live, through the press: towers slide) plus an in-hand correction frozen `FREEZE_Z` above the seat, blends advance with the source hand's path; optional new layout and per-arm time maps (`frames`). `check` = tower built, contacts within the source's + `CONTACT_TOL`, `sim.ArmClearance` within `Limits.of_source` (fixed `ARM_CLEAR_M`/`CAM_CLEAR_M`, or the source's own closest minus `CLEAR_TOL_M`). `generate` samples a layout and writes `source = "generated"`. `scripts/generate_set.py` builds a balanced set. `grasp_offset` (`replay(grasp_shift=)`, `scripts/generate_grasps.py`): the bricks start shifted along the axis between the finger pads (from the source hand's TCP y axis at its first grasp) while the hand targets stay, so the grasp is off centre and the place follows the brick through the in-hand correction; `min_cam_m` loosens the wrist-camera limit for these IK replays. |
| retime | [retime.py](../questcollect/retime.py) | experimental: per arm, moves between holds (`_moves`, with their tower-zone span; with `stage` a move that enters the zone after ≥ `STAGE_MIN_S` is split at its slowest point in the `STAGE_WIN_S` before the zone, so the arm can wait there, eased in and out over `EASE_S` by fractional frames from `_Timeline.indices(ease=)`; a staged zone part starts `EASE_S` after the zone is free). `_plan` places each arm's next move at its earliest start where `too_close` (ArmClearance on precomputed per-arm poses of the commanded joints, plus `MARGIN_M`; source-simultaneous pairs exempt) is false, the arm that can go first first; zone visits exclusive and in source order; holds keep `KEEP_S` or `GRIP_SETTLE_S`; a deadlock delays the move furthest ahead of its recorded time and re-plans. Holds are time-warped. Planning bounds are never looser than `generate.check`'s limits + `PLAN_SLACK_M`. The plan is played with `generate.replay(frames=..., joints=True)` (each arm's recorded joint targets on its timeline, interpolated at fractional frames), else the IK path, and must pass `generate.check`; failures delay the early moves around the failure (`MAX_TRIES`). Writes new files with `source = "retimed"` to another folder. |
| features | [learn/features.py](../questcollect/learn/features.py) | `FeatureSpec`: cameras, image size, joint/ee action, env state, joint velocities and the hand wrench in the state (`joint_vel`, `ee_wrench`, log-compressed; manifests without them load as false; `export` recomputes `obs/ee_wrench` from `sim/state` for files recorded before it; `action_label: executed` (default) labels with the targets actually sent, `BimanualEnv.executed_targets` reads them from the next step's ctrl in `sim/state`, `tcp_poses_at` gives their TCP poses; manifests without it load as `recorded`). The single definition of observation and action layout, used by export *and* the policy runner; stored in each dataset manifest, so a policy is always fed what it was trained on. |
| export | [learn/export.py](../questcollect/learn/export.py) | per segment: `env_from_episode`, `set_state` each step, render `spec.cameras`, `LeRobotDataset.add_frame`. Optional `AppearanceRandomizer`. Refuses an existing folder; writes `questcollect_manifest.json`. All LeRobot dataset API calls are here. |
| train | [learn/train.py](../questcollect/learn/train.py) | turns a preset from `learn.presets` into a `lerobot-train` command (output in `outputs/<run>/train/`) and writes `questcollect_run.json` (preset, dataset manifest, git commit, the command). It refuses a dataset in which a state, environment-state or action input is constant (`constant_inputs`; normalization would divide it by ~0) unless `allow_constant_inputs`. A preset `ema:` key runs [learn/lerobot_train_ema.py](../questcollect/learn/lerobot_train_ema.py) instead: lerobot's training script with patched `update_policy`/`save_checkpoint` that keep a weight EMA and save it as `pretrained_model`. |
| policy | [learn/policy.py](../questcollect/learn/policy.py) | `PolicyRunner`: loads a checkpoint with its LeRobot pre/post-processors, renders the cameras at the dataset's size, returns joint targets (end-effector policies go through `BimanualIK`). Skips rendering while a predicted chunk still has queued actions (ACT; diffusion only with `n_obs_steps` 1, since it otherwise stacks the last frames). Temporal ensembling is ACT only. |
| evaluate | [learn/evaluate.py](../questcollect/learn/evaluate.py) | rollouts on fixed seeds (`eval_seeds`, a separate stream from recording seeds), no wall clock; every action through the safety filter; metrics per rollout and summarized; rollouts saved with `make_step`, so `qc replay` and `qc audit` work on them. |

**Safety filter** ([safety.py](../questcollect/safety.py)). One mink QP per control step (8 substeps, because the
collision limits are linearized) on the robot-only scene: a posture task pulls towards the proposed joint targets,
and limits keep joint ranges and speeds, the arms ≥ `safety.arm_margin` apart, the links ≥ `safety.table_margin`
above the table, and the fingers ≥ `safety.finger_table_margin` above it (pushed hard, a 5 mm margin holds at
~4.7 mm; the linearization undershoots). In free motion the output equals the input. Its own configuration stays
within `ik.max_lead` of the measured joints, like `BimanualIK.follow`. Callers pass the measured finger joints:
the robot-only model would otherwise have the fingers closed, and an open finger of a tilted hand hangs lower (that
let a finger rest on the table). The finger DOFs are frozen in the QP (`DofFreezingTask` as a constraint), so it
can't clear an obstacle by moving them. DART noise goes in before the filter, so it can't push through either.

**Control modes** (`Runtime._who_drives`). `teleop`: the operator, as always. `policy`: `PolicyRunner.act` →
`SafetyFilter.filter` every tick, buttons still work. `shared` (DAgger): the policy drives until an arm is engaged
(Grip); while the policy drives, `Teleop.sync_hold` and `BimanualIK.reset` keep the operator side at the arms'
current pose, so a takeover never jumps; when every arm is released the policy restarts with a fresh chunk. Each
recorded step stores `control/source`, and `curate.dagger_frames: operator` exports only the operator's steps. On a takeover the grippers keep the policy's
last command (`Runtime._grip_latch`) until that controller's Trigger is within `GRIP_CATCH` of it.
`audit.select` also cuts the `curate.cut_contacts` moments (arm on the table / arms touching, plus
`contact_margin_s` either side) out of each take, so the rest of the take still becomes segments.
`curate.until_level` ends each take `until_level_tail_s` after that tower level clicked (a sub-task dataset); a take
that reached that level counts as a success there (e.g. DAgger takes stopped after it).
`generate.generate` and `retime_episode` replay a take's *executed* targets (`sim.executed_actions`), not the
recorded pre-drift ones: with DART only the executed ones rebuild the tower.
`record.action_noise` (DART) adds a slow drift to the *executed* targets of the arms the operator is moving, while
recording, before the safety filter; the clean ones are recorded. It is an Ornstein–Uhlenbeck process
(`DART_TAU_S`) low-passed once more (`DART_SMOOTH_S`): white noise per tick would be smoothed away by the servos
and leave nothing to correct, and a raw OU path made the arms visibly tremble (0.4 mm per tick).

## The web client

[web/](../web/) is one Vite project with two pages that share the scene renderer and the protocol code.

| File | Role |
| --- | --- |
| `src/protocol.ts` | binary frame layouts (mirror of `protocol.py`), status / event / task types |
| `src/net.ts` | auto-reconnecting websocket |
| `src/MujocoScene.ts` | rebuilds the MuJoCo scene in three.js from `/api/scene`, animates it from pose frames, colours the grasp-point markers (reach), draws the wrist cameras' frustums and table footprints, highlights the next brick |
| `src/goal.ts` | draws any goal from `/api/task` as a small isometric picture; used by both pages |
| `src/xr/main.ts` | headset app: XR session, render loop, event handling (haptics, sounds, toasts) |
| `src/xr/input.ts` | samples head, controllers and hand joints into the input frame |
| `src/xr/visuals.ts` | controller models (vendored in `public/xr-profiles`), button hints, status chips, hand skeletons, pointer rays |
| `src/xr/Hud.ts`, `GoalPanel.ts`, `CamPreviewPanel.ts`, `WristMenu.ts`, `Panel.ts` | in-headset panels, drawn on canvases (goal on the operator's left, camera previews polled from `/api/preview` on the right, as a column, row or grid); ray and poke interaction |
| `src/xr/Layout.ts` | `PanelLayout`: the world panels' default poses, grabbing them with a ray (only with a hand whose arm is released; the gripper value is held and the stick not sent while a ray is on UI), poses in the `mj.root` frame in localStorage, lock/reset |
| `src/xr/sounds.ts` | synthesized sound cues (WebAudio, no audio files); cue names match the server's `sound` events |
| `src/dashboard/` | the PC dashboard (React), one shared 3D view + camera strip and two tabs: `App.tsx` Record (recorder with a next-step hint, goal, arms, control mode) and `Episodes.tsx` (dataset menu from `/api/datasets`, episode list with audit verdicts and provenance, labels via `PATCH /api/episodes`, replay timeline with speed, frame steps and loop driving the `replay*` commands). The page fills the window; the side column scrolls on its own |
| `e2e/` | headset-free tests and README screenshots with an emulated Quest 3 (IWER + Playwright) |

## Common changes

| To change… | Edit | Notes |
| --- | --- | --- |
| a setting (rates, scale, workspace, ports, gripper force) | `config.yaml` | see the README's Configuration section |
| the tower, bricks, build area, start layout, tolerances, click forces | constants at the top of `questcollect/sim.py` | the goal panels follow automatically (`task_spec()`); old episodes keep their own `tower` attribute |
| vibration patterns | `HAPTIC_*` in `questcollect/runtime.py` | |
| sounds | `CUES` in `web/src/xr/sounds.ts` | names must match the server's `sound` events |
| controller buttons | `CONTROLLER_COMMANDS` in `teleop.py` **and** `HINTS` in `web/src/xr/visuals.ts` | |
| the wire format | `protocol.py` **and** `web/src/protocol.ts` | keep `test_protocol_roundtrip` passing |
| recorded fields | `recorder.make_step` (called by `recorder.control_step`), the `recorder.py` docstring, bump `SCHEMA_VERSION` | `audit.py` must still read older files |
| what counts as a contact | `BimanualEnv.contact_flags` in `sim.py` | bump `AUDIT_VERSION` in `audit.py` so cached audits are redone |
| which episodes / steps are trained on | `curate` in `config.yaml`; the rules in `audit.verdict` / `select` | |
| policy inputs / outputs | `learn` in `config.yaml`; layout code in `learn/features.py` | affects new datasets only; old ones keep theirs |
| training settings | `learn.presets` in `config.yaml`, or extra args to `qc train` | |
| evaluation metrics | `learn/evaluate.py` (`RolloutResult`, `summarize`) | |
| safety margins / limits | `safety` in `config.yaml`; `safety.py` | |

Changing the scene's degrees of freedom (adding a brick, a joint) changes the size of `sim/state`. The exporter
refuses episodes whose state size doesn't match the scene, rather than replaying them wrongly; any other scene
change shows up as a `scene_hash` mismatch warning.

## Tests

| Command | Covers |
| --- | --- |
| `uv run --group dev pytest` | frames, filters, clutch, mappers, protocol round-trip; scene build, brick physics, clicking, grasping, gripper speed, IK, state replay, scene export, recorder, haptics; audit and selection; safety filter; control modes and takeover |
| `MUJOCO_GL=egl uv run --extra learn --group dev pytest` | the above plus `test_learn.py`: tiny episodes → export → a 2-step `lerobot-train` → a short `qc eval` rollout |
| `cd web && npm run typecheck` | TypeScript |
| `cd web/e2e && npm test` | an emulated Quest 3 against a running server: controllers, hands, clutch, gripper, record → save (setup at the top of `emulated-quest.mjs`) |
| `MUJOCO_GL=egl uv run python scripts/make_media.py` | builds the whole tower with the real IK and gripper; must print `success: True` |
