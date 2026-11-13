# Observation ablation: does joint velocity in the state break the closed loop?

Part of the review of 2026-11-12 ([../REVIEW_2026-11-12.md](../REVIEW_2026-11-12.md), findings V3–V9). All on the
scripted-expert takes (`data/expert_v2`, 200 takes until level 1). Configs here are
[../expert_dagger/config.yaml](../expert_dagger/config.yaml) with only `learn` changed.

## Why

In the demos the joint target leads the measured joints by 0.12 s of motion (`target − q ≈ 0.12 s × q̇`, R² 0.99).
A policy that sees the joint velocity can read its action off it. In closed loop that is a feedback path: drift of
the expert-data camera policy grows the more often it re-plans, and replacing only the velocity input by the
demo's removes most of it.

## Configs

| Config | Inputs | Dataset |
|---|---|---|
| `config_vel.yaml` | brick poses, joints + grippers + velocity + hand wrench | `review_expert_state_vel` |
| `config_novel.yaml` | brick poses, joints + grippers + hand wrench | `review_expert_state_novel` |
| `config_posonly.yaml` | brick poses, joints + grippers | not exported |
| `config_cam_novel.yaml` | cam_low + wrist cameras, joints + grippers + hand wrench | `expert_v2_novel` |

The first three use the preset `dp_state1` (no cameras, `n_obs_steps` 1, horizon 64, 16 steps per chunk, 20k
steps). `dp_state1` min-max normalizes the brick poses. Inputs that never vary in the data (bricks 3 and 4: z and
two quaternion components) then blow up at the start of an eval episode, so evaluate these runs with
`diagnostics/e8_eval_pinned.py`, not `qc eval`.

## Results (2026-11-13)

`outputs/review_oracle2_vel` and `outputs/review_oracle2_novel`, 50 eval seeds; drift = TCP distance from the demo
after 64 steps, median of 6 training takes, started 5 s / 3.5 s before the click (`scripts/drift_test.py`).

| | with velocity | without velocity |
|---|---|---|
| base in area | 98% | 100% |
| level 1 | 0% | 0% |
| blue carried to within 30 mm of red | 18/50 | 33/50 |
| within 10 mm | 2/50 | 16/50 |
| median closest approach while carried | 54 mm | 17 mm |
| drift, 1 step per chunk | 108 / 131 mm | 35 / 27 mm |
| drift, 16 steps per chunk (trained) | 33 / 19 mm | 14 / 24 mm |
| open loop (all channels forced) | 3.6 / 3.0 mm | 7.0 / 5.7 mm |

Without velocity the brick gets much closer and the drift at high re-planning frequency is 3–5× smaller. Neither
model clicks a brick: after 7.5 epochs their open-loop error (3–7 mm) is above what the studs catch (~3 mm).
In 2–3 of 50 rollouts of either model the left gripper never releases the base and the right arm waits to the end.

## Next

The camera policy without velocity. The dataset is exported; one night, pausing at checkpoint 030000 for an
evaluation, resumable at any point:

```bash
experiments/obs_ablation/night_run.sh                 # log: outputs/review_night.log
# pause:   pkill -f night_run.sh; pkill -f train_and_eval.sh; pkill -f 'bin/lerobot-train'
# resume:  QC_CONFIG=experiments/obs_ablation/config_cam_novel.yaml scripts/train_and_eval.sh --resume outputs/<run>
```

Compare with the run with velocity, `outputs/20261111-1316_dp_240_pre_expert_v2`, at the same checkpoint.

## Diagnostics

One-off scripts in `diagnostics/` (run from the repo root with `MUJOCO_GL=egl uv run --extra learn python ...`):
`e2_restart` closed loop from demo states, `e3_drift` drift with and without the safety filter, `e4_channels`
single channels forced, `e5_shortcut` the 0.12 s regression, `e6_baselines` trivial open-loop predictors,
`e7_transit` transit speed and shortfall in rollouts, `e8_eval_pinned` evaluation of the oracles. The general ones
became `scripts/drift_test.py` and the baselines in `scripts/fit_error.py`.
