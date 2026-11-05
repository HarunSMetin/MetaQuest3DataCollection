# tower_v3: unassisted hand demos, Diffusion

73 episodes recorded 2026-11-04/05 (grasp assist removed, DART 0.003 rad, safety filter on teleop, scene
`0175cdac4a97d9c2` = today's scene); `qc audit` keeps 71 (69.5 min). Run on 2026-11-06 overnight:

```bash
MUJOCO_GL=egl uv run --extra learn qc export tower_v3       # config.yaml: joint actions, executed labels
scripts/train_and_eval.sh tower_v3 dp_240
```

## What the data looks like (measured before training)

| | tower_v2 (assist, generated) | tower_v3 |
|---|---|---|
| grasp offset, TCP to brick centre | ~1 mm | median 4.6 mm, p90 11.5 mm, max 21 mm |
| bricks lifted more than once (regrasp / retry) | 0 | 21 of 291 |
| clicks / unclicks | | 224 / 8 |
| upward force on the hand around a click | | median 48 N (p10 11, p90 107); `CLUTCH_PRESS_N` is 6 N |

So the data now has off-centre grasps, some corrections, and a clear force cue for "pressed until it clicked"
(`obs/ee_wrench` is in `observation.state`, `learn.ee_wrench`).

## Why executed-target labels (`learn.action_label: executed`)

With DART the arm executes `recorded target + drift`. Near clicks (the 2 s before each) the drift at the TCP is
2.2 mm std (p90 3.7 mm), about the stud lead-in's ~3 mm. The operator only partly corrects for it: the next 0.5 s
of their target moves −0.33× the drift. So the recorded target is not where the brick was seated, the executed
one is. Open-loop replay from `sim/state[0]` without noise (episodes 5, 30, 60):

| replayed | levels reached |
|---|---|
| recorded joint targets | 1, 1, 2 |
| executed joint targets | 3 (bit-exact, success) |
| recorded TCP targets → IK | 1, 2, 1 |
| executed TCP targets → IK | 3, 2, 2 |

Joint actions stay: executed joint targets reproduce the physics exactly, the IK path loses levels open-loop, and
on tower_v2 the ee oracle was not more precise (7.9 vs 7.3 mm). `config_ee.yaml` here is config.yaml with
`learn.action: ee`, for a later comparison.

## Physics in training and eval

Training is offline on rendered `sim/state`, so the physics only matters through the recorded data. Evaluation
(`qc eval`) runs the full `BimanualEnv`: the clutch welds (`_update_clutch` after every step, 6 N to click, 10 N
to pull apart), stud chamfers, brick cores, gripper force/rate limit and the safety filter, the same as during
teleop (eval scene hash = recording scene hash). DART is off in eval.

## Results

(fill in from outputs/<run>/RESULTS.txt)
