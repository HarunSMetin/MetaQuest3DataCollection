# two_bricks: retimed demos, first two bricks, pretrained image encoder

Follows [tower_v3](../tower_v3/README.md) (dp_240, from-scratch encoder: 0/20, base in area ~60%, level 1 0%).
That run's chunks were already 3–15 mm off its own training frames, grasps in eval 10–39 mm off centre
(the stud lead-in catches ~3 mm). Two changes at once, plus the retime the user wants for real bimanual work:

- **Retimed takes** (`qc retime`, both arms working at once). Retime now replays the *executed* targets
  (`sim.executed_actions`); with the pre-DART recorded ones 2 of 3 takes didn't even rebuild the tower.
  Retimed files are re-simulated without DART, so their recorded action = executed.
- **Sub-task:** `curate.until_level: 1`: each take ends 1 s after the blue brick clicked on the red one.
- **Cameras:** `cam_low` + both wrists instead of `cam_high`: from above, the arms hide most of the tower;
  `cam_low` sees its height from the side (small at 240×320; the hand wrench still gives the click).
- **Pretrained encoder:** preset `dp_240_pre` = dp_240 + ImageNet ResNet18 (BatchNorm kept).

The metric is `level_1` in RESULTS.txt (full-tower success can't happen in this setup).

```bash
uv run qc retime --out data/episodes_retimed_v4 --jobs 4   # staging is the default
MUJOCO_GL=egl uv run --extra learn qc export two_bricks_v4 --config experiments/two_bricks/config.yaml
QC_CONFIG=experiments/two_bricks/config.yaml scripts/train_and_eval.sh two_bricks_v4 dp_240_pre
```

## Results

`20261106-2147_dp_240_pre_two_bricks_v4` (50 retimed takes, 13.7 min, 80k steps):

| Eval | base in area | level 1 (the goal here) |
|---|---|---|
| 16 steps per chunk | 68% | **16%** (8/50) |
| 8 steps per chunk | 60% | 8% |

Checkpoints (50 rollouts each, 16 steps per chunk): 20k base 22% / level 1 0%; 40k 52% / 0%; 60k 84% / 14%;
80k 68% / 16%. It improves up to 60k and then levels off (±5% is noise at n = 50), so longer training is not the
obvious fix.

The best camera policy so far (earlier ones: level 1 0–4%; the brick-pose oracle: 22%). Where the other 42 go
(16 steps per chunk):
- blue never lifted: 13
- blue lifted but never put on red: 22. Mostly held 2–4 cm above or beside the tower for 10–24 s until the time
  limit, while the other arm is ~20 cm away. Demos hover there 2–4 s at most (human and retimed alike).
- blue on red but not clicked: 7. 3–22 mm off; it does press (90–125 N), so the force cue was learned.

Grasps are the root: TCP to brick centre at lift-off, red median 11.7 mm (demos 3.8), blue 19.8 mm (demos 5.1).
With the brick that far off in the hand, it never looks aligned above the tower, and the demos hardly show that
state or how to fix it.

### Fine-tune on grasp-offset demos (`20261107-1744_dp_ft_mix_v5`)

From the 60k checkpoint, 30k steps (`dp_ft`) on `mix_v5`: the 50 retimed takes + 116 generated ones whose bricks
start shifted between the finger pads (`scripts/generate_grasps.py`; 166 episodes, 46 min). `check_pipeline` 16/17
(only the known 5 mrad start-pose outlier).

| Eval | base in area | level 1 |
|---|---|---|
| 16 steps per chunk | 76% | 14% (7/50) |
| 8 steps per chunk | 58% | 8% |

No gain over the 60k checkpoint (84% / 14%). Failures: blue never lifted 20 (was 13), lifted but never put on red
13 (was 22), on red but not clicked 10. Grasps got *worse*: TCP to brick centre at lift-off red median 19.0 mm,
blue 16.4 mm. The policy copied the wider grasp spread of the generated takes instead of learning to place with an
off-centre brick. Of the 20 "never lifted", most do reach the brick and close (several retry 4–7 times) but 14–29 mm
off centre, beyond what the generated takes show (≤ 11.5 mm). Lesson: offset grasps as *start* conditions add
imprecision; the policy needs corrections (re-grasp, re-align), which only DAgger gives. DAgger is run on the 60k
checkpoint, the better of the two.

### Plan A diagnostics (2026-11-08)

**Averaged checkpoint** (weights of 60k and 80k averaged, a free stand-in for an EMA): base 82% / level 1 14%, the
same as 60k. Weight noise between checkpoints is not what holds it back. EMA is now available anyway
(`dp_240_pre_ema`, [learn/lerobot_train_ema.py](../../questcollect/learn/lerobot_train_ema.py)).

**Open-loop fit** ([scripts/fit_error.py](../../scripts/fit_error.py), 60k checkpoint, the moving arm's TCP target vs the
demo's executed target, first action of a fresh chunk):

| frames | training takes | unseen takes (generated) |
|---|---|---|
| last 0.6 s before a grasp | median 7.2 mm, p90 11.5 | 7.8 mm, p90 14.6 |
| last 0.6 s before the click | 4.4 mm, p90 7.8 | 4.6 mm, p90 10.8 |
| elsewhere | 6.3 mm | 7.0 mm |

Baselines on the same grasp frames: repeat the previous target 1.2 mm; hold the current joints 5.3 mm. Training and
unseen are equally off, so this is underfitting, not overfitting. Split of the grasp error: 8 samples per frame
differ by only 3 mm (the rest is systematic); about half is timing (the predicted path matches the demo path a few
steps earlier/later), the *spatial* error, distance from the predicted points to the demo path, is median 4.8 mm
(p90 9.2) before a grasp and 2.6 mm (p90 4.4) before the click. So: ~5 mm of spatial error at the grasp already on
the training data, which grows to the 12–20 mm grasp offsets seen in closed loop. Both causes are real: the fit
(fix: action representation, e.g. targets relative to the current joints) and compounding (fix: DAgger).

### DAgger session

    uv run --extra learn qc serve --config experiments/two_bricks/config_dagger.yaml

Same scene and settings as `config.yaml` here, plus shared control with the 60k checkpoint, 16 steps per chunk (as
trained) and takes written to `data/dagger_v1` (not into the retimed set). Training then uses only the operator's
frames (`curate.dagger_frames: operator`) up to the level-1 click (`until_level: 1`).

### Isolation from the 4-brick setup

This experiment is a feasibility test: can the pipeline place *two* bricks at all? Everything two-brick is
in this folder's configs (`until_level: 1`, its own `out_dir`s: `data/episodes_retimed_v4`, `data/mix_v5`,
`data/dagger_v1`). The scene is unchanged (all 4 bricks still spawn; takes are only cut after level 1). The root
`config.yaml` stays the 4-brick setup (`until_level: null`, `out_dir: data/episodes`, teleop), and `data/episodes`
(the human 4-brick demos) is untouched. Back to 4 bricks = run without `--config` (or `QC_CONFIG`).
