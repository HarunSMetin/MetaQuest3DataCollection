# State oracle: is it the images, or the data?

ACT and Diffusion both scored 0/50 on tower_v2, and they fail at different stages:

| | ACT `act_240` | Diffusion `dp_240` |
|---|---|---|
| base in area | 80% | 48% |
| blue brick never lifted | 10 | 29 |
| blue lifted, never near the base | 10 | 13 |
| blue on the stud tops, not seated | 15 (median 5.7 mm off) | 4 |
| blue seated | 2 | 3 |

ACT also fails when it starts from the exact first frame of a training episode (2/2 tried).

This experiment trains the same Diffusion Policy without cameras. It gets the true brick poses instead
(`learn.env_state`: 4 × position + quaternion). Perception is then perfect, and only the data, the task and the
policy class remain.

```bash
MUJOCO_GL=egl uv run --extra learn qc export tower_v2_state --raw data/mix_parallel --config experiments/state_oracle/config.yaml
QC_CONFIG=experiments/state_oracle/config.yaml scripts/train_and_eval.sh tower_v2_state dp_state
```

`config.yaml` here is the main config with only `learn` changed: no cameras, `env_state: true`, and one preset
`dp_state` (no image encoder, so batch 64, 60k steps ≈ 2 h on the RTX 4050 at ~9 steps/s).

## How to read the result

- **The oracle builds the tower (or at least places level 1 most of the time).** The data is learnable, and the
  bottleneck is seeing the bricks precisely from 240p images. Next: more image data per behaviour, pretrained or
  larger visual encoders, higher-resolution wrist cameras (idea C), image augmentation.
- **The oracle fails in the same places.** The bottleneck is the data or the task, not the vision. Candidates, in
  the order to test them:
  1. Too few distinct behaviours: 108 takes come from only 11 human demos.
  2. Retimed takes make timing multimodal (which arm moves when differs per take).
  3. The 1.8 mm seating tolerance is stricter than real DUPLO (idea B).
  4. There are no recoveries in the data (idea A).

## Results

| Run | Eval | Success | base in area | level 1 | level 2 | level 3 |
|---|---|---|---|---|---|---|
| `20261103-0116_dp_state_tower_v2_state` | 16 steps per chunk | 0% | 90% | 22% | 0% | 0% |
| | 8 steps per chunk | 0% | 92% | 2% | 0% | 0% |

Level 1 rose from 0–4% (cameras) to 22%: perception is part of the problem. With exact brick poses, though, the
blue brick still lands on the stud tops in 18/50 rollouts, a median 7.3 mm off. It is never lifted in 12/50. Of
the 11 that reached level 2, none placed the yellow brick (5 on the stud tops, median 14 mm off).

**Reading:** both perception and data/action are bottlenecks, and the larger one is the action side. A policy
that knows every brick pose to the micrometre still places 7 mm off, because it has to map brick poses to
14 absolute joint angles at mm precision from 108 demos. The next test is end-effector actions (`learn.action:
ee`). The TCP target is then nearly a shift of the brick pose, which is far easier to learn. Before that, the
rotation needs a continuous representation: the recorded TCP quaternions have w ≈ 0 (gripper pointing down) and
flip sign 6002 times between consecutive steps, so a network would average q and −q.

## Next: the same oracle with end-effector actions

`config_ee.yaml` = `config.yaml` here with `learn.action: ee`. The policy outputs per arm the TCP position and a
6D rotation (the first two rotation-matrix columns). `learn/features.py` turns that back into a quaternion for the
teleop IK. Checks before training:
- `tests/test_features.py` passes.
- Recorded TCP targets replayed through the 6D features and the IK rebuild the tower: episodes 0 and 50 reach
  3/3 and episode 90 reaches 2/3, open loop.
- Rotation features change ≤ 0.19 per step; only the gripper commands jump.

```bash
MUJOCO_GL=egl uv run --extra learn qc export tower_v2_state_ee --raw data/mix_parallel --config experiments/state_oracle/config_ee.yaml
QC_CONFIG=experiments/state_oracle/config_ee.yaml scripts/train_and_eval.sh tower_v2_state_ee dp_state
```

Compare with the joint-action oracle above: level 1, and the blue brick's xy error on the stud tops (7.3 mm).

## Results with end-effector actions (2026-11-03)

`20261103-0942_dp_state_tower_v2_state_ee`, eval stopped at 40/50 (no metrics.json; rollouts in `eval/`).
- Base in area 26/26, one full tower (3/3, the first success of any policy), every other rollout 0/3.
- Blue brick (first 25): on the stud tops 9, median **7.9 mm** off (joint actions: 7.3 mm); never lifted 8.
- The policy grasps the blue brick a median 10 mm off-centre (demos: 0.9 mm, thanks to the grasp assist).
- On its own training frames, just before a grasp, a sampled chunk is 2.1 mm off the demo in xy (p90 3.5 mm),
  4.5 mm in z. More denoising steps (DDIM 30/100, DDPM 100) don't help: the fit, not the sampler.

`qc eval --assist` ran the grasp assist on the policy's TCP targets (ee policies, sim only; the option went with the
assist on 2026-11-03, and the configs here lost their unused `assist:` block on 2026-11-13). On 3 rollouts the
grasp offset drops to 1.2 mm, but the blue brick still lands 6.7 mm off: the placement itself is off.

**Reading:** the action representation was not the problem. Placement relative to the base is not learned to
mm precision. One suspect: LeRobot does not normalize `observation.environment_state` (its
`normalization_mapping` has no ENV entry), so a 1 mm base shift is a 0.001 change of a raw input.
