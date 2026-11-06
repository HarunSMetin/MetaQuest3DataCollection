# Experiments

What was tried with the learning pipeline, why, and what came out. Code stays in `questcollect/`; an experiment is
a preset in `config.yaml` (`learn.presets`) plus a note here. Run one with
`scripts/train_and_eval.sh <dataset> <preset>`; results land in `outputs/<run>/RESULTS.txt`.

## Log

| Date | Run | Dataset | Policy | Success | Note |
|---|---|---|---|---|---|
| 2026-10-27 | (archived) | tower_v1: 87 teleop demos | ACT `act_240` | 0/50 | no generalization: misses bricks by 3–6 cm |
| 2026-11-02 | `20261031-2200_act_240_tower_v2` | tower_v2: 108 retimed, generated | ACT `act_240` | 0/50, 0/50 (ensemble) | picks and carries; placement 6–8 mm off ([below](#act-on-tower_v2)) |
| 2026-11-02 | `20261102-0505_dp_240_tower_v2` | tower_v2 | Diffusion `dp_240` | 0/50 | base in area 48%, level 1 4%; the blue brick is never lifted in 29/50 ([diffusion/](diffusion/README.md)) |
| 2026-11-03 | `20261103-0116_dp_state_tower_v2_state` | tower_v2_state | Diffusion `dp_state`, brick poses instead of cameras | 0/50 | level 1 22% (cameras: 0–4%), still 7 mm off; action side is the larger bottleneck ([state_oracle/](state_oracle/README.md)) |
| 2026-11-06 | `20261106-0108_dp_240_tower_v3` | tower_v3: 71 unassisted hand demos (DART on), executed-target labels | Diffusion `dp_240`, joint actions + joint_vel + hand wrench | (running) | [tower_v3/](tower_v3/README.md) |

### ACT on tower_v2

The base brick reached the build area in 80% of rollouts. Where the second (blue) brick ended up:

| Outcome | Chunks as trained | Ensemble 0.01 |
|---|---|---|
| on the stud tops (dz = +4.5 mm = `STUD_H`), not seated | 17 | 20 |
| near the base, not on it | 13 | 12 |
| never reaches the base (missed grasp, dropped, flung) | 20 | 18 |

The rollouts that ended on the stud tops were off by a median 6–8 mm in xy. Seating needs ≤ 1.8 mm (the
`STUD_CHAMFER` lead-in) and `CLUTCH_ALIGN_XY` 1 mm. The training demos are 0.0–0.3 mm off. Once a brick sits on the
studs, the policy never lifts it to try again: no demo shows that.

### Stud lead-in test (2026-11-02)

How far off can a free brick land and still seat, when pressed with 10 N? (`sim.STUD_CHAMFER` patched; not
changed in `sim.py`.)

| Chamfer | Seats up to (along x / along y / diagonal) |
|---|---|
| 1.5 mm (current) | 3 mm / 3 mm / 4 mm |
| 2.5 mm | erratic: fails at 1–3 mm in y; the frustum catches on the walls/tubes |
| 3.5 mm | 5 mm in every direction (a nearly conical stud: 1 mm of cylinder left) |

The current geometry already catches ~3 mm, not the 1.8 mm the chamfer alone suggests. The free brick tilts and
slides.

The trained policies under a 3.5 mm chamfer, on the first 20 eval seeds (same seeds at 1.5 mm in brackets):

| | base in area | blue seated | level 1 |
|---|---|---|---|
| ACT | 14/20 (17) | 4/20 (0) | 0/20 (0) |
| Diffusion | 8/20 (11) | 0/20 | 0/20 (1) |

ACT seats the blue brick perfectly (0.1–0.3 mm, right height) in 4/20, yet it never clicks. Replaying those
rollouts shows why:
- The press force between the bricks while seated is ~0.1 N. Clicking needs `CLUTCH_PRESS_N` = 6 N, and the demos
  press with ~86 N.
- The policy opens the gripper 1–4 cm above the base and drops the brick. With the bigger chamfer it falls into
  place but sits loose, like a real brick that was not pressed.
- It also grasps the blue brick ~19 mm off its centre. The demos grasp within ~1.5 mm, thanks to the grasp assist.

In the demos, by contrast, the hand rests the brick on the stud tops for ~1 s and pushes down until it slides in
and clicks. Only then does it open. That trigger, the click, is a 4.5 mm drop and almost invisible in 240p
images. The policy has no other cue.

**Conclusion:** the tolerance is not the main bottleneck. The policies skip the press-until-click phase, and they
grasp off-centre.

## Open ideas

- **D. Give the policy a contact cue.** Add the measured hand force to `observation.state`. A real Franka estimates
  it as well (`O_F_ext_hat_K`), so this is not privileged. The policy can then learn "press until the force spike,
  then let go". A cheaper sim-only probe would be the click state (`clutch`).
- **E. Consistent grasps without assist.** Done differently (2026-11-03): the grasp assist is gone. It centred
  every demo grasp to ~1 mm, so the data never showed an off-centre grasp or its correction, and the policies
  (without it) grasped 10–19 mm off. New demos are recorded unassisted, through the safety filter; the arm bases
  moved 5 cm back (`sim.arm_base_x`) and the IK got a damping task, so the old data's scene differs.

Not started yet; ranked by how directly they address the ACT failure above.

- **A. Recovery demos.** Let `generate.py` place a share of the takes deliberately 3–8 mm off. When the brick lands
  on the stud tops, lift it slightly, correct and press again. This teaches the state the policy actually ends up in
  (automated DAgger).
- **B. A realistic stud lead-in.** Real DUPLO centres a brick from much further off than 1.8 mm. Raise
  `STUD_CHAMFER` / `CLUTCH_ALIGN_XY` in `sim.py`. This changes the task and the `scene_hash`, so the training set
  has to be regenerated. Decide with the supervisor.
- **C. A sharper view while placing.** Larger wrist-camera images (240×320 → 480×640, which needs a smaller batch
  or model), or shorter chunks while placing.
- **EMA weights for diffusion.** The original Diffusion Policy averages weights (EMA). `lerobot-train` 0.4.4 has
  no EMA option, so it needs a small training wrapper.
