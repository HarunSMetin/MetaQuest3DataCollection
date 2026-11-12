# Scripted expert (two-brick task)

`expert.py` is a privileged scripted expert. It reads the true brick poses and drives both arms through the same
IK → SafetyFilter path as the operator: the left arm places red in the build area at about 90°, the right arm presses
blue onto it. `run_expert.py` tests the expert or records its successful takes as ordinary episodes. Everything here
is opt-in: own config (`config.yaml`, until_level 1), own data dirs (`data/expert_v2`), nothing at the root changes.

    cd experiments/expert_dagger
    MUJOCO_GL=egl uv run python run_expert.py --eval-seeds 50
    MUJOCO_GL=egl uv run python run_expert.py --n 200 --out ../../data/expert_v2      # --jobs 3 max (14 GB RAM)

## E1: Diffusion on 200 expert demos (2026-11-11, `outputs/20261111-1316_dp_240_pre_expert_v2`)

Question: is the human data the problem (inconsistent demos), or is it the model?

Training setup:
- `dp_240_pre`, 80k steps, data `expert_v2`;
- 200 takes, all clicked, click at 16 s (median);
- 169k frames.

| | base in area | level 1 |
|---|---|---|
| expert itself | 100% | 100% |
| policy, chunks as trained (16) | 76% | **2%** |
| policy, 8 steps per chunk | 62% | 0% |
| human-data baseline (two_bricks v4 60k) | 84% | 14% |

The eval horizon is 28 s, and the expert clicks at 16 s. The policy reaches the red brick at 16 s (median); only 2 of
50 rollouts arrive after 24 s. So the time limit is not the cause.

**Open-loop error on the training demos** (`scripts/fit_error.py`, 8 episodes, mm at the TCP):

| phase | median error | p90 |
|---|---|---|
| grasp | 3.8 | 9 |
| place | **1.4** | 2.2 |

The human-trained model had about 5 mm here. So the model fits the expert to the millimetre.

**Closed loop:** blue lands on red 20–70 mm short along red's long axis. Red is at about 90°, so that axis points
toward the right arm. In other words, blue lands on the side the right arm approaches from. The goal is −8 mm in red's
frame. Blue then sits on the stud tops (dz 20–25 mm instead of 19) and stays there with the gripper closed until the
time runs out.

In the timelines the arm already descends while it is still 3–6 cm short. The expert first aligns at hover height and
only then descends. The policy cuts that corner, and once blue touches the studs it cannot slide it any more.

**Reading:**
- Not underfitting, not the model's capacity: the policy reproduces the expert's frames to 1–2 mm.
- The problem is closed-loop compounding (covariate shift). The expert's takes are almost identical, so the data
  contains no state "a few cm off, not yet aligned" and no correction from there.
- The human demos are messier and contain corrections, which is probably why they did better (14%).
- More of the same perfect data will not help. The data has to cover the states the policy actually ends up in.

**Next:**
1. **Expert v3 with perturbations (DART):** put noise on the expert's executed targets and on its approach waypoints,
   and label with the expert's clean targets. Then every take holds corrections from off-nominal states. Cheap: CPU
   only, overnight.
2. **E2 automatic DAgger:** the policy drives, the expert labels (and takes over near failure). The expert must react
   to the current state, not just run its phase programme.
