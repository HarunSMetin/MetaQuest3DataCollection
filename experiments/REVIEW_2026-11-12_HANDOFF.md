# Review 2026-11-12: handoff (session interrupted, work in progress)

The review asked for in the brief (bugs, literature, plan, cleanup, `experiments/REVIEW_2026-11-12.md`) is about
two thirds done. This file is everything needed to continue it in a new session. Delete it when the review is
written.

Branch: `review/2026-11-12` (from `feature/audit-learn-safety`; first commit = snapshot of the work that was
uncommitted when the review started). Nothing is pushed. Nothing in `data/` or `outputs/` was modified; new
things there all start with `review_`.

## 1. Findings so far (all measured; numbers below are final unless marked)

Subject: the expert-data policy `outputs/20261111-1316_dp_240_pre_expert_v2` (level 1 = 2%).

1. **It fails on its own training scenes.** Restarted closed loop from recorded states of 6 training takes
   (`diagnostics/e2_restart.py`): clicked 0/6 from t = 0, 1/6 from 8 s before the click, 0/6 (6 s), 2/6 (4 s),
   1/6 (3 s), 1/6 (2 s), 5/6 (1 s). Blue ends 6–20 mm off, on the stud tops. So the failure does not need states
   "far from the data": it starts on the demo and leaves it within 1–2 s.
2. **The safety filter is not the cause.** Replaying the demo's targets through it is exact (0.0 mm); policy drift
   with the filter off is the same as with it on (`e3_drift.py`).
3. **Drift grows with how often the policy re-plans** (right-TCP distance from the demo, mm, median of 6 training
   takes, started 5 s before the click, after 16 / 32 / 48 / 64 / 96 steps):

   | steps per chunk | 16 | 32 | 48 | 64 | 96 |
   |---|---|---|---|---|---|
   | 64 (open loop) | 0.9 | 0.9 | 1.1 | 1.3 | 2.9 |
   | 32 | 0.9 | 0.9 | 1.3 | 1.3 | 3.9 |
   | 16 (trained) | 0.7 | 1.4 | 2.7 | 3.5 | 6.2 |
   | 8 | 1.7 | 2.8 | 4.4 | 5.6 | 11.5 |
   | 1 | 4.8 | 9.8 | 13.2 | 19.7 | 29.6 |

   Started 3.5 s before the click, 1 step per chunk reaches 123 mm after 64 steps. The existing evals agree: 8
   steps per chunk scored below 16 in every run evaluated both ways (16→8%, 14→8%, 2→0%, state oracle 22→2%).
4. **Joint velocity in `observation.state` is the main feedback path.** Same test, with one observation channel
   replaced by the demo's value at that step (`e4_channels.py`). Distance after 64 steps at 1 step per chunk,
   starts 5 s / 3.5 s: nothing forced 21.6 / 128.3 mm; velocity forced 7.6 / 8.3; position forced 6.4 / 11.6;
   images forced 3.6 / 62.3; wrench forced 159.5 / 120.7 (no help); position + velocity + wrench forced 2.9 / 2.5.
   At the trained 16 steps per chunk, after 128 steps: nothing forced 30.6 / 7.4; velocity forced 3.1 / 3.1;
   wrench forced 30.2 / 6.7; images forced 23.5 / 6.1.
5. **Why (data, `e5_shortcut.py`):** in the demos the joint target leads the measured joints by 0.12 s of motion:
   `target − q ≈ 0.12 s × q̇` explains 99.2% of the lead's variance in expert_v2 (96% in the retimed human takes,
   95.5% in the raw human takes). The expert's targets do not depend on the measured velocity (they follow a
   time programme); the velocity is their consequence. A policy that reads the action off the velocity has the
   causality backwards: a slowed arm is told to keep going slowly instead of being pulled back to the plan
   (causal confusion / copycat, de Haan et al. 2019, Wen et al. 2020).
6. **The eval rollouts show it (`e7_transit.py`):** final transit of the right arm, peak hand speed 0.229 m/s in
   the demos, 0.195 (16 per chunk) and 0.186 (8 per chunk) in the rollouts; blue is 10.7 mm from its goal (xy)
   when the descent starts in the demos, 29.6 mm and 41.7 mm in the rollouts; across rollouts slower transit goes
   with a larger shortfall (r = −0.62 and −0.61). This is the "lands 20–70 mm short on the approach side".
7. **"1.4 mm open-loop fit" is not evidence of precision (`e6_baselines.py`, now also printed by
   `scripts/fit_error.py`).** On the place window a predictor that knows nothing of the task, "keep coasting"
   (`q + 0.12 q̇`, continued), scores 0.1 / 0.4 / 0.6 mm (first step / mean / last of 16); the model scores
   2.0 / 1.6 / 1.6 on the same frames. The model only beats coasting outside the slow phases (other: 2.7 vs 5.9
   mean; grasp: 4.2 vs 7.0). Each fresh chunk starts 2–3 mm off the demo's next target (human-data model: 4–7 mm);
   that is the noise the loop in point 4 feeds on.
8. **Human-data policy (v4) is a different picture** (`diagnostics/e4_v4_nas16.log`): its open-loop error on
   training takes is already 3–6 mm at every step, the closed loop is only moderately worse (4–10 mm over 128
   steps), and forcing velocity does not help in that window. Its first problem is the fit itself. Not finished:
   the chunk-length sweep for v4 (see section 3).
9. **Under-training in epochs (likely contributor to 8, not tested):** 80k steps × batch 32 is 62 epochs of
   two_bricks_v4 (41k frames) and 15 epochs of expert_v2 (169k frames). Diffusion Policy trains image policies for
   3000 epochs (Chi et al., Sec. 5.2); JUICER up to 400 epochs at batch 256 on 50 demos.
10. **LeRobot does not normalize `observation.environment_state`** (verified in the installed code:
    `normalize_processor.py` uses `norm_map.get(feature_type, IDENTITY)` and the diffusion default mapping has no
    `ENV` entry). Effect on the state-oracle results not measured. Fix used in the new preset:
    `policy.normalization_mapping` with `ENV: MIN_MAX`.
11. Checked and found correct: observation t ↔ action t alignment in export, rollout and LeRobot's
    `action_delta_indices`; `executed_targets` (ctrl of the next state); gripper labels; centre crop at inference;
    ImageNet stats; `drop_n_last_frames`. Normalization resolution: 1 mm at the TCP is 0.001–0.002 of the [−1, 1]
    action range.

## 2. The decisive test did not finish: start it again

Two camera-free trainings on the expert takes, to test whether removing joint velocity fixes the closed loop
(brick poses as input, preset `dp_state1`, 20k steps). They were stopped at step ~5400 of 20000 because the laptop
had to be shut down, before the first checkpoint, so nothing of them can be resumed. The empty run folders
`outputs/review_oracle_vel` and `outputs/review_oracle_novel` (only a manifest) and their logs are left as they
are; ignore or delete them.

The datasets are exported and ready: `data/lerobot/review_expert_state_vel` (state = joints + grip + velocity +
wrench) and `data/lerobot/review_expert_state_novel` (joints + grip + wrench).

Start both again under new names (in parallel: ~2.9 GB GPU, ~4 steps/s each, ~85 min; checkpoints every 5k):

```bash
for v in vel novel; do
  nohup uv run --extra learn qc train review_expert_state_$v --preset dp_state1 --save_freq=5000 \
      --config experiments/obs_ablation/config_$v.yaml --name review_oracle2_$v > outputs/review_oracle2_$v.log 2>&1 &
done
tail -c 300 outputs/review_oracle2_vel.log        # progress
```

A stopped run with a checkpoint resumes with
`.venv/bin/lerobot-train --config_path=outputs/<run>/train/checkpoints/last/pretrained_model/train_config.json --resume=true`.

## 3. What is left

1. When both trainings are done, evaluate them (the decisive number; expert itself scores 100% on these seeds):
   ```bash
   uv run --extra learn qc eval outputs/review_oracle2_vel   --config experiments/obs_ablation/config_vel.yaml   --n 50
   uv run --extra learn qc eval outputs/review_oracle2_novel --config experiments/obs_ablation/config_novel.yaml --n 50
   MUJOCO_GL=egl uv run --extra learn python scripts/drift_test.py outputs/review_oracle2_novel data/expert_v2 --config experiments/obs_ablation/config_novel.yaml
   ```
   Read: `novel` clearly above `vel` on level 1, and its drift no longer growing with re-planning → retrain the
   camera policy without velocity (next point). No difference → velocity is a symptom, not the cause; then try
   `config_posonly.yaml` and look at the position channel.
2. If confirmed, the overnight run: export expert_v2 with cameras and `learn.joint_vel: false` (new config next to
   `config_novel.yaml`), train `dp_240_pre`, evaluate. Ask the user before starting it (9.5 h GPU).
3. Drift test of the human-data policy (was stopped half way):
   `MUJOCO_GL=egl uv run --extra learn python scripts/drift_test.py outputs/20261106-2147_dp_240_pre_two_bricks_v4 data/episodes_retimed_v4 --n 6 --starts 6 4 --steps 192`
4. Cheap evals of the existing models with longer chunks (`qc eval ... --n 20 --n-action-steps 32` and `64`).
5. Write `experiments/REVIEW_2026-11-12.md` (findings verified / likely / speculative, literature, ranked plan with
   cheap tests and stop criteria, cleanup done and proposed) and `experiments/obs_ablation/README.md`; add both to
   the log in `experiments/README.md`; mention `scripts/drift_test.py` in `CLAUDE.md`.
6. Cleanup still open: `scripts/train_and_eval.sh` second eval (half the chunk length → longer chunks) with its
   README line; two README edits sit uncommitted in the working tree (`experiments/diffusion/README.md`,
   `experiments/state_oracle/README.md`): they cite the oracle result, so commit them only once it is in.
   Proposed, not to be done without asking: one control-step helper shared by `runtime._tick`,
   `evaluate.rollout`, `expert.run_episode` and `generate.replay`; `scripts/generate_*.py` as `qc generate`;
   the default presets duplicated in `config.py` and `config.yaml`.
7. Final test run (`MUJOCO_GL=egl uv run --extra learn --group dev pytest`; baseline was 77 passed) and the short
   Turkish summary for the user with the top 3 next steps and their commands.

## 4. Literature collected (for the review's source list)

- Chi et al., Diffusion Policy (arXiv 2303.04137): position control beats velocity control (Sec. 4.2); action
  horizon 8 of 16 at 10 Hz (Sec. 5.3); 3000 / 4500 epochs for image / state policies (Sec. 5.2); real robot at
  10 Hz interpolated to 125 Hz (Sec. 6.1).
- Zhao et al., ACT / ALOHA (arXiv 2304.13705): observation = joint positions + 4 cameras, no velocities
  (Sec. IV); actions = leader joint positions at 50 Hz, chunk 100; sim bimanual insertion 20–32%, real slot
  battery 96%, thread velcro 20% (Tables I, II); chunking 1% → 44% (Sec. VI-A).
- Ankile et al., JUICER (arXiv 2404.03729): 50 demos per task, DP one_leg 59%, round_table 18%; trajectory
  augmentation around annotated bottleneck states (reverse of a "disassembly" path) and collect-and-infer raise
  them to 74% and 32% (Table VII); observation horizon 1, predict 32 / execute 8, batch 256, up to 400 epochs.
- Ankile et al., From Imitation to Refinement / ResiP (arXiv 2407.16677): BC plateaus (one_leg 54%, peg-in-hole
  5%); a per-step residual PPO policy on the frozen chunked diffusion policy reaches 98% / 99% (Table I), at
  1024 parallel environments and up to 500M steps (Table VI); 10 Hz control except the bimanual task (50 Hz).
- de Haan, Jayaraman, Levine, Causal Confusion in Imitation Learning (NeurIPS 2019, arXiv 1905.11979); Wen et al.,
  Fighting Copycat Agents in Behavioral Cloning from Observation Histories (NeurIPS 2020, arXiv 2010.14876);
  Chuang et al., Resolving Copycat Problems via Residual Action Prediction (arXiv 2207.09705).
- Laskey et al., DART (CoRL 2017, arXiv 1703.09327); Kelly et al., HG-DAgger (ICRA 2019); Mandlekar et al.,
  Human-in-the-Loop Imitation Learning / IWR (arXiv 2012.06733); Chi et al., UMI (arXiv 2402.10329: relative
  trajectory actions, latency matching).
- Still to check: Octo appendix on proprioception (seen only second hand), APEX-MR (arXiv 2503.15836, bimanual
  LEGO with a press primitive), BrickCraft (arXiv 2605.07605).

## 5. Commits on the branch so far

- Snapshot of the uncommitted work before the review
- tests: default to headless GL (the suite aborted in `test_learn` without `MUJOCO_GL`)
- retime: remove the unused `_Timeline.to_new`
- config: the three diffusion presets written once (YAML anchors; loaded config identical)
- diagnostics: baselines in `fit_error.py`, new `scripts/drift_test.py`
- this handoff, `experiments/obs_ablation/` configs and the one-off diagnostic scripts
