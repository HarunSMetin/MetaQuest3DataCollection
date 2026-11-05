# Diffusion Policy (`dp_240`)

Second policy family after ACT, trained on the same data (`data/lerobot/tower_v2`) and evaluated with the same
`qc eval`, so the two compare directly.

```bash
scripts/train_and_eval.sh tower_v2 dp_240        # overnight: ~9.5 h training + 2 x 50 rollouts
uv run qc train tower_v2 --preset dp_240 --dry-run   # just print the lerobot-train command
uv run qc eval outputs/<run> --n-action-steps 8      # re-plan more often than trained (16)
```

## Why Diffusion Policy, and why this variant

ACT on tower_v2 learned the gross motion but put bricks 6–8 mm off and never corrected
([../README.md](../README.md#act-on-tower_v2)). Two properties of
[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) (Chi et al., RSS 2023) target that:

- **Multimodal actions.** ACT regresses to the mean of the demonstrated actions (an L1 loss with a CVAE). Where
  demos differ (which arm moves first, how the hand approaches), the mean can fall between them. Diffusion
  samples one mode.
- **Short open-loop segments.** It re-plans from a fresh image every 16 steps (0.32 s) instead of every 100 steps
  (2 s). That gives the camera a chance to correct the last millimetres.

The candidates in the installed LeRobot 0.4.4:

| Policy | Verdict |
|---|---|
| **diffusion** (CNN U-Net, ResNet-18 per camera) | **chosen**: fits a 6 GB GPU and trains from scratch on 108 demos; it is the standard image-based baseline |
| diffusion with a transformer denoiser | not in LeRobot; the paper found the CNN version more robust and easier to tune |
| pi0 / pi0.5 / GR00T / X-VLA / WALL-X (VLAs, 2–3 B parameters) | do not fit a 6 GB GPU for fine-tuning |
| SmolVLA (450 M) | would need 512×512 images and a smaller batch; it uses language, and this task has one fixed instruction |
| VQ-BeT | less common, and weaker on fine manipulation in published comparisons |

## The settings

LeRobot's defaults are tuned for PushT at 10 Hz with one camera. What changed, and why:

| Setting | Value | Reason |
|---|---|---|
| `horizon` | 64 (1.28 s) | the paper's 16 steps at 10 Hz is 1.6 s; at 50 Hz that needs ~64 steps (a multiple of 2³ for the U-Net) |
| `n_action_steps` | 16 (0.32 s) | short open-loop segments for the placement; eval also tries 8 |
| `n_obs_steps` | 1 | two frames 20 ms apart are nearly identical, and `joint_vel` is already in the state. With 1 the batch can be 32 instead of 16 at the same step rate. |
| `drop_n_last_frames` | 48 | `horizon − n_action_steps − n_obs_steps + 1`; LeRobot does not derive it |
| `resize_shape`, `crop_ratio` | 240×320, 0.9 | the paper's random-crop augmentation (216×288) on the exported resolution; centre crop at inference |
| `use_separate_rgb_encoder_per_camera` | true | the top camera and the wrist cameras see very different images |
| backbone | ResNet-18 from scratch, GroupNorm, spatial softmax | as in the paper: BatchNorm breaks with the EMA and small batches, and ImageNet weights cannot be used with GroupNorm |
| `down_dims` | 256, 512, 1024 | the default 512–2048 does not fit 6 GB next to three image encoders (103 M parameters now) |
| `noise_scheduler_type` | DDIM, 100 train / 10 inference steps | ~10× faster sampling than DDPM: a chunk in well under 0.32 s, so it also runs live in `qc serve` |
| `batch_size`, `steps` | 32, 80k | measured: ~5.1 GB, 2.4 steps/s → ~9.5 h, 2.6 M samples (ACT got 1.6 M) |

Measured on the RTX 4050: batch 32 with `n_obs_steps` 2 runs out of memory; batch 16 with 2 runs at the same step
rate as batch 32 with 1.

Not available yet: **EMA** of the weights (the paper uses it; `lerobot-train` has no option). See the open ideas in
[../README.md](../README.md#open-ideas).

## What to expect

Diffusion is no cure for missing data. It can follow the demos more precisely and correct more often, but the
demos still contain no recovery from a brick stuck on the stud tops. If it also stalls there, recovery demos
(idea A) are the next step, for either policy. Compare the per-stage rates (`base_in_area`, `level_1`), not only
success.

## Results

| Run | Eval | Success | base in area | level 1 | level 2 | level 3 |
|---|---|---|---|---|---|---|
| | 16 steps per chunk | | | | | |
| | 8 steps per chunk | | | | | |
