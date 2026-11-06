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

(fill in)
