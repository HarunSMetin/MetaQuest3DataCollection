"""`lerobot-train` with an exponential moving average (EMA) of the policy weights.

The original Diffusion Policy evaluates the EMA of the weights, not the last ones; LeRobot's diffusion policy has
no EMA. This wrapper runs lerobot's own training script unchanged, but
- after every optimizer step updates the average: ema <- d * ema + (1 - d) * weights, with the warm-up schedule of
  the Diffusion Policy code (diffusers EMAModel): d = min(max_decay, 1 - (1 + step) ** -power), power 0.75,
- saves the averaged weights as each checkpoint's pretrained_model (what `qc eval` and `qc serve --policy` load);
  the raw weights go next to them as model_raw.safetensors.

Started by `qc train` (learn/train.py) when the preset has `ema: <max_decay>`; the other arguments are lerobot-train's.
Resuming such a run restarts from the averaged weights.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

import lerobot.scripts.lerobot_train as lt

MAX_DECAY = float(os.environ.get("QC_EMA_MAX_DECAY", "0.9999"))
POWER = 0.75


class Ema:
    def __init__(self):
        self.shadow: dict[str, torch.Tensor] | None = None
        self.step = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        if self.shadow is None:
            self.shadow = {k: v.detach().clone().float() if v.is_floating_point() else v.detach().clone()
                           for k, v in state.items()}
            return
        self.step += 1
        d = min(MAX_DECAY, 1.0 - (1.0 + self.step) ** -POWER)
        for k, v in state.items():
            s = self.shadow[k]
            if v.is_floating_point():
                s.mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:  # e.g. BatchNorm's num_batches_tracked
                s.copy_(v)

    @torch.no_grad()
    def swap(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Load the averaged weights into the model; returns the raw ones to put back."""
        raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: self.shadow[k].to(raw[k].dtype) for k in raw})
        return raw


EMA = Ema()
_update_policy, _save_checkpoint = lt.update_policy, lt.save_checkpoint


def update_policy(train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator, *args, **kwargs):
    out = _update_policy(train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator, *args, **kwargs)
    EMA.update(accelerator.unwrap_model(policy, keep_fp32_wrapper=False))
    return out


def save_checkpoint(checkpoint_dir: Path, step, cfg, policy, *args, **kwargs):
    if EMA.shadow is None:
        return _save_checkpoint(checkpoint_dir, step, cfg, policy, *args, **kwargs)
    from safetensors.torch import save_file

    raw = EMA.swap(policy)
    try:
        _save_checkpoint(checkpoint_dir, step, cfg, policy, *args, **kwargs)
    finally:
        policy.load_state_dict(raw)
    pretrained = Path(checkpoint_dir) / "pretrained_model"
    save_file({k: v.contiguous() for k, v in raw.items()}, str(pretrained / "model_raw.safetensors"))
    (pretrained / "ema.txt").write_text(f"EMA weights, max decay {MAX_DECAY}, power {POWER}, {EMA.step} updates\n")


lt.update_policy, lt.save_checkpoint = update_policy, save_checkpoint

if __name__ == "__main__":
    lt.main()
