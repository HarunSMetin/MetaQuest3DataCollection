#!/usr/bin/env bash
# One unattended night for the camera policy without joint velocity (README.md here):
#   train (scripts/train_and_eval.sh) -> pause at a checkpoint -> evaluate that checkpoint -> resume to the end.
# The pause gives an early number that can be compared with the same checkpoint of the run with velocity.
#
#   experiments/obs_ablation/night_run.sh [dataset=expert_v2_novel] [preset=dp_240_pre] [pause_at=030000]
#
# Detaches and keeps the laptop awake. Log: outputs/review_night.log. Safe to interrupt at any time:
#   pause:   pkill -f night_run.sh; pkill -f train_and_eval.sh; pkill -f 'bin/lerobot-train'
#   resume:  QC_CONFIG=experiments/obs_ablation/config_cam_novel.yaml scripts/train_and_eval.sh --resume outputs/<run>
# (continues from the last checkpoint, saved every 10k steps, and evaluates at the end).
set -uo pipefail
self="$(realpath "$0")"
cd "$(dirname "$self")/../.."
dataset=${1:-expert_v2_novel}
preset=${2:-dp_240_pre}
pause_at=${3:-030000}
log=outputs/review_night.log

if [[ -z "${QC_NIGHT_DETACHED:-}" ]]; then
    QC_NIGHT_DETACHED=1 setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch --who=questcollect \
        --why="obs_ablation night run" "$self" "$@" >>"$log" 2>&1 </dev/null &
    echo "started in the background (pid $!); follow: tail -f $log"
    exit 0
fi
export QC_CONFIG=experiments/obs_ablation/config_cam_novel.yaml
export QC_TRAIN_ARGS="--save_freq=10000"
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }
training() { pgrep -f 'bin/lerobot-train' >/dev/null; }

stamp "start: $dataset with $preset, pause at $pause_at"
scripts/train_and_eval.sh "$dataset" "$preset" 50
sleep 180
run=$(ls -d outputs/*_"${preset}"_"${dataset}" 2>/dev/null | tail -n 1)
if [[ -z $run ]] || ! training; then
    stamp "the training did not start (see outputs/latest.log)"
    exit 1
fi
stamp "run: $run"

# `last` is re-pointed only after a checkpoint is written completely.
until [[ $(readlink "$run/train/checkpoints/last" 2>/dev/null) == "$pause_at" ]] || ! training; do sleep 30; done
if [[ $(readlink "$run/train/checkpoints/last" 2>/dev/null) != "$pause_at" ]]; then
    stamp "the training ended before checkpoint $pause_at; nothing more to do here (train_and_eval.sh evaluates)"
    exit 0
fi
stamp "checkpoint $pause_at saved: pausing the training to evaluate it"
pkill -f train_and_eval.sh
pkill -f 'bin/lerobot-train'
for _ in $(seq 60); do training || break; sleep 2; done
sleep 10
uv run --extra learn qc eval "$run/train/checkpoints/$pause_at/pretrained_model" --config "$QC_CONFIG" --n 50 \
    | tail -n 12 || stamp "eval of checkpoint $pause_at failed"
stamp "resuming $run"
scripts/train_and_eval.sh --resume "$run" 50
sleep 180
if training; then stamp "training is running again; it evaluates itself at the end ($run/RESULTS.txt)"; else
    stamp "the training did NOT come back: run  QC_CONFIG=$QC_CONFIG scripts/train_and_eval.sh --resume $run"; fi
