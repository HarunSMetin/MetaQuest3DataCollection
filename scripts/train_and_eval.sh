#!/usr/bin/env bash
# Unattended run: train a dataset with a preset, then evaluate the last checkpoint (plain, then with ACT temporal
# ensembling, or for other policies re-planning twice as often) and write a short results file. Detaches from the
# terminal (closing it is fine) and keeps the laptop awake, also with the lid closed, until done.
#
#   scripts/train_and_eval.sh <dataset> [preset=act_240] [rollouts=50]
#   scripts/train_and_eval.sh --resume outputs/<run> [rollouts=50]   # continue from its last checkpoint
#   QC_CONFIG=experiments/<x>/config.yaml scripts/train_and_eval.sh ...  # presets from another config
#
# Log: outputs/<run>.log (outputs/latest.log points to it). Results: outputs/<run>/RESULTS.txt.
# Stop training early with `pkill -f train_and_eval.sh; pkill -f lerobot-train`, and resume later with --resume
# (from the last saved checkpoint, every 20k steps). `pkill -f lerobot-train` alone evaluates that checkpoint now.
set -uo pipefail
self="$(realpath "$0")"
cd "$(dirname "$self")/.."

if [[ ${1:-} == --resume ]]; then
    run=${2:?usage: scripts/train_and_eval.sh --resume outputs/<run> [rollouts]}
    run=${run%/}
    name=$(basename "$run")
    n=${3:-50}
    resume=1
    [[ -e "$run/train/checkpoints/last" ]] || { echo "no checkpoint in $run/train/checkpoints to resume from"; exit 1; }
else
    dataset=${1:?usage: scripts/train_and_eval.sh <dataset> [preset] [rollouts]}
    preset=${2:-act_240}
    n=${3:-50}
    name="$(date +%Y%m%d-%H%M)_${preset}_${dataset}"
    resume=0
fi

if [[ -z "${QC_DETACHED:-}" ]]; then  # start a detached copy of this script under a sleep/lid inhibitor
    if pgrep -f lerobot-train >/dev/null; then
        echo "lerobot-train is already running (pgrep -fa lerobot-train); stop it first, the GPU fits one"
        exit 1
    fi
    mkdir -p outputs
    ln -sfn "$name.log" outputs/latest.log
    QC_DETACHED=1 setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch --who=questcollect \
        --why="qc train + eval $name" "$self" "$@" >>"outputs/$name.log" 2>&1 </dev/null &
    echo "started in the background (pid $!): outputs/$name"
    echo "follow:  tail -f outputs/latest.log      results at the end: outputs/$name/RESULTS.txt"
    exit 0
fi
run="outputs/$name"
cfg=(); [[ -n ${QC_CONFIG:-} ]] && cfg=(--config "$QC_CONFIG")
qc() { uv run --extra learn qc "$@" "${cfg[@]}"; }
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

if (( resume )); then
    stamp "resume $run from checkpoint $(readlink "$run/train/checkpoints/last")"
    .venv/bin/lerobot-train --config_path="$run/train/checkpoints/last/pretrained_model/train_config.json" \
        --resume=true
else
    stamp "train $dataset with $preset -> $run"
    qc train "$dataset" --preset "$preset" --name "$name"
fi
status=$?
if (( status == 0 )); then
    stamp "training finished"
else
    stamp "training stopped with status $status; evaluating the last saved checkpoint if there is one"
fi
if [[ ! -e "$run/train/checkpoints/last" ]]; then
    stamp "no checkpoint saved, nothing to evaluate"
    exit 1
fi

stamp "eval: $n rollouts, chunks as trained"
qc eval "$run" --n "$n" || stamp "eval failed"
policy_cfg="$run/train/checkpoints/last/pretrained_model/config.json"
policy=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['type'])" "$policy_cfg")
if [[ $policy == act ]]; then
    stamp "eval: $n rollouts, temporal ensembling 0.01"
    qc eval "$run" --n "$n" --ensemble 0.01 || stamp "eval (ensemble) failed"
else
    steps=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['n_action_steps'] // 2)" "$policy_cfg")
    stamp "eval: $n rollouts, re-planning every $steps steps"
    qc eval "$run" --n "$n" --n-action-steps "$steps" || stamp "eval (n_action_steps $steps) failed"
fi

stamp "results"
python3 - "$run" <<'EOF' | tee "$run/RESULTS.txt"
import json, sys
from pathlib import Path

run = Path(sys.argv[1])
print(f"run: {run}")
for path in sorted(run.glob("eval/*/metrics.json")):
    m = json.loads(path.read_text())
    s, r = m["settings"], m["summary"]
    mode = (f"temporal ensembling {s['temporal_ensemble']}" if s.get("temporal_ensemble")
            else f"{s['n_action_steps']} steps per chunk" if s.get("n_action_steps") else "chunks as trained")
    stages = "  ".join(f"{k} {v:.0%}" for k, v in r["stage_rates"].items())
    print(f"\n[{mode}]  {s['n']} rollouts  ->  success {r['success_rate']:.0%}\n  {stages}\n"
          f"  rollouts with contact: {r['rollouts_with_contact']}  safety mean {r['safety_mean_rad']:.4f} rad, "
          f"max {r['safety_max_rad']:.3f} rad\n  {path}")
EOF
stamp "done"
