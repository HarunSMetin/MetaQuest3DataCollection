#!/usr/bin/env bash
# Sparse-clone the Franka Panda model from mujoco_menagerie into assets/third_party.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/assets/third_party/mujoco_menagerie"
if [ -d "$DEST/franka_emika_panda" ]; then
  echo "menagerie panda already present at $DEST"; exit 0
fi
mkdir -p "$ROOT/assets/third_party"
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$DEST"
git -C "$DEST" sparse-checkout set franka_emika_panda
echo "done: $DEST"
