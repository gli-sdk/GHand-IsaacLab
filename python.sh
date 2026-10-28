#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/workspace/IsaacLab}"
if [[ ! -f "$ISAACLAB_PATH/source/isaaclab/isaaclab/__init__.py" ]]; then
  echo "Set ISAACLAB_PATH to your Isaac Lab checkout (validated version: 2.3.0)." >&2
  exit 2
fi
ISAACLAB_PATH="$(cd "$ISAACLAB_PATH" && pwd)"
export ISAACLAB_PATH
SIM_PYTHON="${ISAACSIM_PYTHON:-$ISAACLAB_PATH/_isaac_sim/python.sh}"
if [[ ! -x "$SIM_PYTHON" ]]; then
  echo "Set ISAACSIM_PYTHON to the Python executable/launcher for your Isaac Sim environment." >&2
  exit 2
fi
export PYTHONPATH="$PROJECT_DIR/source:$ISAACLAB_PATH/source/isaaclab:$ISAACLAB_PATH/source/isaaclab_assets:$ISAACLAB_PATH/source/isaaclab_tasks:$ISAACLAB_PATH/source/isaaclab_rl${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$PROJECT_DIR"
exec env -u CONDA_PREFIX "$SIM_PYTHON" "$@"
