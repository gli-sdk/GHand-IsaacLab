#!/usr/bin/env bash
# Launch a frozen GHand policy in the existing local Isaac Lab runtime.
set -euo pipefail
GHAND_DELIVERY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  echo 'Usage: bash run.sh [num_envs] [--episodes N] [--seconds N] [--seed N] [--headless] [--dry-run]'
  echo 'Defaults: 16 environments, 100 episodes per environment, 30 seconds per episode, seed 321.'
}
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
GHAND_BUNDLE="$GHAND_DELIVERY/models/cube_reorientation/teacher"
GHAND_TASK=Isaac-Reorient-Cube-GHand-v0
GHAND_ENVS=16
if [[ $# -gt 0 && $1 != --* ]]; then GHAND_ENVS=$1; shift; fi
GHAND_EPISODES=100; GHAND_SECONDS=30; GHAND_SEED=321; GHAND_DRY=0
GHAND_EXTRA=()
while [[ $# -gt 0 ]]; do
 case "$1" in
  --headless) GHAND_EXTRA+=(--headless); shift ;;
  --dry-run) GHAND_DRY=1; shift ;;
  --episodes|--seconds|--seed)
   [[ $# -ge 2 ]] || { usage; exit 2; }
   case "$1" in --episodes) GHAND_EPISODES=$2 ;; --seconds) GHAND_SECONDS=$2 ;; --seed) GHAND_SEED=$2 ;; esac
   shift 2 ;;
  *) usage; exit 2 ;;
 esac
done
python3 - "$GHAND_BUNDLE" "$GHAND_TASK" "$GHAND_ENVS" "$GHAND_EPISODES" "$GHAND_SECONDS" "$GHAND_SEED" <<'VERIFY'
import hashlib,json,math,sys
from pathlib import Path
root=Path(sys.argv[1]); data=json.loads((root/'selection.json').read_text())
assert data['task']==sys.argv[2], 'Task mismatch'
assert int(sys.argv[3])>0 and int(sys.argv[4])>0 and int(sys.argv[6])>=0, 'Invalid environment/episode count or seed'
assert math.isfinite(float(sys.argv[5])) and float(sys.argv[5])>0, 'Invalid duration'
for name,expected in data['bundle_hashes'].items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
  raise SystemExit('Frozen file changed: '+name)
print('Verified frozen model: '+root.name)
VERIFY
if [[ $GHAND_DRY -eq 1 ]]; then
 GHAND_RUN="$GHAND_DELIVERY/runs/NEW_UNIQUE_DIRECTORY"
else
 mkdir -p "$GHAND_DELIVERY/runs"
 GHAND_RUN="$(mktemp -d "$GHAND_DELIVERY/runs/reorientation_XXXXXXXX")"
fi
GHAND_COMMAND=(bash "$GHAND_DELIVERY/python.sh"
 scripts/cube_reorientation/evaluate.py
 --task "$GHAND_TASK" --checkpoint "$GHAND_BUNDLE/nn/policy.pth"
 --num_envs "$GHAND_ENVS" --episodes_per_env "$GHAND_EPISODES" --episode_seconds "$GHAND_SECONDS" --seed "$GHAND_SEED"
 --relative_goal_probability 0 --tolerance 0.1 --output "$GHAND_RUN/evaluation.json" "${GHAND_EXTRA[@]}")
if [[ $GHAND_DRY -eq 1 ]]; then printf '%q ' "${GHAND_COMMAND[@]}"; printf '\n'; exit 0; fi
echo "Results: $GHAND_RUN"
cd "$GHAND_DELIVERY"
exec "${GHAND_COMMAND[@]}"
