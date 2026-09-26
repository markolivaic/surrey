#!/usr/bin/env bash
# Bring the Phase 4 outputs back from the rented box and check every file against the hash taken there.
# The laptop's own M0 checkpoints are copied aside first, since the new ones carry the same names.
#
#   bash scripts/pipeline/rented_pull.sh HOST PORT
set -euo pipefail
HOST=${1:?host, the one vast.ai shows in the ssh line}
PORT=${2:?port, the one vast.ai shows in the ssh line}
# The key vast.ai holds for this project; SURREY_SSH_KEY points elsewhere if it is named otherwise.
KEY="${SURREY_SSH_KEY:-${HOME}/.ssh/vast_surrey}"
REMOTE=/workspace/surrey
cd "$(dirname "$0")/../.."

mkdir -p data/checkpoints/before_phase4
for f in data/checkpoints/m0_seedA.pt data/checkpoints/m0_seedB.pt data/checkpoints/m0_seedC.pt \
  data/checkpoints/m0_big_seedA.pt; do
  if [ -f "$f" ] && [ ! -f "data/checkpoints/before_phase4/$(basename "$f")" ]; then
    cp "$f" data/checkpoints/before_phase4/
  fi
done

ssh -i "$KEY" -p "$PORT" "root@$HOST" "cd $REMOTE && tar -czf - data/phase4_outputs.sha256 \
  data/checkpoints/*.pt data/predictions data/artifacts/votes.json data/artifacts/train_m0_seed?.json \
  data/artifacts/train_m0_big_seedA.json data/artifacts/rented_setup.json \
  --transform 's,^data/logs,data/logs/phase4,' data/logs" \
  | tar -xzf -

sha256sum --quiet -c data/phase4_outputs.sha256
echo "every checkpoint and prediction matches the hash taken on the card"
