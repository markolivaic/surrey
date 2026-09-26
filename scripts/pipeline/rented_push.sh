#!/usr/bin/env bash
# Copy the code to a rented box, and nothing else: no data, no caches, no checkpoints, no virtualenv.
#
# The instance rebuilds its own data from the public sources, which is the point: it is the first time
# the download and preparation scripts run on a machine that has never seen this project. Only if that
# turns out to cost more than two hours of billed time do we send the prepared cache instead, which is
# what --with-cache does.
#
#   bash scripts/pipeline/rented_push.sh HOST PORT
#   bash scripts/pipeline/rented_push.sh HOST PORT --with-cache
#   bash scripts/pipeline/rented_push.sh HOST PORT --phase4
#
# --phase4 also sends what Phase 4 predicts with and cannot rebuild on the box: the San Francisco tiles
# as the laptop prepared them, which the laptop's references are indexed by, and the old checkpoints
# that are predicted again without training (docs/phase4_criteria.md). Each file is hashed here first,
# and scripts/pipeline/phase4_chain.sh checks the hashes on the box before it starts.
set -euo pipefail
HOST=${1:?host, the one vast.ai shows in the ssh line}
PORT=${2:?port, the one vast.ai shows in the ssh line}
OPTION=${3:-}
# The key vast.ai holds for this project; SURREY_SSH_KEY points elsewhere if it is named otherwise.
KEY="${SURREY_SSH_KEY:-${HOME}/.ssh/vast_surrey}"
REMOTE=/workspace/surrey
cd "$(dirname "$0")/../.."

ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new "root@$HOST" "mkdir -p $REMOTE"

echo "sending the code"
tar --exclude=./data --exclude=./.venv --exclude=./.git --exclude=./node_modules \
    --exclude=./web/node_modules --exclude=./web/dist --exclude=./docs/design/page \
    --exclude=./docs/design/prototype --exclude=./__pycache__ --exclude='*.pyc' \
    -czf - . | ssh -i "$KEY" -p "$PORT" "root@$HOST" "tar -xzf - -C $REMOTE"

# The committed artefacts travel with the code: training reads the class fractions and the split from
# them, and the box must not have to rebuild what the repository already records.
echo "sending the committed artefacts"
tar -czf - data/artifacts | ssh -i "$KEY" -p "$PORT" "root@$HOST" "tar -xzf - -C $REMOTE"

if [ "$OPTION" = "--with-cache" ]; then
  echo "sending the prepared cache as well, about 7 GB"
  tar -czf - data/prep/dales data/prep/dales_water_ignore data/prep/swiss \
    | ssh -i "$KEY" -p "$PORT" "root@$HOST" "tar -xzf - -C $REMOTE"
fi

if [ "$OPTION" = "--phase4" ]; then
  echo "hashing and sending the San Francisco tiles and the old checkpoints"
  LIST=$(mktemp)
  find data/prep/target -type f | sort > "$LIST"
  ls data/checkpoints/dayton.pt data/checkpoints/m0prime_seed?.pt data/checkpoints/m1_seed?.pt >> "$LIST"
  xargs -a "$LIST" sha256sum > data/phase4_inputs.sha256
  echo data/phase4_inputs.sha256 >> "$LIST"
  tar -czf - -T "$LIST" | ssh -i "$KEY" -p "$PORT" "root@$HOST" "tar -xzf - -C $REMOTE"
  rm -f "$LIST"
fi

# Files written on Windows carry CRLF, and a shell script whose lines end in a carriage return dies on
# the first "set -euo pipefail" with a message about an invalid option name.
ssh -i "$KEY" -p "$PORT" "root@$HOST" \
  "cd $REMOTE && find . -name '*.sh' -print0 | xargs -0 sed -i 's/\r//'"

echo "code is on the box at $REMOTE"
echo "next: ssh -i $KEY -p $PORT root@$HOST 'cd $REMOTE && bash scripts/pipeline/rented_setup.sh'"
