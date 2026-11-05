#!/usr/bin/env bash
# Copy the code to a rented box, and nothing else: no data, no caches, no checkpoints, no virtualenv.
#
# The instance rebuilds its own data from the public sources, which also tests the download and
# preparation scripts on a clean machine. --with-cache sends the prepared cache instead, for when the
# rebuild would cost more than the transfer.
#
#   bash scripts/pipeline/rented_push.sh HOST PORT
#   bash scripts/pipeline/rented_push.sh HOST PORT --with-cache
#   bash scripts/pipeline/rented_push.sh HOST PORT --tiles
#
# --tiles also sends the San Francisco tiles as this machine prepared them, which the references are
# indexed by. Each file is hashed here first into data/rented_inputs.sha256, and
# scripts/pipeline/rented_chain.sh checks the hashes on the box before it starts.
set -euo pipefail
HOST=${1:?host, the one vast.ai shows in the ssh line}
PORT=${2:?port, the one vast.ai shows in the ssh line}
OPTION=${3:-}
# The key vast.ai holds for this machine, named by SURREY_SSH_KEY so no default path is assumed.
KEY="${SURREY_SSH_KEY:?the SSH key vast.ai holds for this machine}"
REMOTE=/workspace/surrey
cd "$(dirname "$0")/../.."

ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new "root@$HOST" "mkdir -p $REMOTE"

echo "sending the code"
tar --exclude=./data --exclude=./.venv --exclude=./.git --exclude=./node_modules \
    --exclude=./web/node_modules --exclude=./web/dist --exclude=./docs/design/page \
    --exclude=./__pycache__ --exclude='*.pyc' \
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

# A large set goes in 100 MB parts, one after another and each retried: a single tar stream through the
# vast.ai SSH proxy can break partway, and parallel copies stall.
send_parts() {
  local list=$1 tmp part try
  tmp=$(mktemp -d)
  tar -czf - -T "$list" | split -b 100M - "$tmp/part_"
  ssh -i "$KEY" -p "$PORT" "root@$HOST" "rm -rf $REMOTE/.incoming && mkdir -p $REMOTE/.incoming"
  for part in "$tmp"/part_*; do
    for try in 1 2 3 4 5; do
      if scp -q -i "$KEY" -P "$PORT" -o ServerAliveInterval=15 "$part" "root@$HOST:$REMOTE/.incoming/"; then
        echo "  sent $(basename "$part")"
        break
      fi
      echo "  $(basename "$part") failed, try $try"
      sleep 5
    done
  done
  ssh -i "$KEY" -p "$PORT" "root@$HOST" "cd $REMOTE && cat .incoming/part_* | tar -xzf - && rm -rf .incoming"
  rm -rf "$tmp"
}

if [ "$OPTION" = "--tiles" ]; then
  echo "hashing and sending the San Francisco tiles"
  LIST=$(mktemp)
  find data/prep/target -type f | sort > "$LIST"
  xargs -a "$LIST" sha256sum > data/rented_inputs.sha256
  echo data/rented_inputs.sha256 >> "$LIST"
  send_parts "$LIST"
  rm -f "$LIST"
fi

# Files written on Windows carry CRLF, and a shell script whose lines end in a carriage return dies on
# the first "set -euo pipefail" with a message about an invalid option name.
ssh -i "$KEY" -p "$PORT" "root@$HOST" \
  "cd $REMOTE && find . -name '*.sh' -print0 | xargs -0 sed -i 's/\r//'"

echo "code is on the box at $REMOTE"
echo "next: ssh -i $KEY -p $PORT root@$HOST 'cd $REMOTE && bash scripts/pipeline/rented_setup.sh'"
