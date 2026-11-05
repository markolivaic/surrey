#!/usr/bin/env bash
# Bring the outputs of scripts/pipeline/rented_chain.sh back from the rented card, check every file
# against the hash taken there, and put them in place: the four M0 checkpoints, their predictions, the
# training and vote artefacts, and the chain's log.
#
# From the private Hugging Face dataset the chain uploaded to, with the login on this machine:
#
#   bash scripts/pipeline/rented_pull.sh <owner>/<dataset>
#
# Or, where no login could be put on the card, over the card's direct SSH port (the address and port
# vast.ai lists for direct connections, not the ssh*.vast.ai proxy):
#
#   bash scripts/pipeline/rented_pull.sh --ssh <address> <port>
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE=data/rented_download
MODELS=(m0_seedA m0_seedB m0_seedC m0_big_seedA)

if [ "${1:-}" = "--ssh" ]; then
  HOST=${2:?the public address of the card}
  PORT=${3:?the direct SSH port of the card}
  KEY=${SURREY_SSH_KEY:?the SSH key vast.ai holds for this machine}
  rm -rf "$STAGE"
  mkdir -p "$STAGE"
  ssh -i "$KEY" -p "$PORT" "root@$HOST" "cd /workspace/surrey/data/rented_out && tar -cf - ." |
    tar -xf - -C "$STAGE"
else
  REPO=${1:?the Hugging Face dataset the card uploaded to, or --ssh ADDRESS PORT}
  HF=.venv/Scripts/hf.exe
  [ -x "$HF" ] || HF=.venv/bin/hf
  "$HF" download "$REPO" --repo-type dataset --local-dir "$STAGE"
fi
(cd "$STAGE" && sha256sum --quiet -c rented_outputs.sha256)
echo "every file matches the hash taken on the card"

for v in "${MODELS[@]}"; do
  rm -rf "data/predictions/$v"
  cp "$STAGE/data/checkpoints/$v.pt" data/checkpoints/
  cp -r "$STAGE/data/predictions/$v" data/predictions/
done
# rented_setup.json records the first preparation off the laptop and is not replaced; the card that
# trained these models is named in each training artefact and in the chain's log.
for f in "$STAGE"/data/artifacts/*.json; do
  [ "$(basename "$f")" = rented_setup.json ] || cp "$f" data/artifacts/
done
mkdir -p data/logs
cp "$STAGE"/data/logs/* data/logs/
echo "outputs in place; the hash lists are $STAGE/rented_outputs.sha256 and data/rented_inputs.sha256"
