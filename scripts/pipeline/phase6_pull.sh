#!/usr/bin/env bash
# Bring the Phase 6 outputs back from the Hugging Face dataset the card uploaded them to, check every
# hash, and put them in place. The Phase 4 checkpoints and predictions of the same four models are
# moved aside first, into data/checkpoints/phase4 and data/predictions/phase4, because Phase 6 is read
# against them (docs/phase6_criteria.md).
#
#   bash scripts/pipeline/phase6_pull.sh markolivaic/surrey-phase6
#
# The download uses the Hugging Face login on this machine, which only its owner puts there.
set -euo pipefail
REPO=${1:?the Hugging Face dataset the card uploaded to}
cd "$(dirname "$0")/../.."
HF=.venv/Scripts/hf.exe
[ -x "$HF" ] || HF=.venv/bin/hf
STAGE=data/phase6_download
MODELS=(m0_seedA m0_seedB m0_seedC m0_big_seedA)

"$HF" download "$REPO" --repo-type dataset --local-dir "$STAGE"
(cd "$STAGE" && sha256sum --quiet -c phase6_outputs.sha256)
echo "every file matches the hash taken on the card"

mkdir -p data/checkpoints/phase4 data/predictions/phase4
for v in "${MODELS[@]}"; do
  if [ -f "data/checkpoints/$v.pt" ] && [ ! -f "data/checkpoints/phase4/$v.pt" ]; then
    mv "data/checkpoints/$v.pt" data/checkpoints/phase4/
  fi
  if [ -d "data/predictions/$v" ] && [ ! -d "data/predictions/phase4/$v" ]; then
    mv "data/predictions/$v" data/predictions/phase4/
  fi
  rm -rf "data/predictions/$v"
  cp "$STAGE/data/checkpoints/$v.pt" data/checkpoints/
  cp -r "$STAGE/data/predictions/$v" data/predictions/
done
for s in A B C; do
  if [ -f "data/artifacts/train_m0_seed$s.json" ] && [ ! -f "data/artifacts/train_m0phase4_seed$s.json" ]; then
    cp "data/artifacts/train_m0_seed$s.json" "data/artifacts/train_m0phase4_seed$s.json"
  fi
done
# Phase 4's vote count stays readable beside Phase 6's: the kept predictions were made with it.
if [ -f data/artifacts/votes.json ] && [ ! -f data/artifacts/votes_phase4.json ]; then
  cp data/artifacts/votes.json data/artifacts/votes_phase4.json
fi
# rented_setup.json stays Phase 4's, the first run of the preparation off the laptop, which is what it
# says it records; the Phase 6 box is in each training artefact and in the chain's log.
for f in "$STAGE"/data/artifacts/*.json; do
  [ "$(basename "$f")" = rented_setup.json ] || cp "$f" data/artifacts/
done
cp "$STAGE/phase6_outputs.sha256" data/phase6_inputs.sha256 data/artifacts/logs/
mkdir -p data/logs
cp "$STAGE"/data/logs/* data/logs/
echo "Phase 6 outputs in place; Phase 4's are in data/checkpoints/phase4 and data/predictions/phase4"
