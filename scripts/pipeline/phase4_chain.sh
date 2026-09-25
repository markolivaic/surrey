#!/usr/bin/env bash
# The Phase 4 chain on the rented card, as docs/phase4_criteria.md lays it out. It runs after
# rented_setup.sh has built the DALES-2 data on the box and rented_push.sh --phase4 has sent the
# San Francisco tiles and the old checkpoints with their hashes:
#
#   bash scripts/pipeline/phase4_chain.sh
#
# Every step goes into data/artifacts/logs/gpu_chain_phase4.log with its UTC time, and the first
# failure stops the chain there. Nothing here scores; the laptop scores after rented_pull.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
LOG=data/artifacts/logs/gpu_chain_phase4.log
mkdir -p data/artifacts/logs data/logs
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
trap 'say "stopped: exit $? at line $LINENO"' ERR

say "phase 4 chain starts on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
sha256sum --quiet -c data/phase4_inputs.sha256
say "the San Francisco tiles and the old checkpoints match the hashes taken on the laptop"
python -m pytest tests/integrity/test_prepared_flags.py -q -p no:cacheprovider
say "no prepared tile holds a point with a dropped flag"

# 200 steps of a configuration, timed and written down before its long run starts.
time_200() {
  local name=$1
  shift
  local t0
  t0=$(date +%s)
  python -u scripts/pipeline/train.py "$@" --steps 200 --eval-every 1000000 --final-eval-blocks 1 \
    --ckpt "/tmp/probe_$name.pt" --out "/tmp/probe_$name.json" > "data/logs/probe_$name.log" 2>&1
  say "200 steps of $name took $(($(date +%s) - t0)) s"
}

for seed in 0 1 2; do
  letter=$(echo ABC | cut -c$((seed + 1)))
  time_200 "m0_seed$letter" --recipe m0 --seed "$seed"
  say "m0_seed$letter training starts"
  python -u scripts/pipeline/train.py --recipe m0 --seed "$seed" --resume > "data/logs/train_m0_seed$letter.log" 2>&1
  say "m0_seed$letter trained, checkpoint $(sha256sum "data/checkpoints/m0_seed$letter.pt" | cut -c1-16)"
done

# The capacity control, in the configuration the earlier large run used (docs/rented_gpu.md).
BIG=(--recipe m0 --seed 0 --width 512 --batch 8 --lr 0.004)
time_200 m0_big_seedA "${BIG[@]}"
say "m0_big_seedA training starts"
python -u scripts/pipeline/train.py "${BIG[@]}" --steps 6000 --resume \
  --ckpt data/checkpoints/m0_big_seedA.pt --out data/artifacts/train_m0_big_seedA.json \
  > data/logs/train_m0_big_seedA.log 2>&1
say "m0_big_seedA trained, checkpoint $(sha256sum data/checkpoints/m0_big_seedA.pt | cut -c1-16)"

# The number of voting passes, by the rule, on held-out Surrey and before any San Francisco figure.
python -u scripts/pipeline/choose_votes.py --ckpt data/checkpoints/m0_seedA.pt > data/logs/choose_votes.log 2>&1
VOTES=$(python -c "import json; print(json.load(open('data/artifacts/votes.json'))['votes'])")
say "votes chosen by the rule: $VOTES"

predict() {
  local ckpt=$1 out=$2
  shift 2
  local t0
  t0=$(date +%s)
  python -u scripts/pipeline/predict_cache.py --ckpt "$ckpt" --out-dir "$out" --votes "$VOTES" "$@" \
    > "data/logs/predict_$(basename "$out").log" 2>&1
  say "predicted $ckpt into $out with $VOTES votes in $(($(date +%s) - t0)) s"
}
for v in m0_seedA m0_seedB m0_seedC m0_big_seedA m0prime_seedA m0prime_seedB m0prime_seedC \
  m1_seedA m1_seedB m1_seedC; do
  predict "data/checkpoints/$v.pt" "data/predictions/$v"
done
predict data/checkpoints/dayton.pt data/predictions
predict data/checkpoints/dayton.pt data/predictions/ret0 --ret-value 0

find data/checkpoints data/predictions -type f \( -name '*.pt' -o -name '*.npz' -o -name 'manifest.json' \) \
  -print0 | sort -z | xargs -0 sha256sum > data/phase4_outputs.sha256
say "outputs hashed: $(wc -l < data/phase4_outputs.sha256) files in data/phase4_outputs.sha256"
say "phase 4 chain done"
