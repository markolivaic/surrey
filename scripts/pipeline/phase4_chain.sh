#!/usr/bin/env bash
# The Phase 4 chain on the rented card, as docs/phase4_criteria.md lays it out. It runs after
# rented_setup.sh --m0-only has built the DALES-2 data on the box and rented_push.sh --phase4 has sent
# the San Francisco tiles and the old checkpoints with their hashes:
#
#   bash scripts/pipeline/phase4_chain.sh
#
# Before each step the chain checks that every file the step reads is on the box, and before each long
# step it writes an estimate of how long it will take. Every line goes into
# data/logs/gpu_chain_phase4.log with its UTC time, and the first failure stops the chain
# there, inside a function as well as outside one. Run again after a stop, it skips every step whose
# output is already complete and says so. Nothing here scores; the laptop scores after rented_pull.sh.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
LOG=data/logs/gpu_chain_phase4.log
mkdir -p data/logs
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
trap 'say "stopped: exit $? at line $LINENO"' ERR

# need STEP PATH...: every path must exist before STEP runs.
need() {
  local step=$1
  shift
  for f in "$@"; do
    if [ ! -e "$f" ]; then
      say "missing before $step: $f"
      exit 1
    fi
  done
  say "inputs for $step are on the box"
}
# count STEP N GLOB: exactly N directories must match GLOB.
count() {
  local step=$1 want=$2 glob=$3 got
  got=$(compgen -G "$glob" | wc -l)
  if [ "$got" -ne "$want" ]; then
    say "before $step: $glob matches $got, expected $want"
    exit 1
  fi
}

say "phase 4 chain starts on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
need hashes data/phase4_inputs.sha256
sha256sum --quiet -c data/phase4_inputs.sha256
say "the San Francisco tiles and the old checkpoints match the hashes taken on the laptop"
need flags data/prep/dales data/prep/target
count flags 40 "data/prep/dales/*/meta.json"
count flags 8 "data/prep/target/*/meta.json"
python -m pytest tests/integrity/test_prepared_flags.py -q -p no:cacheprovider
say "no prepared tile holds a point with a dropped flag"

TRAIN_IN=(data/prep/dales data/artifacts/prep_dales.json data/artifacts/dales_stats.json)
python - <<'PY'
import json, sys
split = json.load(open("data/artifacts/dales_stats.json"))["per_split"]["train"]
tiles = json.load(open("data/artifacts/prep_dales.json"))["tiles"]
ok = split["tiles"] == 29 and sum(t["split"] == "train" for t in tiles) == 29
sys.exit(0 if ok else "the class fractions or the split do not name the 29 training tiles")
PY
say "class weights and the split both name the 29 training tiles"

# 200 steps of a configuration, timed, and the full run's time estimated from them, before it starts.
time_200() {
  local name=$1 steps=$2
  shift 2
  local t0 took
  t0=$(date +%s)
  python -u scripts/pipeline/train.py "$@" --steps 200 --eval-every 1000000 --final-eval-blocks 1 \
    --ckpt "/tmp/probe_$name.pt" --out "/tmp/probe_$name.json" > "data/logs/probe_$name.log" 2>&1
  took=$(($(date +%s) - t0))
  say "200 steps of $name took $took s; the $steps step run is estimated at $((took * steps / 200 / 60)) min"
}

for seed in 0 1 2; do
  letter=$(echo ABC | cut -c$((seed + 1)))
  if [ -f "data/checkpoints/m0_seed$letter.pt" ] && [ -f "data/artifacts/train_m0_seed$letter.json" ]; then
    say "m0_seed$letter already trained, checkpoint $(sha256sum "data/checkpoints/m0_seed$letter.pt" | cut -c1-16); skipped"
    continue
  fi
  need "m0_seed$letter" "${TRAIN_IN[@]}"
  time_200 "m0_seed$letter" 12000 --recipe m0 --seed "$seed"
  say "m0_seed$letter training starts"
  python -u scripts/pipeline/train.py --recipe m0 --seed "$seed" --resume > "data/logs/train_m0_seed$letter.log" 2>&1
  say "m0_seed$letter trained, checkpoint $(sha256sum "data/checkpoints/m0_seed$letter.pt" | cut -c1-16)"
done

# The capacity control, in the configuration the earlier large run used (docs/rented_gpu.md).
BIG=(--recipe m0 --seed 0 --width 512 --batch 8 --lr 0.004)
if [ -f data/checkpoints/m0_big_seedA.pt ] && [ -f data/artifacts/train_m0_big_seedA.json ]; then
  say "m0_big_seedA already trained, checkpoint $(sha256sum data/checkpoints/m0_big_seedA.pt | cut -c1-16); skipped"
else
  need m0_big_seedA "${TRAIN_IN[@]}"
  time_200 m0_big_seedA 6000 "${BIG[@]}"
  say "m0_big_seedA training starts"
  python -u scripts/pipeline/train.py "${BIG[@]}" --steps 6000 --resume \
    --ckpt data/checkpoints/m0_big_seedA.pt --out data/artifacts/train_m0_big_seedA.json \
    > data/logs/train_m0_big_seedA.log 2>&1
  say "m0_big_seedA trained, checkpoint $(sha256sum data/checkpoints/m0_big_seedA.pt | cut -c1-16)"
fi

# The number of voting passes, by the rule, on held-out Surrey and before any San Francisco figure.
if [ -f data/artifacts/votes.json ]; then
  say "the number of votes was already chosen; data/artifacts/votes.json is kept"
else
  need votes data/checkpoints/m0_seedA.pt data/prep/dales data/artifacts/prep_dales.json
  say "choosing the number of votes: 16 passes over the 11 held-out tiles, estimated from the earlier" \
    "rented pass (86.8 s for all 19 tiles, 0.711 of the points held-out Surrey) at about 17 min"
  python -u scripts/pipeline/choose_votes.py --ckpt data/checkpoints/m0_seedA.pt > data/logs/choose_votes.log 2>&1
fi
VOTES=$(python -c "import json; print(json.load(open('data/artifacts/votes.json'))['votes'])")
PASS=$(python -c "import json; d = json.load(open('data/artifacts/votes.json')); print(round(d['seconds_all_passes'] / 16 / 0.711))")
say "votes chosen by the rule: $VOTES; one pass of all 19 tiles is about $PASS s, so the eleven" \
  "small-model predictions are estimated at $((VOTES * PASS * 11 / 60)) min and the large one at" \
  "about $((VOTES * PASS * 155 / 100 / 60)) min"

MODELS=(m0_seedA m0_seedB m0_seedC m0_big_seedA m0prime_seedA m0prime_seedB m0prime_seedC
  m1_seedA m1_seedB m1_seedC)
PREDICT_IN=(data/artifacts/votes.json data/prep/target data/prep/dales data/artifacts/prep_dales.json
  data/checkpoints/dayton.pt)
for v in "${MODELS[@]}"; do PREDICT_IN+=("data/checkpoints/$v.pt"); done
need predictions "${PREDICT_IN[@]}"

predict() {
  local ckpt=$1 out=$2
  shift 2
  local t0
  if [ -f "$out/manifest.json" ] && python - "$out/manifest.json" "$ckpt" "$VOTES" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
done = m.get("votes") == int(sys.argv[3]) and m.get("checkpoint") == sys.argv[2] and len(m.get("tiles", {})) == 19
sys.exit(0 if done else 1)
PY
  then
    say "$out already predicted from $ckpt with $VOTES votes; skipped"
    return
  fi
  t0=$(date +%s)
  python -u scripts/pipeline/predict_cache.py --ckpt "$ckpt" --out-dir "$out" --votes "$VOTES" "$@" \
    > "data/logs/predict_$(basename "$out").log" 2>&1
  say "predicted $ckpt into $out with $VOTES votes in $(($(date +%s) - t0)) s"
}
for v in "${MODELS[@]}"; do
  predict "data/checkpoints/$v.pt" "data/predictions/$v"
done
predict data/checkpoints/dayton.pt data/predictions
predict data/checkpoints/dayton.pt data/predictions/ret0 --ret-value 0

find data/checkpoints data/predictions -type f \( -name '*.pt' -o -name '*.npz' -o -name 'manifest.json' \) \
  -print0 | sort -z | xargs -0 sha256sum > data/phase4_outputs.sha256
say "outputs hashed: $(wc -l < data/phase4_outputs.sha256) files in data/phase4_outputs.sha256"
say "phase 4 chain done"
