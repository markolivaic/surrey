#!/usr/bin/env bash
# Phase 6 on the rented card, in the order docs/phase6_criteria.md sets out: check the inputs, check that
# the neighbour search matches an exact search under autocast on this card, train M0 on three seeds and
# the large control, choose the number of votes by the same rule, predict the four new models, hash
# everything, and upload it to the Hugging Face dataset named by HF_REPO.
#
#   HF_REPO=markolivaic/surrey-phase6 bash scripts/pipeline/phase6_chain.sh
#
# The upload uses the Hugging Face login already on the box, which only its owner puts there. Without
# it the chain stops after the hashes and says so; the outputs stay on the box. Every line goes into
# data/logs/gpu_chain_phase6.log with its UTC time, and the first failure stops the chain there. Run
# again after a stop, it skips every step whose output is already complete and says so.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
LOG=data/logs/gpu_chain_phase6.log
mkdir -p data/logs
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
trap 'say "stopped: exit $? at line $LINENO"' ERR
HF_REPO=${HF_REPO:?the Hugging Face dataset the outputs go to, for example markolivaic/surrey-phase6}

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
count() {
  local step=$1 want=$2 glob=$3 got
  got=$(compgen -G "$glob" | wc -l)
  if [ "$got" -ne "$want" ]; then
    say "before $step: $glob matches $got, expected $want"
    exit 1
  fi
}

say "phase 6 chain starts on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
# The committed artefacts travel with the code, Phase 4's vote count among them. Phase 6 chooses its
# own, so on the first start that file goes; a restart keeps the one this chain chose.
if [ ! -f data/phase6_started ]; then
  rm -f data/artifacts/votes.json
  touch data/phase6_started
fi
need hashes data/phase6_inputs.sha256
sha256sum --quiet -c data/phase6_inputs.sha256
say "the San Francisco tiles match the hashes taken on the laptop"
need flags data/prep/dales data/prep/target
count flags 40 "data/prep/dales/*/meta.json"
count flags 8 "data/prep/target/*/meta.json"
python -m pytest tests/integrity/test_prepared_flags.py -q -p no:cacheprovider
say "no prepared tile holds a point with a dropped flag"
python -m pytest tests/unit/test_knn_precision.py -q -p no:cacheprovider
say "the neighbour search matches an exact search under fp16 autocast, on the CPU and on this card"

TRAIN_IN=(data/prep/dales data/artifacts/prep_dales.json data/artifacts/dales_stats.json)
python - <<'PY'
import json, sys
split = json.load(open("data/artifacts/dales_stats.json"))["per_split"]["train"]
tiles = json.load(open("data/artifacts/prep_dales.json"))["tiles"]
ok = split["tiles"] == 29 and sum(t["split"] == "train" for t in tiles) == 29
sys.exit(0 if ok else "the class fractions or the split do not name the 29 training tiles")
PY
say "class weights and the split both name the 29 training tiles"

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
  if [ -f "data/checkpoints/m0_seed$letter.pt" ] && [ -f "data/phase6_trained_m0_seed$letter" ]; then
    say "m0_seed$letter already trained, checkpoint $(sha256sum "data/checkpoints/m0_seed$letter.pt" | cut -c1-16); skipped"
    continue
  fi
  need "m0_seed$letter" "${TRAIN_IN[@]}"
  time_200 "m0_seed$letter" 12000 --recipe m0 --seed "$seed"
  say "m0_seed$letter training starts"
  python -u scripts/pipeline/train.py --recipe m0 --seed "$seed" --resume > "data/logs/train_m0_seed$letter.log" 2>&1
  touch "data/phase6_trained_m0_seed$letter"
  say "m0_seed$letter trained, checkpoint $(sha256sum "data/checkpoints/m0_seed$letter.pt" | cut -c1-16)"
done

# The capacity control, in the configuration Phase 4 used (docs/rented_gpu.md).
BIG=(--recipe m0 --seed 0 --width 512 --batch 8 --lr 0.004)
if [ -f data/checkpoints/m0_big_seedA.pt ] && [ -f data/phase6_trained_m0_big_seedA ]; then
  say "m0_big_seedA already trained, checkpoint $(sha256sum data/checkpoints/m0_big_seedA.pt | cut -c1-16); skipped"
else
  need m0_big_seedA "${TRAIN_IN[@]}"
  time_200 m0_big_seedA 6000 "${BIG[@]}"
  say "m0_big_seedA training starts"
  python -u scripts/pipeline/train.py "${BIG[@]}" --steps 6000 --resume \
    --ckpt data/checkpoints/m0_big_seedA.pt --out data/artifacts/train_m0_big_seedA.json \
    > data/logs/train_m0_big_seedA.log 2>&1
  touch data/phase6_trained_m0_big_seedA
  say "m0_big_seedA trained, checkpoint $(sha256sum data/checkpoints/m0_big_seedA.pt | cut -c1-16)"
fi

# The number of voting passes, by the same rule as Phase 4, on the new M0 seed A.
if [ -f data/artifacts/votes.json ]; then
  say "the number of votes was already chosen in this phase; data/artifacts/votes.json is kept"
else
  need votes data/checkpoints/m0_seedA.pt data/prep/dales data/artifacts/prep_dales.json
  say "choosing the number of votes on the new M0 seed A: 16 passes over the 11 held-out tiles"
  python -u scripts/pipeline/choose_votes.py --ckpt data/checkpoints/m0_seedA.pt > data/logs/choose_votes.log 2>&1
fi
VOTES=$(python -c "import json; print(json.load(open('data/artifacts/votes.json'))['votes'])")
say "votes chosen by the rule: $VOTES"

MODELS=(m0_seedA m0_seedB m0_seedC m0_big_seedA)
PREDICT_IN=(data/artifacts/votes.json data/prep/target data/prep/dales data/artifacts/prep_dales.json)
for v in "${MODELS[@]}"; do PREDICT_IN+=("data/checkpoints/$v.pt"); done
need predictions "${PREDICT_IN[@]}"

predict() {
  local ckpt=$1 out=$2
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
  python -u scripts/pipeline/predict_cache.py --ckpt "$ckpt" --out-dir "$out" --votes "$VOTES" \
    > "data/logs/predict_$(basename "$out").log" 2>&1
  say "predicted $ckpt into $out with $VOTES votes in $(($(date +%s) - t0)) s"
}
for v in "${MODELS[@]}"; do
  predict "data/checkpoints/$v.pt" "data/predictions/$v"
done

# Everything that comes back, in one folder with its hashes: the checkpoints, the predictions, the
# artefacts written here and this chain's log. Hard links, so nothing is copied twice on the disk.
# Built once: a restart after a broken upload keeps the folder, and with it the upload's own state.
OUT=data/phase6_out
if [ ! -f "$OUT/phase6_outputs.sha256" ]; then
  rm -rf "$OUT"
  mkdir -p "$OUT/data/checkpoints" "$OUT/data/predictions" "$OUT/data/artifacts" "$OUT/data/logs"
  for v in "${MODELS[@]}"; do
    cp -l "data/checkpoints/$v.pt" "$OUT/data/checkpoints/"
    cp -al "data/predictions/$v" "$OUT/data/predictions/"
  done
  cp -l data/artifacts/votes.json data/artifacts/train_m0_seed?.json data/artifacts/train_m0_big_seedA.json \
    data/artifacts/rented_setup.json "$OUT/data/artifacts/"
  (cd "$OUT" && find data -type f -print0 | sort -z | xargs -0 sha256sum > phase6_outputs.sha256)
fi
say "outputs hashed: $(wc -l < "$OUT/phase6_outputs.sha256") files in $OUT/phase6_outputs.sha256"
cp "$LOG" "$OUT/data/logs/"

# hf 0.35 prints "Not logged in" and still exits 0, so the words are checked as well as the status.
if ! who=$(hf auth whoami 2>&1) || [[ "$who" == *"Not logged in"* ]]; then
  say "no Hugging Face login on this box; the outputs wait in $OUT for the owner's login, then run this again"
  exit 1
fi
say "uploading $OUT to the dataset $HF_REPO"
hf upload-large-folder "$HF_REPO" "$OUT" --repo-type dataset --private
say "uploaded; phase 6 chain done"
