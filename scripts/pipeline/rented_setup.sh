#!/usr/bin/env bash
# Set up a rented GPU box to train this project, and say what it cost in time.
#
# Runs on the instance, once, after the code has been copied over. It does not create accounts, does
# not read tokens and does not download anything private: DALES-2 comes from the public Hugging Face
# mirror and the Swiss tiles from swisstopo's public STAC service, both anonymously.
#
# The image already carries a CUDA build of torch. requirements.txt deliberately pins a CPU wheel so
# that CI can run without a GPU, so installing it here would replace a working CUDA torch with a CPU
# one and every run would be silently slow. Everything except torch is installed from the pins, and
# torch is left alone unless --pin-torch is given, which installs the same version this project has
# been using locally so the two sides differ by hardware only.
#
#   bash scripts/pipeline/rented_setup.sh            packages, then the data, timed
#   bash scripts/pipeline/rented_setup.sh --pin-torch   also install torch 2.11.0+cu128 over the image's
#   bash scripts/pipeline/rented_setup.sh --packages-only
#   bash scripts/pipeline/rented_setup.sh --m0-only     DALES-2 alone, for a round that trains M0 only
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
LOGS="$ROOT/data/logs"
mkdir -p "$LOGS" "$ROOT/data/artifacts"
STAMP="$LOGS/rented_setup.log"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STAMP"; }

# The Vast image keeps its Python in a virtual environment that a non-login shell does not pick up,
# so "python" is not on the path until it is activated.
if [ -f /venv/main/bin/activate ]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
fi

PIN_TORCH=0
PACKAGES_ONLY=0
M0_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --pin-torch) PIN_TORCH=1 ;;
    --packages-only) PACKAGES_ONLY=1 ;;
    --m0-only) M0_ONLY=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

say "host $(hostname), $(nproc) cpu cores, $(free -g | awk '/^Mem:/{print $2}') GB ram"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$STAMP"

say "installing everything except torch, from the pins in requirements.txt"
grep -viE '^(torch|#|--extra-index-url|$)' requirements.txt > /tmp/requirements_no_torch.txt
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r /tmp/requirements_no_torch.txt
if [ "$PIN_TORCH" = "1" ]; then
  say "installing torch 2.11.0+cu128, the version the laptop runs, so only the hardware differs"
  python -m pip install --quiet --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128
fi
python - <<'PY' | tee -a "$STAMP"
import torch
print(f"torch {torch.__version__}, cuda available {torch.cuda.is_available()}, "
      f"device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
PY

if [ "$PACKAGES_ONLY" = "1" ]; then
  say "packages only, as asked"
  exit 0
fi

# The data pipeline, on a machine that has never seen this project: this is also the clean-machine test
# of the download and preparation scripts, which have only ever run on the laptop.
prep_start=$(date +%s)
say "fetching and preparing DALES-2 from the Hugging Face mirror"
python -u scripts/pipeline/fetch_dales2.py 2>&1 | tail -3 | tee -a "$STAMP"
python -u scripts/pipeline/prep_tiles.py --source dales --force 2>&1 | tail -3 | tee -a "$STAMP"
# M0prime and M1 leave DALES water out of the loss; the mask comes from the BC Freshwater Atlas.
if [ "$M0_ONLY" = "0" ]; then
  python -u scripts/pipeline/dales_water_check.py 2>&1 | tail -3 | tee -a "$STAMP"
fi
dales_done=$(date +%s)
say "DALES ready in $(( (dales_done - prep_start) / 60 )) min"

if [ "$M0_ONLY" = "0" ]; then
  say "fetching and preparing the Swiss sub-tiles"
  python -u scripts/pipeline/prep_swiss.py 2>&1 | tail -3 | tee -a "$STAMP"
fi
prep_done=$(date +%s)
say "preparation done in $(( (prep_done - prep_start) / 60 )) min"

python - "$prep_start" "$dales_done" "$prep_done" <<'PY' | tee -a "$STAMP"
import json
import subprocess
import sys
from pathlib import Path

start, dales, done = (int(x) for x in sys.argv[1:4])
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
out = Path("data/artifacts/rented_setup.json")
out.write_text(json.dumps({
    "where": "a rented GPU box, not the laptop",
    "gpu": gpu,
    "minutes": {"dales_fetch_and_prep": round((dales - start) / 60, 1),
                "swiss_fetch_and_prep": round((done - dales) / 60, 1),
                "whole_preparation": round((done - start) / 60, 1)},
    "why_this_is_recorded": ("the first time the download and preparation scripts ran anywhere but the "
                             "laptop, which is also a test of them on a clean machine, and the rule "
                             "says that past two hours of billed time the cache is copied instead"),
}, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out}")
PY
say "setup done"
