#!/bin/bash -l

set -e
set -u

if [ $# -lt 1 ]; then
  echo "Error: Model path not provided."
  echo "Usage: $0 <model_path>"
  exit 1
fi

MODEL_PATH="${1}"

python3 ./scripts/memorised_set.py \
  --model_path=${MODEL_PATH} \
  --dataset="nqswap" \
  --k_shot=32
