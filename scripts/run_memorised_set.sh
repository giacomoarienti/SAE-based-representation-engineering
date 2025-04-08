#!/bin/bash -l

set -e
set -u

if [ $# -lt 2 ]; then
  echo "Error: Model path not provided."
  echo "Usage: $0 <model_path> <dataset>"
  exit 1
fi

MODEL_PATH="${1}"
DATASET="${2}"

python3 ./scripts/memorised_set.py \
  --model_path=${MODEL_PATH} \
  --dataset=${DATASET} \
  --k_shot=32
