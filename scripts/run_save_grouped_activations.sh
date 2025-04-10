#!/bin/bash -l

set -e
set -u

ulimit -n 65536

if [ $# -lt 3 ]; then
  echo "Error: Insufficient arguments provided."
  echo "Usage: $0 <model_path> <dataset> <rog_method>"
  exit 1
fi

MODEL_PATH="${1}"
MODEL_NAME=$(basename "${MODEL_PATH}")
DATASET="${2}"
ROG_METHOD="${3}"

python3 -m spare.save_grouped_activations \
  --data_name=${DATASET} \
  --model_path=${MODEL_PATH} \
  --rog_method=${ROG_METHOD} \
  --load_data_name="grouped_prompts"\
  --shots_to_encode 3 4 5 \
  --seeds_to_encode 42 43 44 45 46 \
  --save_hiddens_name="grouped_activations"

wait