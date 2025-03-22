#!/bin/bash -l

set -e
set -u

ulimit -n 10240

if [ $# -lt 1 ]; then
  echo "Error: Model path not provided."
  echo "Usage: $0 <model_path>"
  exit 1
fi

MODEL_PATH="${1}"

python3 -m spare.save_grouped_activations \
  --data_name="nqswap" \
  --model_path=${MODEL_PATH} \
  --load_data_name="grouped_prompts"\
  --shots_to_encode 3 4 5 \
  --seeds_to_encode 42 43 44 45 46 \
  --save_hiddens_name="grouped_activations"

wait