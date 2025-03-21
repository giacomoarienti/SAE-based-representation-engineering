#!/bin/bash -l

set -e
set -u

ulimit -n 10240

export CUDA_VISIBLE_DEVICES=0


python -m spare.save_grouped_activations \
  --data_name="nqswap" \
  --model_path=${MODEL_PATH} \
  --load_data_name="grouped_prompts"\
  --shots_to_encode 3 \
  --seeds_to_encode 42 43 \
  --save_hiddens_name="grouped_activations_3shot_seeds42-43"

wait