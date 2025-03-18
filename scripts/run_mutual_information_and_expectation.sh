#!/bin/bash -l

set -e
set -u

export CUDA_VISIBLE_DEVICES=0

LOAD_HIDDENS_NAME="grouped_activations_3shot_seeds42-43"
MI_SAVE_NAME="multiprocess-mutual_information-grouped_activations_3shot_seeds42-43"

# Loop through each layer
for layer in $LAYERS; do
  python -m spare.mutual_information_and_expectation \
    --num_proc=64 \
    --data_name="nqswap" \
    --model_path=${MODEL_PATH} \
    --load_hiddens_name=${LOAD_HIDDENS_NAME} \
    --layer_idx=${layer} \
    --minmax_normalisation \
    --mutual_information_save_name=${MI_SAVE_NAME}
done