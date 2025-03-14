#!/bin/bash -l

set -e
set -u

export CUDA_VISIBLE_DEVICES=1

LOAD_HIDDENS_NAME="grouped_activations_3shot_seeds42-43"
MI_SAVE_NAME="multiprocess-mutual_information-grouped_activations_3shot_seeds42-43"

MODEL_PATH="google/gemma-2-2b"

python -m spare.mutual_information_and_expectation \
  --num_proc=64 \
  --data_name="nqswap" \
  --model_path=${MODEL_PATH} \
  --load_hiddens_name=${LOAD_HIDDENS_NAME} \
  --layer_idx=12 \
  --minmax_normalisation \
  --mutual_information_save_name=${MI_SAVE_NAME}

python -m spare.mutual_information_and_expectation \
  --num_proc=64 \
  --data_name="nqswap" \
  --model_path=${MODEL_PATH} \
  --load_hiddens_name=${LOAD_HIDDENS_NAME} \
  --layer_idx=13 \
  --minmax_normalisation \
  --mutual_information_save_name=${MI_SAVE_NAME}

python -m spare.mutual_information_and_expectation \
  --num_proc=64 \
  --data_name="nqswap" \
  --model_path=${MODEL_PATH} \
  --load_hiddens_name=${LOAD_HIDDENS_NAME} \
  --layer_idx=14 \
  --minmax_normalisation \
  --mutual_information_save_name=${MI_SAVE_NAME}


python -m spare.mutual_information_and_expectation \
  --num_proc=64 \
  --data_name="nqswap" \
  --model_path=${MODEL_PATH} \
  --load_hiddens_name=${LOAD_HIDDENS_NAME} \
  --layer_idx=15 \
  --minmax_normalisation \
  --mutual_information_save_name=${MI_SAVE_NAME}
