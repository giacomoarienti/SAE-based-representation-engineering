#!/bin/bash -l

set -e
set -u

if [ $# -lt 2 ]; then
  echo "Error: Insufficient arguments provided."
  echo "Usage: $0 <model_path> <dataset> <rog_method>"
  exit 1
fi

MODEL_PATH="${1}"
MODEL_NAME=$(basename "${MODEL_PATH}")
DATASET="${2}"
ROG_METHOD="${3}"
SAVE_DIR_NAME="grouped_prompts"  # save to PROJ_DIR / "cache_data" / model_name / SAVE_DIR_NAME

python3 -m spare.group_prompts \
  --model_path=${MODEL_PATH} \
  --save_dir_name=${SAVE_DIR_NAME} \
  --dataset_name=${DATASET} \
  --rog_method=${ROG_METHOD} \
  --k_shot=3 \
  --seeds_to_encode 42 43 44 45 46

python3 -m spare.group_prompts \
  --model_path=${MODEL_PATH} \
  --save_dir_name=${SAVE_DIR_NAME} \
  --dataset_name=${DATASET} \
  --rog_method=${ROG_METHOD} \
  --k_shot=4 \
  --seeds_to_encode 42 43 44 45 46

python3 -m spare.group_prompts \
  --model_path=${MODEL_PATH} \
  --save_dir_name=${SAVE_DIR_NAME} \
  --dataset_name=${DATASET} \
  --rog_method=${ROG_METHOD} \
  --k_shot=5 \
  --seeds_to_encode 42 43 44 45 46
