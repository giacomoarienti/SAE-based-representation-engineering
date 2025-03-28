#!/bin/bash -l

set -e
set -u

if [ $# -lt 1 ]; then
  echo "Error: Model path not provided."
  echo "Usage: $0 <model_path>"
  exit 1
fi

MODEL_PATH="${1}"
SAVE_DIR_NAME="grouped_prompts"  # save to PROJ_DIR / "cache_data" / model_name / SAVE_DIR_NAME

python3 -m spare.group_prompts \
  --model_path=${MODEL_PATH} \
  --save_dir_name=${SAVE_DIR_NAME} \
  --k_shot=3 \
  --seeds_to_encode 42 43 44 45 46

python3 -m spare.group_prompts \
  --model_path=${MODEL_PATH} \
  --save_dir_name=${SAVE_DIR_NAME} \
  --k_shot=4 \
  --seeds_to_encode 42 43 44 45 46

python3 -m spare.group_prompts \
  --model_path=${MODEL_PATH} \
  --save_dir_name=${SAVE_DIR_NAME} \
  --k_shot=5 \
  --seeds_to_encode 42 43 44 45 46
