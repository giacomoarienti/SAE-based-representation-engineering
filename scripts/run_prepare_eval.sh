#!/bin/bash -l

set -e
set -u

if [ $# -lt 3 ]; then
  echo "Error: Model path not provided."
  echo "Usage: $0 <model_path> <dataset> <rog_method>"
  exit 1
fi

MODEL_PATH="${1}"
MODEL_NAME=$(basename "${MODEL_PATH}")
DATASET="${2}"
ROG_METHOD="${3}"

K_SHOT=32

python3 -m spare.prepare_eval \
  --exp_name="${DATASET}-${MODEL_NAME}-${K_SHOT}shot-examples-closebook" \
  --model_path=${MODEL_PATH} \
  --k_shot=${K_SHOT} \
  --seed=42 \
  --batch_size=1 \
  --demonstrations_org_context \
  --demonstrations_org_answer \
  --run_close_book \
  --write_logs \
  --flash_attn \
  --rog_method=${ROG_METHOD}

K_SHOT=3

python3 -m spare.prepare_eval \
  --exp_name="${DATASET}-${MODEL_NAME}-${K_SHOT}shot-examples-openbook" \
  --model_path=${MODEL_PATH} \
  --k_shot=${K_SHOT} \
  --seed=42 \
  --batch_size=1 \
  --demonstrations_org_context \
  --demonstrations_org_answer \
  --run_open_book \
  --write_logs \
  --flash_attn \
  --rog_method=${ROG_METHOD}

# python3 -m spare.prepare_eval \
#   --exp_name="${DATASET}-${MODEL_NAME}-${K_SHOT}shot-examples-openbook-noconflict" \
#   --model_path=${MODEL_PATH} \
#   --k_shot=${K_SHOT} \
#   --seed=42 \
#   --batch_size=1 \
#   --demonstrations_org_context \
#   --demonstrations_org_answer \
#   --test_example_org_context \
#   --run_open_book \
#   --write_logs \
#   --flash_attn
