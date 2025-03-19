#!/bin/bash -l

set -e
set -u


K_SHOT=32

python -m spare.prepare_eval \
  --exp_name="nqswap-${MODEL_NAME}-${K_SHOT}shot-examples-closebook" \
  --model_path=${MODEL_PATH} \
  --k_shot=${K_SHOT} \
  --seed=42 \
  --batch_size=1 \
  --demonstrations_org_context \
  --demonstrations_org_answer \
  --run_close_book \
  --write_logs

K_SHOT=3

python -m spare.prepare_eval \
  --exp_name="nqswap-${MODEL_NAME}-${K_SHOT}shot-examples-openbook" \
  --model_path=${MODEL_PATH} \
  --k_shot=${K_SHOT} \
  --seed=42 \
  --batch_size=1 \
  --demonstrations_org_context \
  --demonstrations_org_answer \
  --run_open_book \
  --write_logs

python -m spare.prepare_eval \
  --exp_name="nqswap-${MODEL_NAME}-${K_SHOT}shot-examples-openbook-noconflict" \
  --model_path=${MODEL_PATH} \
  --k_shot=${K_SHOT} \
  --seed=42 \
  --batch_size=1 \
  --demonstrations_org_context \
  --demonstrations_org_answer \
  --test_example_org_context \
  --run_open_book \
  --write_logs
