#!/bin/bash -l

set -e
set -u


K_SHOT=32

python -m spare.prepare_eval \
  --exp_name="nqswap-gemma-2-9b-${K_SHOT}shot-examples-closebook" \
  --model_path="google/gemma-2-2b" \
  --k_shot=${K_SHOT} \
  --seed=42 \
  --batch_size=1 \
  --demonstrations_org_context \
  --demonstrations_org_answer \
  --run_close_book \
  --write_logs

K_SHOT=4

python -m spare.prepare_eval \
  --exp_name="nqswap-gemma-2-9b-${K_SHOT}shot-examples-openbook" \
  --model_path="google/gemma-2-2b" \
  --k_shot=${K_SHOT} \
  --seed=42 \
  --batch_size=1 \
  --demonstrations_org_context \
  --demonstrations_org_answer \
  --run_open_book \
  --write_logs

python -m spare.prepare_eval \
  --exp_name="nqswap-gemma-2-9b-${K_SHOT}shot-examples-openbook-noconflict" \
  --model_path="google/gemma-2-2b" \
  --k_shot=${K_SHOT} \
  --seed=42 \
  --batch_size=1 \
  --demonstrations_org_context \
  --demonstrations_org_answer \
  --test_example_org_context \
  --run_open_book \
  --write_logs
