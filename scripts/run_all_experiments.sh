#!/bin/bash -l

set -e
set -u

if [ $# -lt 2 ]; then
  echo "Error: Required arguments not provided."
  echo "Usage: $0 <model_path> <dataset>"
  exit 1
fi

MODEL_PATH="${1}"
DATASET="${2}"

# for seed in $(seq 42 46); do
for seed in $(seq 43 46); do
  python3 ./scripts/run_spare.py \
    --model_path="${MODEL_PATH}" \
    --data_name="${DATASET}" \
    --layer_ids 8 9 \
    --edit_degree=2.0 \
    --select_topk_proportion=0.07 \
    --seed=${seed} \
    --hiddens_name="grouped_activations" \
    --mutual_information_save_name="mutual_information" \
    --run_use_parameter \
    --run_use_context
done

for seed in $(seq 42 46); do
  python3 ./scripts/run_spare.py \
    --model_path="${MODEL_PATH}" \
    --data_name="${DATASET}" \
    --layer_ids 9 10 \
    --edit_degree=2.0 \
    --select_topk_proportion=0.07 \
    --seed=${seed} \
    --hiddens_name="grouped_activations" \
    --mutual_information_save_name="mutual_information" \
    --run_use_parameter \
    --run_use_context
done

for seed in $(seq 42 46); do
  python3 ./scripts/run_spare.py \
    --model_path="${MODEL_PATH}" \
    --data_name="${DATASET}" \
    --layer_ids 10 11 \
    --edit_degree=2.0 \
    --select_topk_proportion=0.07 \
    --seed=${seed} \
    --hiddens_name="grouped_activations" \
    --mutual_information_save_name="mutual_information" \
    --run_use_parameter \
    --run_use_context
done