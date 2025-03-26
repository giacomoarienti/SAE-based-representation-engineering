#!/bin/bash -l

set -e
set -u

for seed in $(seq 42 46); do
  python3 ./scripts/run_spare.py \
    --model_path="meta-llama/Llama-3.2-1B" \
    --data_name="nqswap" \
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
    --model_path="meta-llama/Llama-3.2-1B" \
    --data_name="nqswap" \
    --layer_ids 10 11 \
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
    --model_path="meta-llama/Llama-3.2-1B" \
    --data_name="nqswap" \
    --layer_ids 8 9 10 \
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
    --model_path="meta-llama/Llama-3.2-1B" \
    --data_name="nqswap" \
    --layer_ids 9 10 11 \
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
    --model_path="meta-llama/Llama-3.2-1B" \
    --data_name="nqswap" \
    --layer_ids 6 7 8 9 \
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
    --model_path="meta-llama/Llama-3.2-1B" \
    --data_name="nqswap" \
    --layer_ids 7 8 9 10 \
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
    --model_path="meta-llama/Llama-3.2-1B" \
    --data_name="nqswap" \
    --layer_ids 6 \
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
    --model_path="meta-llama/Llama-3.2-1B" \
    --data_name="nqswap" \
    --layer_ids 7 \
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
    --model_path="meta-llama/Llama-3.2-1B" \
    --data_name="nqswap" \
    --layer_ids 8 \
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
    --model_path="meta-llama/Llama-3.2-1B" \
    --data_name="nqswap" \
    --layer_ids 9 \
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
    --model_path="meta-llama/Llama-3.2-1B" \
    --data_name="nqswap" \
    --layer_ids 10 \
    --edit_degree=2.0 \
    --select_topk_proportion=0.07 \
    --seed=${seed} \
    --hiddens_name="grouped_activations" \
    --mutual_information_save_name="mutual_information" \
    --run_use_parameter \
    --run_use_context
done
