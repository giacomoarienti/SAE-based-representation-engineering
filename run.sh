pip install -e . 

python ./scripts/run_spare.py \
    --model_path="meta-llama/Meta-Llama-3-8B" \
    --data_name="nqswap" \
    --layer_ids 13 14 15 16 \
    --edit_degree=2.0 \
    --select_topk_proportion=0.07 \
    --seed=42 \
    --hiddens_name="grouped_activations" \
    --mutual_information_save_name="mutual_information" \
    --run_use_parameter \
    --run_use_context