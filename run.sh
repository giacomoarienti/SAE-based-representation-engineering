MODEL_PATH="meta-llama/Llama-3.2-1B"

pip install -e . 

# 1.
# sh ./scripts/run_prepare_eval.sh "${MODEL_PATH}"

# 2.
# sh ./scripts/run_memorised_set.sh "${MODEL_PATH}"

# 3.
# sh ./scripts/run_group_prompts.sh "${MODEL_PATH}"

# 4.
sh ./scripts/run_save_grouped_activations.sh "${MODEL_PATH}"