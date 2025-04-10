#!/bin/bash -l

set -e
set -u

if [ $# -lt 3 ]; then
  echo "Error: Required arguments not provided."
  echo "Usage: $0 <model_path> <dataset> <layers>"
  echo "Example: $0 /path/to/model \"1 2 3 4\""
  exit 1
fi

MODEL_PATH="${1}"
DATASET="${2}"
LAYERS="${3}"

LOAD_HIDDENS_NAME="grouped_activations"
MI_SAVE_NAME="mutual_information"

# Loop through each layer
for layer in $LAYERS; do
  python3 -m spare.mutual_information_and_expectation \
    --num_proc=64 \
    --data_name=${DATASET} \
    --model_path=${MODEL_PATH} \
    --load_hiddens_name=${LOAD_HIDDENS_NAME} \
    --layer_idx=${layer} \
    --minmax_normalisation \
    --mutual_information_save_name=${MI_SAVE_NAME}
done