#!/bin/bash
 
PHYS_DIR="/home/arienti/SAE-based-representation-engineering" # e.g., /home/molfetta/project1
LLM_CACHE_DIR="/llms"
DOCKER_INTERNAL_CACHE_DIR="/llms"

# Check if required environment variables are set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set."
    echo "Please set it with: export HF_TOKEN=your_huggingface_token"
    exit 1
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY environment variable is not set."
    echo "Please set it with: export WANDB_API_KEY=your_wandb_key"
    exit 1
fi

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_TOKEN="$HF_TOKEN" \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    --rm \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    --memory="10g" \
    -w /workspace \
    spare \
    "/workspace/run.sh"