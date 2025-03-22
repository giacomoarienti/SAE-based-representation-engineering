PHYS_DIR="/home/arienti/SAE-based-representation-engineering" # e.g., /home/molfetta/project1
LLM_CACHE_DIR="/llms"
DOCKER_INTERNAL_CACHE_DIR="/llms"

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_TOKEN="$HF_TOKEN" \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    -w /workspace \
    --rm \
    -it \
    spare \
    bash