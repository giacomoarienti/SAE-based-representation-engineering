import argparse
import os
import logging
import json
from spare.utils import PROJ_DIR
from spare.spare_for_generation import run_sae_patching_evaluate
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--layer_ids", type=int, nargs='+', required=True)
    parser.add_argument("--edit_degree", type=float, required=True)
    parser.add_argument("--select_topk_proportion", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hiddens_name", type=str, required=True)
    parser.add_argument("--mutual_information_save_name", type=str, required=True)
    parser.add_argument("--run_use_parameter", action="store_true")
    parser.add_argument("--run_use_context", action="store_true")
    return parser.parse_args()


def main():
    output_dir = PROJ_DIR / "spare_outputs"

    args = get_args()
    logger.info(f"\n{json.dumps(vars(args), indent=4)}")

    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.basename(args.model_path)
    str_layer_ids = ",".join([str(lid) for lid in args.layer_ids])
    exp_name = f"{model_name}-{args.data_name}-{str_layer_ids}-{args.select_topk_proportion}-{args.edit_degree}-" \
               f"{args.hiddens_name}-{args.mutual_information_save_name}-{args.run_use_parameter}-" \
               f"{args.run_use_context}-{args.seed}"


    wandb.init(
        project="spare",
        id=exp_name,
        config=vars(args)
    )

    output_path = output_dir / exp_name

    results = run_sae_patching_evaluate(
        model_path=args.model_path,
        data_name=args.data_name,
        layer_ids=args.layer_ids,
        seed=args.seed,
        k_shot=3,
        edit_degree=args.edit_degree,
        hiddens_name=args.hiddens_name,
        mutual_information_save_name=args.mutual_information_save_name,
        select_topk_proportion=args.select_topk_proportion,
        run_use_parameter=args.run_use_parameter,
        run_use_context=args.run_use_context,
    )
    json.dump(results, open(output_path, "w"), indent=4, ensure_ascii=False)
    wandb.finish()

import resource
import sys

def set_memory_limit(max_memory_gb):
    """
    Set a maximum memory usage limit (in GB) for the current process.
    """
    max_memory_bytes = max_memory_gb * 1024 ** 3
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, hard))

if __name__ == '__main__':
    MAX_RAM_MEMORY = 40
    try:
        set_memory_limit(MAX_RAM_MEMORY)
        main()
    except MemoryError:
        sys.stderr.write('Memory limit exceeded.\n')
        sys.exit(1)
