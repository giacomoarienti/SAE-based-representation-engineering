import os
import json
import argparse
import torch
from pathlib import Path

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memorised set creator script.

This script uses pre-computed closed-book evaluation results to identify 
which QA samples can be answered correctly without context. Samples that 
achieve an exact match score of 1 are saved to the memorized set.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Create a memorised set of QA samples from pre-computed evaluations")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset", type=str, default="nqswap", help="Dataset used in evaluation")
    parser.add_argument("--k_shot", type=int, default=32, help="Number of shots used in evaluation")
    parser.add_argument("--cache_dir", type=str, default="./cache_data", help="Directory containing cached evaluation results")
    return parser.parse_args()


def create_memorised_set(args):
    # Extract model name from path
    model_name = Path(args.model_path).name
    
    # Construct path to the cached evaluation file
    cached_file_path = os.path.join(
        args.cache_dir,
        "prepare_eval", 
        f"{args.dataset}-{model_name}-{args.k_shot}shot-examples-closebook/results.json"
    )
    
    print(f"Loading cached evaluation results from {cached_file_path}...")
    
    if not os.path.exists(cached_file_path):
        raise FileNotFoundError(f"Cached evaluation file not found at {cached_file_path}")
    
    # Load cached evaluation results
    with open(cached_file_path, 'r') as f:
        eval_results = json.load(f)
    
    # Extract index of memorised samples
    all_scores = torch.tensor(eval_results["all_close_book_scores"])
    tensor = torch.nonzero(all_scores).squeeze()
    return set(tensor.tolist())


def main():
    args = parse_args()
    model_name = Path(args.model_path).name
    output_file = f"{args.cache_dir}/{args.dataset}-{model_name}-memorised_set"
    
    print(f"Creating memorised set for model {args.model_path}...")
    memorised_samples = create_memorised_set(args)
    
    print(f"Found {len(memorised_samples)} memorised samples")
    
    # save to output path
    torch.save(memorised_samples, output_file)
    
    print(f"Memorised set saved to {output_file}")


if __name__ == "__main__":
    main()
