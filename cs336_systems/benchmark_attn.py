import functools
import json
import logging
import math
import os
import timeit
import argparse
from einops import rearrange, einsum
import einx

from typing import Callable, Dict, Any
from pathlib import Path
from functools import wraps
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch import Tensor
import torch.cuda.nvtx as nvtx

from attention import (
    _make_attn_inputs,
    scaled_dot_product_attention,
    scaled_dot_product_attention
)

logger = logging.getLogger("benchmarking")

def setup_logging(log_level: str = "WARNING") -> logging.Logger:
    """Setup logging configuration."""
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def get_device(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{index}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(index)}")
        return device
    else:
        logger.info("CUDA not available, using CPU")
        return torch.device("cpu")

def mean(x: list[float]) -> float:
    return sum(x) / len(x)

def std(x: list[float]) -> float:
    if len(x) <= 1:
        return 0.0
    mu = mean(x)
    return (sum((xi - mu) ** 2 for xi in x) / (len(x) - 1)) ** 0.5

def memory_profiling(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping memory profiling")
            return func(*args, **kwargs)
            
        logger.info("Enabled memory profiling.")
        torch.cuda.memory._record_memory_history(max_entries=100000) # Start recording memory history
        result = func(*args, **kwargs)
        ## Output handling
        output_dir = Path("memory_snapshots")
        output_dir.mkdir(exist_ok=True)
        snapshot_file = output_dir / \
            f"naive_attention-{args[0].n_queries}-{args[0].head_dim}.pickle"
        torch.cuda.memory._dump_snapshot(snapshot_file)
        torch.cuda.memory._record_memory_history(enabled=None) # Stop recording history.
        logger.info(f"Saved memory snapshot to {snapshot_file}")
        return result
    return wrapper

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark attention implementation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n-queries', type=int, default=128)
    parser.add_argument('--n-keys', type=int, default=128)
    parser.add_argument('--head-dim', type=int, default=64)
    parser.add_argument('--num-warmups', type=int, default=10)
    parser.add_argument('--num-trials', type=int, default=100)
    parser.add_argument('--csv', type=str, default="results/naive_attention.csv")
    
    return parser.parse_args()

def benchmarking(args):
    logger.info(f"Benchmarking config: {vars(args)}")
    device = get_device(args.gpu_index)
    Q, K, V, dO = _make_attn_inputs(
        device=device,
        batch_size=8,
        n_queries=args.n_queries,
        n_keys=args.n_keys,
        head_dim=args.head_dim
    )

    def _forward():
        outputs = scaled_dot_product_attention(Q, K, V, mask=None)

    run = _forward
    logger.info(f"Running {args.num_warmups} warmup iterations...")
    for _ in range(args.num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info(f"Running {args.num_trials} benchmark iterations...")
    times = []
    for trial in range(args.num_trials):
        start_time = timeit.default_timer()
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append((end_time - start_time) * 1000)

    statistics = dict(
        mean=mean(times),
        std=std(times),
        min=min(times),
        max=max(times)
    )
    statistics = {k: f"{v:6.2f}" for k, v in statistics.items()}
    return statistics

def main():
    args = parse_args()
    setup_logging()
    run = memory_profiling(benchmarking)       
    statistics = run(args)
    
    # logger.info(f"Running time:\n  {statistics}")
    if args.csv:
        output_file = Path(args.csv)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        content = (
            f"{args.head_dim},{args.n_queries},"
            f"{statistics['mean'].strip()},{statistics['std'].strip()},"
            f"{statistics['min'].strip()},{statistics['max'].strip()}\n"
        )
        with open(output_file, 'a') as f:
            f.write(content)

if __name__ == "__main__":
    main()

    










    