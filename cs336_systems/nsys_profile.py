import cs336_basics as lib
import cs336_basics.model as nn
import cs336_basics.optimizer as optim
import torch
import torch.cuda.nvtx as nvtx
import timeit
import argparse
import logging
import json
from typing import Callable, Dict, Any
from pathlib import Path
from nn_utils import cross_entropy

# Model configuration dictionaries
CONFIGS = json.load(open("model_configs.json", "r"))

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
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

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Transformer language models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', 
        choices=list(CONFIGS.keys()),
        default="small",
        help='Model configurations to benchmark'
    )
    
    parser.add_argument(
        '--gpu-index', 
        type=int, 
        default=0,
        help='GPU index to use (if available)'
    )

    parser.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
        default='INFO',
        help='Logging level'
    )

    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=1,
        help='Batch size for benchmarking'
    )
    
    parser.add_argument(
        '--num-steps', 
        type=int, 
        default=20,
        help='Number of steps in training loop'
    )
    
    parser.add_argument(
        '--sequence-length', 
        type=int, 
        default=128,
        help='Sequence length for benchmarking'
    )

    parser.add_argument(
        '--use-optimizer', 
        action='store_true',
        help='Include optimizer.step()'
    )

    return parser.parse_args()

def run_training(args):

    # Define a model (with random weights)
    cfg = CONFIGS[args.config]
    logger.info(f"Benchmarking config: {args.config}")
    logger.info(f"Model parameters: {cfg}")
    device = get_device(args.gpu_index)
    
    with nvtx.range("define_model"):
        model = nn.BasicsTransformerLM(**cfg).to(device)
    
    # Initialize optimizer if requested
    if args.use_optimizer:
        optimizer = optim.AdamW(model.parameters())

    # Define an input and output
    with nvtx.range("define_input_output"):
        vocab_size = model.vocab_size
        X = torch.randint(
            high=vocab_size, size=(args.batch_size, args.sequence_length),
            device=next(model.parameters()).device
        )
        labels = torch.randint(0, vocab_size, (args.batch_size, args.sequence_length))
        labels = labels.to(next(model.parameters()).device)

    # Run the model `num_steps` times
    for step in range(args.num_steps):
        if step > 10:
            # start profiling after 10 warmup iterations
            torch.cuda.cudart().cudaProfilerStart()

        nvtx.range_push(f"step_{step}")
        
        # Zero gradients
        if args.use_optimizer:
            optimizer.zero_grad()
        else:
            model.zero_grad(set_to_none=True)

        # Forward
        with nvtx.range("forward"):
            logits = model(X) # (args.batch_size, args.sequence_length, vocab_size)
            loss = cross_entropy(logits, labels)

        # Backward
        with nvtx.range("backward"):
            loss.backward()

        # Optimizer step if enabled
        if args.use_optimizer:
            with nvtx.range("optimizer_step"):
                #print(f"Step {step}, loss: {y.item():.6f}")
                optimizer.step()
        
        nvtx.range_pop()

def main():
    args = parse_args()
    global logger
    logger = setup_logging(log_level=args.log_level)
    run_training(args)

if __name__ == "__main__":
    main()
    