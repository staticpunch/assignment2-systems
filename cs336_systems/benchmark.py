import cs336_basics as lib
import cs336_basics.model as nn
import torch
import timeit
import argparse
import logging
import json
from typing import Callable, Dict, Any
from pathlib import Path

# Model configuration dictionaries
CONFIGS = json.load(open("model_configs.json", "r"))

def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
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
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
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

def benchmark(
    model,
    batch_size: int = 1,
    sequence_length: int = 64,
    end_to_end: bool = False,
    num_warmups: int = 1, 
    num_trials: int = 3
) -> Dict[str, float]:
    """Benchmark a single model configuration."""
    # Initialization
    vocab_size = model.vocab_size
    X = torch.randint(high=vocab_size, size=(batch_size, sequence_length))
    X = X.to(next(model.parameters()).device)
    Y = torch.ones(vocab_size).to(next(model.parameters()).device)

    def forward():
        with torch.no_grad(): model(X)

    def backward():
        output = model(X)
        loss = ((Y - output) ** 2).sum()
        loss.backward()

    run = backward if end_to_end else forward
    
    # Warmup runs
    logger.debug(f"Running {num_warmups} warmup iterations...")
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark runs
    logger.debug(f"Running {num_trials} benchmark iterations...")
    times: list[float] = []
    for trial in range(num_trials):
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
    return statistics

def benchmark_statistics(args):
    if args.configs is None:
        args.configs = list(CONFIGS.keys())
    
    logger.info(f"Starting benchmark with configs: {args.configs}")
    logger.info(f"Parameters: batch_size={args.batch_size}, seq_len={args.sequence_length}, "
               f"end_to_end={args.end_to_end}, warmups={args.num_warmups}, trials={args.num_trials}")
    
    all_statistics = {}
    device = get_device(args.gpu_index)
    
    for cfg_name in args.configs:
        if cfg_name not in CONFIGS:
            logger.warning(f"Config '{args.cfg_name}' not found, skipping...")
            continue
            
        cfg = CONFIGS[cfg_name]
        logger.info(f"Benchmarking config: {cfg_name}")
        logger.debug(f"Model parameters: {cfg}")

        if args.annotate_attention:
            from attention import annotated_scaled_dot_product_attention
            nn.scaled_dot_product_attention = annotated_scaled_dot_product_attention

        model = nn.BasicsTransformerLM(**cfg).to(device)
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model {cfg_name} loaded with {num_params:,} parameters")
        
        statistics = benchmark(
            model, 
            args.batch_size, 
            args.sequence_length,
            args.end_to_end, 
            args.num_warmups, 
            args.num_trials
        )
        
        all_statistics[cfg_name] = statistics
        logger.info(
            f"Config {cfg_name}: {statistics['mean']:.2f}±{statistics['std']:.2f}ms "
            f"(min: {statistics['min']:.2f}ms, max: {statistics['max']:.2f}ms)"
        )
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                
    return all_statistics

def save_results(results: Dict[str, Any], output_file: str):
    """Save benchmark results to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Transformer language models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        '--configs', 
        nargs='+', 
        choices=list(CONFIGS.keys()),
        default=None,
        help='Model configurations to benchmark (default: all)'
    )
    
    # Benchmark parameters
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=1,
        help='Batch size for benchmarking'
    )
    
    parser.add_argument(
        '--sequence-length', 
        type=int, 
        default=128,
        help='Sequence length for benchmarking'
    )
    
    parser.add_argument(
        '--end-to-end', 
        action='store_true',
        help='Include backward pass in benchmark (default: forward only)'
    )
    
    parser.add_argument(
        '--num-warmups', 
        type=int, 
        default=5,
        help='Number of warmup iterations'
    )
    
    parser.add_argument(
        '--num-trials', 
        type=int, 
        default=10,
        help='Number of benchmark trials'
    )
    
    parser.add_argument(
        '--gpu-index', 
        type=int, 
        default=0,
        help='GPU index to use (if available)'
    )
    
    # Logging and output
    parser.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--log-file', 
        type=str,
        help='Log file path (optional)'
    )
    
    parser.add_argument(
        '--output-file', 
        type=str,
        help='Output JSON file for results (optional)'
    )
    
    parser.add_argument(
        '--quiet', 
        action='store_true',
        help='Suppress console output (only log to file)'
    )

    parser.add_argument(
        '--annotate_attention', 
        action='store_true',
        help='Add NVTX annotatton to scaled dot product attention'
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging(
        log_level=args.log_level if not args.quiet else 'ERROR',
        log_file=args.log_file
    )
    
    logger.info("Starting Transformer benchmark")
    logger.info(f"Arguments: {vars(args)}")
    
    # Run benchmark
    try:
        results = benchmark_statistics(args)
        full_results = {
            'metadata': vars(args),
            'results': results
        }
        print("Benchmark Results:")
        for cfg_name, stats in results.items():
            print(f"{cfg_name:>8}: {stats['mean']:6.2f} ± {stats['std']:5.2f}ms")
        print()
        
        if args.output_file:
            save_results(full_results, args.output_file)
            
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main()