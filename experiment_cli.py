#!/usr/bin/env python3

import subprocess
import sys
import argparse


def benchmark():
    warmups_values = [0, 5]
    end_to_end_values = ["", "--end-to-end"]
    configs = [
        "small", "medium", "large", "xl", "2.7B"
    ]
    
    for warmups in warmups_values:
        for end_to_end in end_to_end_values:
            # Display what we're running (equivalent to echo)
            end_to_end_display = "True" if end_to_end else "False"
            print(f"Running: warmups={warmups}, end_to_end={end_to_end_display}")
            
            # Build the command arguments
            cmd = [
                "python", "cs336_systems/measure.py",
                "--configs", configs[0], 
                "--num-warmups", str(warmups),
                "--quiet",
                "--log-level", "INFO"
            ]
            
            # Add end_to_end flag if it's not empty
            if end_to_end: cmd.append("--end-to-end")
            
            # Run the command
            print(cmd)
            try:
                result = subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Command failed with return code {e.returncode}", file=sys.stderr)
                sys.exit(e.returncode)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Transformer language models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--benchmark', 
        action='store_true',
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.benchmark: benchmark()