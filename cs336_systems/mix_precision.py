import torch.nn as nn
import torch
import logging
import argparse
from contextlib import nullcontext
import torch.cuda.nvtx as nvtx

logger = logging.getLogger(__name__)
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    formatter = logging.Formatter("%(message)s")
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

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        logger.info(f"FORWARD PASS:")
        x = self.relu(self.fc1(x))
        logger.info(f"  first feed-forward: {x.dtype}")
        x = self.ln(x)
        logger.info(f"  layer norm: {x.dtype}")
        x = self.fc2(x)
        logger.info(f"  second feed-forward (logits): {x.dtype}")
        return x

class Trainer:
    def __init__(
        self,
        model,
        data,
        optimizer,
        num_steps=1
    ):
        self.model = model
        self.optimizer = optimizer
        self.data = data
        self.num_steps = num_steps

    def step(self, x, y):
        self.optimizer.zero_grad()
        outputs = self.model(x)
        loss = ((y - outputs) ** 2).sum()
        logger.info(f"  loss: {loss.dtype}")
        loss.backward()
        logger.info(f"BACKWARD PASS:\n  model's gradients: {self.model.fc1.weight.grad.dtype}")
        self.optimizer.step()

    def train(self):
        logger.info(f"MODEL INIT:\n  model parameters: {next(self.model.parameters()).dtype}")
        x, y = self.data[0], self.data[1]
        for step in range(self.num_steps):
            self.step(x, y)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mixed precision training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--dtype', 
        type=str,
        default='fp32',
        choices=['fp32', 'fp16', 'bf16'],
        help='Data type for training: fp32 (full precision), fp16 (half precision), or bf16 (bfloat16)'
    )

    return parser.parse_args()
    
def train():
    args = parse_args()
    setup_logging()
    device = get_device(index=0)
    batch_size, in_features, out_features = 16, 8, 4
    model = ToyModel(in_features, out_features).to(device)
    data = (
        torch.rand(batch_size, in_features, device=device),
        torch.rand(batch_size, out_features, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters())
    trainer = Trainer(model, data, optimizer)

    # Map dtype string to torch dtype and determine context
    dtype_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }
    
    dtype = dtype_map[args.dtype]
    
    if args.dtype == 'fp32':
        context = nullcontext()
        logger.info("Training with full precision (fp32).")
    else:
        context = torch.autocast(device_type="cuda", dtype=dtype)
        logger.info(f"Training with mixed precision ({args.dtype}).")
    
    with context:
        trainer.train()

if __name__ == "__main__":
    train()