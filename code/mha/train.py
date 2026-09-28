import os 
import torch 
import numpy as np 
import torch.nn as nn 
from pathlib import Path 
import argparse

from config import Config, load_config

# parents[0] = mha, parents[1] = code, parents[2] = transformer_lab
data_dir = Path(__file__).resolve().parents[2] / "data"
device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if "cuda" in device else "cpu"

def get_batch(split, batch_size, block_size):
    fname = "train.bin" if split == "train" else "validation.bin"
    data = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - block_size - 1, (batch_size, ))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])

    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y

def train(cfg: Config):
    xval, yval = get_batch(split="validation", batch_size=cfg.batch_size, block_size=cfg.block_size)
    print('input')
    print(xval)
    print('output')
    print(yval)
    print('shape of input and target')
    print(xval.shape, yval.shape)

def main_cli():
    parser = argparse.ArgumentParser(description="Pretraining on webtext using Mutli-Head Attention Arch.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)

if __name__ == "__main__":
    main_cli()




