import os 
import torch 
import numpy as np 
import torch.nn as nn 
from pathlib import Path 
import argparse
import torch.optim as optim

from config import Config, load_config
from mha import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = load_config(config_path=args.config)

    # parents[0] = mha, parents[1] = code, parents[2] = transformer_lab
    data_dir = Path(__file__).resolve().parents[2] / "data"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"

    def get_batch(split):
        fname = "train.bin" if split == "train" else "validation.bin"
        data = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode="r")

        ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size, ))
        x = torch.stack([torch.from_numpy(data[i : i + cfg.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + cfg.block_size].astype(np.int64)) for i in ix])

        if device_type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)

        return x, y

    model = Model(cfg=cfg)    

    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    for i in range(50):
        xb, yb = get_batch("train")
        optimizer.zero_grad()        
        logits, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        print(f'loss: {loss.item()}')
