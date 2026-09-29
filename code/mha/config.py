# Configuration for Multi-Head Attention
from typing import Literal

import yaml
from pydantic import BaseModel

class Config(BaseModel):
    """"""
    block_size: int = 8
    batch_size: int = 4
    d_model:    int = 32
    head_size:  int = 16
    n_heads:    int = 2
    vocab_size: int = 50257

def load_config(config_path: str) -> Config:
    """Load configuration from a YAML file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)