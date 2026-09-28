# Configuration for Multi-Head Attention
from typing import Literal

import yaml
from pydantic import BaseModel

class Config(BaseModel):
    """"""
    block_size: int = 1024
    batch_size: int = 16

def load_config(config_path: str) -> Config:
    """Load configuration from a YAML file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)