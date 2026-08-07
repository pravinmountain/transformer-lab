import yaml 
from pydantic import BaseModel

class Config(BaseModel):
    file_path: str = "data/originofspecies00darwuoft_djvu.txt"
    block_size: int = 8
    batch_size: int = 4
    vocab_size: int = 50257
    n_layer: int = 4
    n_heads: int = 4
    n_embd: int = 32
    encoding: str = 'gpt2'


def load_config(config_path: str) -> Config:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)