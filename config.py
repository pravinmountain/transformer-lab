import yaml 
from pydantic import BaseModel

class Config(BaseModel):
    file_path: str = "lotr/lord-of-the-rings.txt"
    block_size: int = 8
    batch_size: int = 4


def load_config(config_path: str) -> Config:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)