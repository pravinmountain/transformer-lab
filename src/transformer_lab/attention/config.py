from dataclasses import dataclass

@dataclass
class AttentionConfig:
    """Config for all attention mechanisms."""
    d_model: int 
    n_heads: int 
    dropout: float = 0.0
    bias: bool = True 
