from transformer_lab.attention.config import AttentionConfig
from abc import ABC, abstractmethod
from typing import Optional
import torch 
import torch.nn as nn 

class AttentionBase(nn.Module, ABC):
    """Base Interface for all attention mechanisms."""

    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.cfg = cfg 


    @abstractmethod
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the attention output.

        Args:
            x:    Input tensor with shape (batch, seq_len, d_model).
            mask: Optional mask tensor with shape (batch, seq_len, seq_len).
                  None is valid because linear attention variants (e.g. delta attention)
                  do not use an attention matrix and ignore this argument.

        Returns:
            torch.Tensor: Output tensor with shape (batch, seq_len, d_model).
        """
        ...
