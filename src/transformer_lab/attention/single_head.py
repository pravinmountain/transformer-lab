import math
from typing import Optional, Tuple 

import torch 
import torch.nn as nn 
from torch import Tensor 

from transformer_lab.core.interface import AttentionModule

class SingleHeadAttention(AttentionModule):
    """"""

    def __init__(
        self,
        d_model: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model 
        self.d_k = d_k if d_k is not None else d_model
        self.d_v = d_v if d_v is not None else self.d_k

        self.w_q = nn.Linear(d_model, self.d_k, bias=False)
        self.w_k = nn.Linear(d_model, self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, self.d_v, bias=False)
        self.w_out = nn.Linear(self.d_v, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None 
    ) -> Tuple[Tensor, Tensor]:
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = attn_weights @ v
        output = self.w_out(context)

        return output, attn_weights