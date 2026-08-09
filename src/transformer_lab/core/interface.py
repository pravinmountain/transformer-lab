"""
Shared abstract base classes ("interfaces") that every implementation in a 
given category must follow.

Why this exists
---------------
transformer-lab holds many alternative implementations of the same concept
(e.g. a dozen attention variants). Without a shared interface, each one
would invent its own method names and call signature, making it impossible
to:
    - swap one attention variant for another inside a model without rewriting
    surrounding code
    - loop over every variant in a benchmark/experiment script generically
    - test them all with the same test harness

Every module in `attention/` subclasses `AttentionModule` below and must 
implement `forward(query, key, value, mask=None)` returning
`(output, attn_weights)`. Other categories (positional_encoding,
normalization, feedforward, ...) will get their own interface here as they
are implemented, following the same pattern.
"""

from abc import ABC, abstractmethod 
from typing import Optional, Tuple 

import torch.nn as nn 
from torch import Tensor 

class AttentionModule(nn.Module, ABC):
    """Common interface for every attention variant in transformer_lab.attention.
    
    Subclasses must implement `forward`. All attention modules take
    query/key/value tensors of shape (batch, seq_len, d_model) and an 
    optional mask, and return attened output plus the attention
    weights (useful for inspection, visualization, and testing).
    """
    @abstractmethod
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args: 
            query:  (batch, seq_len_q, d_model)
            key:    (batch, seq_len_k, d_model)
            value:  (batch, seq_len_k, d_model)
            mask:   optional bool/int tensor broadcastable to
                    (batch, seq_len_q, seq_len_k). Nonzero/True = attend,
                    0/False = block (implementation apply -inf before softmax)

        Returns:
            outputs:        (batch, seq_len_q, d_model)
            attn_weights:   (batch, seq_len_q, seq_len_k)
        """
        raise NotImplementedError 