import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from config import load_config, Config

class Head(nn.Module):
    """Single Self Attention Head"""
    def __init__(self, cfg: Config):
        super().__init__()
        self.head_size = cfg.head_size
        self.key   = nn.Linear(cfg.d_model, self.head_size, bias=False)
        self.query = nn.Linear(cfg.d_model, self.head_size, bias=False)
        self.value = nn.Linear(cfg.d_model, self.head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x):
        B, T, C = x.size()
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(2, 1) * self.head_size ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ v
        
class MultiHeadAttention(nn.Module):
    """"""
    def __init__(self, cfg: Config):
        super().__init__()
        self.heads = nn.ModuleList([Head(cfg) for _ in range(cfg.n_heads)])
        self.proj  = nn.Linear(cfg.head_size * cfg.n_heads, cfg.d_model, bias=False)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.proj(out)

class Model(nn.Module):
    """"""
    def __init__(self, cfg: Config):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.heads = MultiHeadAttention(cfg=cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, x, targets):
        token_embed = self.tok_emb(x)
        posit_embed = self.pos_emb(torch.arange(x.shape[1]))
        x = token_embed + posit_embed
        out = self.heads(x)
        logits = self.lm_head(out)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

