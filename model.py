import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        x_ln1 = self.ln1(x)
        attn_output, _ = self.attn(x_ln1, x_ln1, x_ln1) # x_ln1 is used for query, key, and value
        x = x + attn_output
        x_ln2 = self.ln2(x)
        mlp_output = self.mlp(x_ln2)
        x = x + mlp_output
        return x

class LanguageModel(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_embd: int, n_layer: int, n_head: int):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # cross_entropy expects input of shape (N, C) and target of shape (N)
            loss = F.cross_entropy(logits, targets) 

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # get the last block_size tokens from idx
            # toy example: if idx is of shape (B, T) and block_size is 8, 
            # we want to get the last 8 tokens for each batch.
            # if T < block_size, we just take all tokens
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            # get the last token's logits
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            # we are sampling one token for each batch, so we use num_samples=1
            idx_next = torch.multinomial(probs, num_samples=1)
            # concatenate the new token to the existing idx
            idx = torch.cat((idx, idx_next), dim=1)
        return idx