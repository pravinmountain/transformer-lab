"""
Single-Head Attention Demo
===========================
Shows SingleHeadAttention running on a toy sequence: prints the shapes at
each step, then plots the resulting attention weight matrix as a heatmap
so you can see which tokens attend to which.

Run:
    python examples/attention/single_head_demo.py
"""

import torch
import matplotlib.pyplot as plt

from transformer_lab.attention.single_head import SingleHeadAttention

torch.manual_seed(42)

# --- toy "sentence" of 6 tokens, embedding dim 16 (random, untrained) ---
tokens = ["The", "cat", "sat", "on", "the", "mat"]
batch, seq_len, d_model = 1, len(tokens), 16
x = torch.randn(batch, seq_len, d_model)

attn = SingleHeadAttention(d_model=d_model, d_k=8, d_v=8)
output, attn_weights = attn(x, x, x)

print(f"Input shape:             {tuple(x.shape)}")
print(f"Output shape:            {tuple(output.shape)}")
print(f"Attention weights shape: {tuple(attn_weights.shape)}")
print()
print("Attention weights (query rows -> key columns):")
print(attn_weights[0].detach().round(decimals=2))

# --- visualize the attention pattern for the single batch element ---
weights = attn_weights[0].detach().numpy()

fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(weights, cmap="viridis")
ax.set_xticks(range(seq_len))
ax.set_yticks(range(seq_len))
ax.set_xticklabels(tokens, rotation=45)
ax.set_yticklabels(tokens)
ax.set_xlabel("Key / Value tokens")
ax.set_ylabel("Query tokens")
ax.set_title("Single-Head Attention Weights\n(untrained, random init)")
fig.colorbar(im, ax=ax, label="attention weight")
fig.tight_layout()

out_path = "single_head_attention_weights.png"
fig.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
