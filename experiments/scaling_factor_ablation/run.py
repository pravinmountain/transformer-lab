"""
Experiment: does scaling by 1/sqrt(d_k) actually prevent softmax saturation?

Question
--------
"Attention Is All You Need" scales dot-product scores by 1/sqrt(d_k)
specifically because, for large d_k, the dot products grow large in
magnitude and push softmax into regions with extremely small gradients
(Section 3.2.1). This experiment measures that effect directly using
SingleHeadAttention's own Q/K projections, across a range of d_k, with and
without the scaling factor.

Metric: average Shannon entropy (bits) of the attention distribution.
Lower entropy = a more peaked/saturated softmax (closer to one-hot) =
smaller gradients flowing back through it. Max possible entropy for a
uniform distribution over `seq_len` keys is log2(seq_len).
"""

import math
from pathlib import Path

import torch
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from transformer_lab.attention.single_head import SingleHeadAttention

HERE = Path(__file__).parent


def attention_entropy(weights: torch.Tensor) -> float:
    """Mean Shannon entropy (bits) across all query positions/batches."""
    eps = 1e-12
    entropy = -(weights * (weights + eps).log2()).sum(dim=-1)
    return entropy.mean().item()


def run():
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    torch.manual_seed(cfg["seed"])

    rows = []
    for d_k in cfg["d_k_values"]:
        scaled_entropies, unscaled_entropies = [], []

        for _ in range(cfg["num_trials"]):
            attn = SingleHeadAttention(d_model=cfg["d_model"], d_k=d_k, d_v=d_k)
            x = torch.randn(cfg["batch_size"], cfg["seq_len"], cfg["d_model"])

            with torch.no_grad():
                q = attn.w_q(x)
                k = attn.w_k(x)

                raw_scores = q @ k.transpose(-2, -1)
                scaled_scores = raw_scores / math.sqrt(d_k)

                scaled_weights = torch.softmax(scaled_scores, dim=-1)
                unscaled_weights = torch.softmax(raw_scores, dim=-1)

            scaled_entropies.append(attention_entropy(scaled_weights))
            unscaled_entropies.append(attention_entropy(unscaled_weights))

        rows.append({
            "d_k": d_k,
            "scaled_entropy_bits": sum(scaled_entropies) / len(scaled_entropies),
            "unscaled_entropy_bits": sum(unscaled_entropies) / len(unscaled_entropies),
            "max_possible_entropy_bits": math.log2(cfg["seq_len"]),
        })

    df = pd.DataFrame(rows)

    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    df.to_csv(results_dir / "metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["d_k"], df["scaled_entropy_bits"], marker="o", label="scaled by 1/sqrt(d_k)")
    ax.plot(df["d_k"], df["unscaled_entropy_bits"], marker="o", label="unscaled (raw dot product)")
    ax.axhline(df["max_possible_entropy_bits"].iloc[0], color="gray", linestyle="--",
               label="max entropy (uniform attention)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("d_k")
    ax.set_ylabel("attention entropy (bits)")
    ax.set_title("Effect of 1/sqrt(d_k) scaling on softmax saturation")
    ax.legend()
    fig.tight_layout()

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    fig.savefig(plots_dir / "entropy_vs_dk.png", dpi=150)

    print(df.to_string(index=False))
    print(f"\nSaved metrics to {results_dir / 'metrics.csv'}")
    print(f"Saved plot to   {plots_dir / 'entropy_vs_dk.png'}")


if __name__ == "__main__":
    run()
