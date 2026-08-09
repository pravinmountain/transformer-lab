import math

import torch

from transformer_lab.attention.single_head import SingleHeadAttention

def test_output_shapes():
    batch, seq_len, d_model = 4, 8, 32
    attn = SingleHeadAttention(d_model)
    x = torch.randn(batch, seq_len, d_model)

    out, weights = attn(x, x, x)

    assert out.shape == (batch, seq_len, d_model)
    assert weights.shape == (batch, seq_len, seq_len)

def test_mismatched_d_k_and_d_v_shapes():
    """value projection and output layer must use d_v, not d_k."""
    batch, seq_len, d_model = 2, 5, 16
    d_k, d_v = 8, 12
    attn = SingleHeadAttention(d_model, d_k=d_k, d_v=d_v)
    x = torch.randn(batch, seq_len, d_model)

    out, weights = attn(x, x, x)

    assert out.shape == (batch, seq_len, d_model)
    assert weights.shape == (batch, seq_len, seq_len)
    assert attn.w_v.out_features == d_v
    assert attn.w_out.in_features == d_v

def test_cross_attention_shapes():
    """query and key/value can have different sequence lengths (e.g. encoder-decoder)."""
    batch, seq_len_q, seq_len_kv, d_model = 2, 4, 7, 16
    attn = SingleHeadAttention(d_model)
    q = torch.randn(batch, seq_len_q, d_model)
    kv = torch.randn(batch, seq_len_kv, d_model)

    out, weights = attn(q, kv, kv)

    assert out.shape == (batch, seq_len_q, d_model)
    assert weights.shape == (batch, seq_len_q, seq_len_kv)


def test_attention_weights_sum_to_one():
    attn = SingleHeadAttention(d_model=8)
    x = torch.randn(3, 6, 8)

    _, weights = attn(x, x, x)

    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_gradient_flow():
    attn = SingleHeadAttention(d_model=8)
    x = torch.randn(2, 4, 8, requires_grad=True)

    out, _ = attn(x, x, x)
    out.sum().backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    for p in attn.parameters():
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()


def test_causal_mask_blocks_future_positions():
    d_model, seq_len = 8, 5
    attn = SingleHeadAttention(d_model)
    x = torch.randn(1, seq_len, d_model)

    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    _, weights = attn(x, x, x, mask=causal_mask)

    upper_triangle = weights.masked_select(~causal_mask)
    assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle), atol=1e-6)


def test_matches_manual_scaled_dot_product():
    """Numerical equivalence against a plain-tensor reference computation."""
    torch.manual_seed(0)
    d_model = 8
    attn = SingleHeadAttention(d_model, dropout=0.0)
    x = torch.randn(1, 3, d_model)

    with torch.no_grad():
        q = attn.w_q(x)
        k = attn.w_k(x)
        v = attn.w_v(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(attn.d_k)
        ref_weights = torch.softmax(scores, dim=-1)
        ref_out = attn.w_out(ref_weights @ v)

    out, weights = attn(x, x, x)

    assert torch.allclose(out, ref_out, atol=1e-6)
    assert torch.allclose(weights, ref_weights, atol=1e-6)
