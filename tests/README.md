# tests

Correctness checks for every implementation in `src/transformer_lab/`.
This folder **mirrors the `src/transformer_lab/` structure 1:1** — every
module should have a corresponding test file at the matching path.

## Structure

```
tests/
├── attention/
│   ├── test_scaled_dot_product.py
│   ├── test_multi_head.py
│   └── test_grouped_query.py
├── positional_encoding/
│   └── test_rotary.py
├── normalization/
└── ...
```

## What every module's test should cover

1. **Shape correctness** — output shapes match expectations for a range of
   input shapes (batch size, sequence length, head count, etc.).
2. **Gradient flow** — a backward pass runs cleanly and gradients are
   non-zero/non-NaN where expected.
3. **Numerical equivalence to a reference**, where one exists — e.g. a custom
   multi-head attention implementation should match
   `torch.nn.MultiheadAttention` (or the paper's reference implementation)
   on the same weights within a numerical tolerance.
4. **Edge cases relevant to the algorithm** — e.g. masking behavior for
   attention, extrapolation length for positional encodings, expert
   capacity overflow for MoE.

## Conventions

- Use `pytest`. One test file per source module, named `test_<module>.py`.
- Prefer small, fast, CPU-only tests — this suite should run in seconds, not
  minutes. Anything requiring real training belongs in `experiments/`.
- New PRs adding a module to `src/transformer_lab/` should include a
  matching test file in the same PR.