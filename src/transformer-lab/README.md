# transformer_lab (src)

This is the core library — the actual, importable implementations. Everything
here is meant to be **correct, minimal, and self-contained** rather than
optimized for production use.

## Structure

```
transformer_lab/
├── core/                # shared base classes / interfaces (e.g. AttentionModule)
├── attention/            # scaled dot-product, MHA, GQA, MQA, sliding window, linear...
├── positional_encoding/   # sinusoidal, learned, RoPE, ALiBi
├── normalization/         # LayerNorm, RMSNorm, pre/post-norm wrappers
├── feedforward/            # standard MLP, gated MLP (SwiGLU/GeGLU), MoE
├── tokenization/            # BPE, WordPiece
├── optimization/             # warmup schedules, custom optimizers
├── models/                    # assembled architectures (GPT, ViT, encoder-decoder...)
└── utils/                       # benchmarking helpers, attention visualization
```

## Conventions

- **One concept per file.** A file should be readable top-to-bottom without
  jumping elsewhere to understand the algorithm.
- **Consistent interfaces.** Every module in a given category (e.g. every
  attention variant) implements the same base class / call signature from
  `core/interfaces.py`, so implementations are swappable and directly
  comparable in experiments and benchmarks.
- **Match paper terminology in naming.** File and class names should mirror
  the terms used in the source paper (`rotary.py`, not `pe_v2.py`) so the
  right implementation is easy to find.
- **Docstrings link to the paper.** Every module's docstring should include
  the paper title/link and a one-line description of what makes this variant
  different from the others in its category.
- **No hidden cross-imports between categories** unless assembling a model in
  `models/`. Building blocks should not depend on each other.

## Adding a new implementation

1. Pick the right category folder (or propose a new one if it doesn't fit).
2. Subclass the relevant interface from `core/`.
3. Add a docstring with the paper reference and complexity notes.
4. Add a mirrored test in `tests/`.
5. Optionally add a demo notebook/script in `examples/`.