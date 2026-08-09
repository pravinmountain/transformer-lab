# examples

Clean, minimal, **teaching-oriented** demos of how to use each module in
`src/transformer_lab/`. If `experiments/` answers "what happens when...",
`examples/` answers "how do I use this?"

## Structure

Mirrors the `src/transformer_lab/` category layout:

```
examples/
├── attention/
│   └── multi_head_demo.ipynb
├── positional_encoding/
├── models/
│   └── train_tiny_gpt.py
└── ...
```

## What belongs here

- A notebook or script that imports one (or a small handful of) module(s)
  and shows it working on a toy input — shapes in, shapes out, a plot if
  relevant (e.g. attention maps, positional encoding curves).
- Should run in under a minute on CPU. No real training runs, no large
  datasets, no hyperparameter sweeps — that's what `experiments/` is for.
- Prefer notebooks for anything visual (attention patterns, embeddings);
  prefer plain `.py` scripts for anything meant to be copy-pasted.

## What doesn't belong here

- Comparative studies or ablations → `experiments/`
- Speed/memory profiling → `benchmarks/`
- Correctness checks → `tests/`

## Adding a new example

1. Put it in the folder matching its module's category in `src/`.
2. Keep it to a single concept — don't combine five modules into one demo.
3. Add a one-line comment at the top stating what it demonstrates and which
   module(s) it imports.