# experiments

Exploratory, results-driven research work: comparisons, ablations, and
"what happens if..." questions that take real compute and produce real
metrics. If `examples/` answers "how do I use this?", `experiments/` answers
"what happens when I compare/ablate this at scale?"

## Structure

**One folder per research question**, not per module — experiments often
compare multiple building blocks against each other, so they're organized
by question rather than by the single module they touch.

```
experiments/
├── attention_variant_comparison/
│   ├── config.yaml
│   ├── run.py
│   └── results/
│       ├── metrics.csv
│       ├── results.md        # plain-English summary of findings
│       └── plots/
├── positional_encoding_extrapolation/
├── moe_routing_analysis/
└── _template/                # copy this to start a new experiment
    ├── config.yaml
    ├── run.py
    └── results/.gitkeep
```

## Conventions

- **`config.yaml` + `run.py` pattern.** Hyperparameters live in the config,
  not hardcoded in the script — keeps experiments reproducible and diffable
  in PRs.
- **Commit lightweight results, not checkpoints.** `metrics.csv` and plots
  are worth versioning. Raw checkpoints/logs are large and should be
  `.gitignore`d or pointed at external storage (e.g. W&B) instead.
- **Write a `results.md`.** A short plain-English summary of what the
  experiment found. Future readers (including future-you) shouldn't have to
  re-run a script just to learn the conclusion.
- **Start from `_template/`.** Copy it rather than starting from scratch —
  keeps every experiment folder the same shape as the collection grows.

## Adding a new experiment

1. `cp -r experiments/_template experiments/<your_question>`
2. Fill in `config.yaml` with hyperparameters / variants under test.
3. Write `run.py` to load modules from `src/transformer_lab/`, run the
   comparison, and dump metrics/plots into `results/`.
4. Summarize findings in `results/results.md`.