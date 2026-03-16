# Residual Stream Geometry in a Non-Ergodic Mess3 Mixture

Analysis of how transformers represent belief states and component identity when trained on a non-ergodic mixture of Mess3 Hidden Markov Models.

## Overview

We train a small transformer on next-token prediction over a non-ergodic dataset constructed as a mixture of K=3 distinct Mess3 ergodic processes. Each training sequence is generated entirely by one component, requiring the model to implicitly handle both within-component belief tracking and between-component uncertainty.

See [FINAL_REPORT.md](FINAL_REPORT.md) for the full report with all results.

## Quick Start

```bash
pip install -r requirements.txt

# 1. Generate data (50k sequences per component)
python data/generate_nonergodic_mess3.py --n_train 150000 --n_val 15000 --output_dir results

# 2. Train full transformer
python train/train_transformer.py --config configs/main_full.json \
    --data_dir results --output_dir results/checkpoints

# 3. Run baseline analysis (PCA, probes, emergence)
python analysis/analyze_geometry.py --results_dir results \
    --checkpoint_dir results/checkpoints --device cuda
python analysis/extra_analysis.py --results_dir results \
    --checkpoint_dir results/checkpoints --device cuda

# 4. Run experiments 1-5, 7
python analysis/run_all_experiments.py --results_dir results \
    --checkpoint_dir results/checkpoints --device cuda

# 5. Run experiment 6 (K scaling)
python analysis/run_experiment6.py --output_dir results/experiments --device cuda
```

Use `--device cpu` if no GPU is available (slower).

## Smoke Test

```bash
python data/generate_nonergodic_mess3.py --n_train 1000 --n_val 200 --output_dir results
python train/train_transformer.py --config configs/smoke.json \
    --data_dir results --output_dir results/checkpoints
```

## Project Structure

```
├── data/
│   └── generate_nonergodic_mess3.py   # Dataset generation with ground-truth beliefs
├── train/
│   └── train_transformer.py           # Model training (full transformer with MLP)
├── analysis/
│   ├── analyze_geometry.py            # PCA and linear probe analysis
│   ├── extra_analysis.py              # Emergence analysis (layer × position)
│   ├── run_all_experiments.py         # Experiments 1-5, 7
│   └── run_experiment6.py             # Experiment 6 (dimensionality scaling with K)
├── configs/
│   ├── main_full.json                 # Full transformer config
│   └── smoke.json                     # Quick test config
├── fwh_core/                          # Core library (Mess3, HMM, etc.)
├── experiments/models/                # Model definitions
├── FINAL_REPORT.md                    # Full report with all results
├── honor_code_prediction.md           # Pre-registered predictions
└── requirements.txt
```

## Key Results

| Experiment | Key Finding |
|---|---|
| **1. Hierarchical Posterior** | Joint posterior Y linearly accessible (R² ≈ 0.88) |
| **2. Dimensionality** | k*₀.₉₅ grows from 2→5+ with context position |
| **3. Separability** | MLP jump: comp R² 0.03→0.43 at Block 0 |
| **5. Fractal Recovery** | Per-component fractal geometry recovered (R² ≈ 0.85–0.95) |
| **7. Orthogonality** | comp-ID vs belief overlap < 0.01 (surprisingly factored) |

## Outputs

- `results/figures/` — All baseline analysis figures
- `results/experiments/` — Experiment 1-7 figures and results
- `results/checkpoints/` — Model weights and training history
- `results/analysis_results.json` — Probe R² and PCA metrics
