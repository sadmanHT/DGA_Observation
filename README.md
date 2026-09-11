# Adaptive Observation-Efficient Power Transformer Fault Diagnosis from Partial DGA Histories

Complete reproducibility repository for the updated IEEE-format paper.

## What this repository contains

The current paper is built from **three deliberately separated experimental protocols**. This repository
preserves all code that contributes to the reported results rather than presenting only the latest scripts.

```text
.
├── canonical/      # original 25/50/75/100% paper experiment
├── advanced/       # dense horizons, adaptive stopping, interaction, conformal, SHAP, stress tests
├── refinement/     # convergence-stable grid, Temporal-70, causal smoothing/noise refinement
├── notebooks/      # exact successful Kaggle execution notebooks
├── evidence/       # machine-readable outputs used by the paper
├── paper/          # current LaTeX manuscript + native vector PDF figures
├── requirements/   # locked and historical environments
├── tools/          # evidence checks
├── data/           # dataset placement instructions (raw data not redistributed)
└── run_full_pipeline.py
```

## Why the protocols are separated

The canonical paper experiment used the original numerical logistic-regression protocol and selected
`C=30`, `class_weight=None` from training-only five-fold CV. The advanced extension holds that classifier
choice fixed while using a tighter `tol=1e-6` for numerical convergence. The final refinement independently
reruns the 18-combination LR grid under `tol=1e-6` and uses its selected configuration only for the
redundancy/noise-refinement study. This avoids silently overwriting the original paper metrics.

## Dataset

The repository does not redistribute the raw Power Transformers FDD and RUL dataset. See `data/README.md`.
Place the original ZIP at, for example:

```text
data/raw/power_transformers_fdd_and_rul.zip
```

## Recommended environment

Python 3.12 is recommended. The successful advanced/refinement run used Python 3.12.13.

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements/reproducible.txt
```

Exact paper reproduction expects **scikit-learn 1.9.0**.

## One-command full computational reproduction

```bash
python run_full_pipeline.py \
  --archive data/raw/power_transformers_fdd_and_rul.zip \
  --runs-dir runs \
  --bootstrap 5000 \
  --robustness-repeats 20
```

This executes, in order:

1. canonical feature extraction/model selection/four-horizon evaluation/endpoint SHAP;
2. dense horizon + adaptive stopping + formal interaction + advanced uncertainty/robustness/XAI analyses;
3. final convergence-stable redundancy/noise refinement.

No GPU is required.

## Reproducing only one layer

See the README inside `canonical/`, `advanced/`, or `refinement/`.

## Checked-in evidence

The repository includes the machine-readable result files used in the manuscript, including canonical
per-case prediction files required for the formal interaction contrast. To check headline values:

```bash
python tools/verify_paper_evidence.py
```

## Paper and figures

The current LaTeX manuscript is under `paper/`. The repository uses **native vector PDF figures**.
Regenerate them from evidence with:

```bash
python paper/generate_vector_figures.py
```

Then compile with IEEEtran using the commands in `paper/README.md`.

## Reproducibility principles

- Supplied 2,100/900 train-test split is preserved.
- Prefixes are truncated before feature extraction; no future observations enter shorter horizons.
- Model/policy/filter selection is confined to training or training OOF data as documented by each protocol.
- Macro-F1 is the primary metric because the benchmark is strongly imbalanced.
- Paired bootstrap comparisons resample identical held-out cases.
- Sequential conformal reuse is explicitly exploratory; no anytime-valid claim is made.
- Robustness tests are controlled synthetic stress tests, not field validation.
- SHAP is model attribution, not physical causality.

## License

Code is released under the MIT License. The dataset retains its own license and terms of use.
