# Adaptive Observation-Efficient Power Transformer Fault Diagnosis from Partial DGA Histories

This repository contains the code and reference results for the research project **Adaptive Observation-Efficient Power Transformer Fault Diagnosis from Partial DGA Histories Using Explainable Lightweight Machine Learning**.

The project studies a practical condition-monitoring question: **how much dissolved-gas-analysis (DGA) history is actually needed before a reliable transformer diagnosis can be issued?** Instead of treating the full monitoring record as mandatory, the experiments evaluate diagnosis as evidence accumulates and ask when the system should **stop, continue observing, or defer to engineering review**.

The implementation is intentionally lightweight. The principal classifier is standardized multinomial logistic regression; the research contribution is the joint design of **observation horizon, feature representation, stopping policy, uncertainty, explanation, and robustness**, not a larger neural architecture.

## Main findings

The current manuscript is supported by three deliberately separated experimental layers: the original canonical experiment, the dense/adaptive extension, and the final robustness/refinement study.

| Finding | Result |
| --- | --- |
| 75% partial-history diagnosis | Temporal-82 reaches **0.8971 macro-F1** versus **0.9026** at 100%, retaining **99.39%** of endpoint performance |
| Horizon × representation interaction | Difference-of-differences = **0.0761**, 95% CI **[0.0278, 0.1263]** |
| Dense observation study | Temporal features provide their clearest advantage in the **intermediate-history regime (~55–75%)**, not at every horizon |
| Adaptive stopping | Training-selected policy (`tau=0.95`, persistence `K=3`) reaches **0.8865 macro-F1**, retains **98.29%** of its endpoint reference, and stops at **37.8% history on average** |
| Observation demand | Median stopping horizon = **20%**; **93.6%** of held-out cases stop before full history |
| Fixed-horizon conformal uncertainty | At 75% history, empirical coverage = **90.56%** for a nominal 90% target |
| Representation redundancy | Temporal-82 has numerical rank **70**; a 36-feature statistical-plus-endpoint representation captures much of the 75% benefit |
| Robustness failure and mitigation | At 2% independent multiplicative noise, Temporal-82 falls to **0.2476 macro-F1**; redundancy removal plus causal smoothing raises this to **0.6250**, while clean performance remains near **0.896** |

These results support the central engineering conclusion:

> **Observation duration should be case-specific, and feature representation should be designed together with the decision horizon.**

## Dataset

Experiments use the public **Power Transformers FDD and RUL** dataset by I. Katser:

https://www.kaggle.com/datasets/yuriykatser/power-transformers-fdd-and-rul

The repository does not redistribute the dataset. The benchmark contains:

- **3,000** labeled transformer histories;
- **2,100 training** and **900 test** histories in the supplied split;
- **420** measurements per history at **12-hour** intervals;
- four gases: **H2, CO, C2H4, C2H2**;
- four classes: normal mode, partial discharge, low-energy discharge, and low-temperature overheating.

Place the original archive at:

```text
data/raw/power_transformers_fdd_and_rul.zip
```

See [`data/README.md`](data/README.md) for the expected archive contents.

## Repository structure

```text
.
├── canonical/          # original four-horizon experiment and endpoint SHAP
├── advanced/           # dense horizons, adaptive stopping, interaction, conformal, SHAP stability, stress tests
├── refinement/         # Temporal-70/redundancy analysis and causal smoothing/noise refinement
├── evidence/           # reference machine-readable outputs used by the manuscript
├── notebooks/          # optional exact Kaggle execution notebooks
├── requirements/       # pinned environment for exact reproduction
├── data/               # dataset placement instructions; raw data is ignored by Git
├── tools/              # result verification utility
├── run_full_pipeline.py
├── generate_figures.py # regenerate the six current manuscript figures as vector PDFs
├── CITATION.cff
├── LICENSE
└── README.md
```

The command-line scripts are the primary reproduction path. The notebooks are retained only as optional execution provenance for the successful Kaggle runs.

## Environment

The successful advanced/refinement environment used Python 3.12.13 and scikit-learn 1.9.0. Exact package versions are pinned in `requirements/reproducible.txt`.

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\Activate.ps1     # Windows PowerShell
pip install -r requirements/reproducible.txt
```

The full study is CPU-compatible; no GPU is required.

## Reproduce the complete project

After placing the dataset archive under `data/raw/`, run:

```bash
python run_full_pipeline.py \
  --archive data/raw/power_transformers_fdd_and_rul.zip \
  --runs-dir runs \
  --bootstrap 5000 \
  --robustness-repeats 20
```

This executes the complete computational path used by the manuscript:

1. **Canonical experiment**
   - leakage-safe 25/50/75/100% prefixes;
   - Statistical-28 and Temporal-82 feature extraction;
   - lightweight model comparison and training-only model selection;
   - endpoint evaluation, class-wise metrics, paired bootstrap comparisons;
   - fixed-horizon selective diagnosis and endpoint SHAP.
2. **Advanced extension**
   - dense 10, 15, ..., 100% observation horizons;
   - formal horizon × representation interaction test;
   - training-OOF adaptive stopping;
   - feature-family ablation;
   - class-conditional split conformal prediction;
   - SHAP at 50/75/100% with rank/family stability;
   - missingness, contiguous-gap, drift, and synthetic-noise stress tests.
3. **Final refinement**
   - convergence-stable 18-configuration logistic-regression grid;
   - redundancy-aware Temporal-70 analysis;
   - compact feature-family comparisons;
   - training-only causal smoothing selection;
   - independent held-out noise evaluation.

Generated outputs are written under `runs/` and are ignored by Git.

## Reproduce one experiment layer

### Canonical

```bash
python canonical/run_canonical.py \
  --archive data/raw/power_transformers_fdd_and_rul.zip \
  --run-dir runs/canonical_final_5fold \
  --tuning-iterations 20 \
  --bootstrap 5000
```

### Advanced

```bash
python advanced/run_advanced.py \
  --archive data/raw/power_transformers_fdd_and_rul.zip \
  --canonical-final runs/canonical_final_5fold \
  --work-dir runs/advanced \
  --robustness-repeats 20
```

### Refinement

```bash
python refinement/run_refinement.py \
  --archive data/raw/power_transformers_fdd_and_rul.zip \
  --canonical-final runs/canonical_final_5fold \
  --work-dir runs/refinement \
  --bootstrap 5000 \
  --noise-test-repeats 20
```

Each experiment directory contains a short README explaining its protocol and why it is kept separate from the other layers.

## Reference results and verification

`evidence/` contains the machine-readable outputs from the successful runs used to write the manuscript. They are checked in so that reported values can be audited without repeating every experiment.

To verify the headline values:

```bash
python tools/verify_paper_evidence.py
```

The verifier checks, among other quantities, the formal interaction estimate, adaptive-stopping policy, convergence-stable refinement model, selected causal smoothing window, and 2% noise result.

## Reproduce the manuscript figures

The six figures used in the current manuscript can be regenerated directly from the checked-in reference outputs as **native vector PDFs**:

```bash
python generate_figures.py
```

They are written to `figures/` (ignored by Git). The plotting code uses vector primitives rather than PNG-to-PDF conversion, so text, axes, lines, markers, bars, and confusion-matrix cells remain sharp under zoom.

## Experimental protocol notes

The three experiment layers are intentionally not collapsed into one fitted model:

- **Canonical:** original paper protocol; `C=30`, `class_weight=None`, original logistic-regression tolerance.
- **Advanced:** keeps `C=30`, `class_weight=None`, but uses `tol=1e-6` for convergence stability.
- **Refinement:** reruns the full 18-combination logistic-regression grid under `tol=1e-6`; the selected configuration is used only for the redundancy/noise-refinement study.

This separation prevents later numerical refinements from silently overwriting the original canonical paper results.

Across all layers:

- the supplied 2,100/900 train-test split is preserved;
- shorter histories are truncated **before** feature extraction, preventing future-information leakage;
- model, stopping-policy, and smoothing-window selection use training or training-OOF data only;
- macro-F1 is the primary metric because the benchmark is strongly imbalanced;
- paired comparisons resample the same held-out transformer cases;
- sequential reuse of ordinary split-conformal sets is treated as exploratory, not anytime-valid;
- robustness experiments are controlled synthetic stress tests, not field sensor validation;
- SHAP is used for model attribution, not as proof of physical causality.

## Scope and limitations

This repository reproduces the reported benchmark study; it does not establish deployment performance on another transformer fleet. External validation remains the main unresolved step. The synthetic high-frequency-noise experiment is deliberately a stress test and should not be interpreted as a calibrated model of field DGA sensor error.

## Citation

This repository accompanies the manuscript:

**Taufikur Rahman Fuad, _Adaptive Observation-Efficient Power Transformer Fault Diagnosis from Partial DGA Histories Using Explainable Lightweight Machine Learning_, 2026.**

Until final publication metadata is available, the repository can be cited as:

```bibtex
@misc{fuad2026dgaobservation,
  author       = {Taufikur Rahman Fuad},
  title        = {Adaptive Observation-Efficient Power Transformer Fault Diagnosis from Partial DGA Histories Using Explainable Lightweight Machine Learning},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/sadmanHT/DGA_Observation}
}
```

## License

Code in this repository is released under the [MIT License](LICENSE). The external dataset remains subject to its own license and terms of use.
