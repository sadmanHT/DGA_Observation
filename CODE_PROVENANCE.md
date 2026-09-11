# Code provenance and paper-result map

This repository intentionally retains the code lineage that produced the **current updated paper**.
The paper is not based on one monolithic run; it contains three explicitly separated experimental layers.

## Canonical layer

Exact computational scripts copied from the original public-code package:

- `canonical/scripts/01_build_features.py`
  - raw ZIP parsing;
  - leakage-safe 25/50/75/100% prefix construction;
  - Statistical-28 / Temporal-82 feature tables.
- `canonical/scripts/02_run_5fold_experiments.py`
  - five-family lightweight model benchmark;
  - training-only LR tuning;
  - endpoint metrics;
  - feature ablation;
  - four-horizon representation results;
  - paired bootstrap 75-vs-100 and representation comparisons;
  - class-weight sensitivity;
  - fixed-horizon selective diagnosis;
  - canonical prediction CSVs and fingerprints.
- `canonical/scripts/03_shap_explanations.py`
  - original endpoint SHAP analysis.
- `canonical/run_canonical.py`
  - exact original orchestration.

These scripts support the canonical values retained in the paper, including Temporal-82 macro-F1 0.8971 at 75% and 0.9026 at 100%.

## Advanced layer

Exact scripts copied from the successful convergence-stabilized advanced package:

- `advanced/scripts/04_build_dense_features.py` - 10,15,...,100% leakage-safe prefixes.
- `advanced/scripts/05a_dense_representation_curve.py` - dense Statistical-28 vs Temporal-82 curve and paired intervals.
- `advanced/scripts/05_dense_adaptive_stopping.py` - training-OOF threshold/persistence selection and held-out adaptive evaluation.
- `advanced/scripts/06_formal_interaction.py` - canonical paired difference-of-differences test.
- `advanced/scripts/07_feature_family_ablation.py` - original family ablation extension.
- `advanced/scripts/08_robustness_stress.py` - missingness/gaps/drift/Gaussian-noise stress tests.
- `advanced/scripts/09_conformal_dense.py` - class-conditional split-conformal analysis.
- `advanced/scripts/10_shap_early_stability.py` - 50/75/100% SHAP and stability analysis.
- `advanced/scripts/11_summarize_advanced_results.py` - result summary.
- `advanced/scripts/advanced_common.py` - fixed advanced LR configuration and shared bootstrap/metric utilities.
- `advanced/scripts/01_build_features.py` - exact feature formulas imported by dense/stress scripts.
- `advanced/run_advanced.py` - successful suite orchestration.

The exact Kaggle execution workflow is also retained in `notebooks/DGA_Advanced_Thesis_Kaggle_FULL_NoVenv_v3.ipynb`.

## Final refinement layer

The exact successful Kaggle notebook is retained as:

- `notebooks/DGA_Final_Compact_Robustness_Refinement_Kaggle.ipynb`

For normal command-line reproduction, the same computation is exposed as:

- `refinement/run_refinement.py`

It performs:

1. the exact 18-combination LR grid under `tol=1e-6`;
2. redundancy-aware Statistical-28 / Temporal-70 / Temporal-82 analysis;
3. training-only OOF selection of causal smoothing from {1,3,5,7,11};
4. independent 2/5/10% held-out Gaussian-noise evaluation with 20 paired test perturbation repeats.

## Paper figures

- `paper/generate_vector_figures.py` regenerates the six figures used by the current manuscript directly from checked-in evidence as native vector PDFs.

## Checked-in evidence

- `evidence/canonical/` - canonical result tables plus per-case prediction files.
- `evidence/advanced/` - complete successful advanced result tree and environment proof.
- `evidence/refinement/` - complete final refinement outputs.

`tools/verify_paper_evidence.py` asserts key headline values against these files.

## Intentionally excluded from the final-result code path

Preliminary scikit-learn 1.8 runs, smoke-test outputs, backup files, and exploratory scripts that did **not** supply numbers to the current manuscript are not part of the main reproduction path. This avoids presenting preliminary diagnostics as final evidence.
