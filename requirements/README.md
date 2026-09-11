# Environments

For paper reproduction, use `reproducible.txt`. It matches the successful advanced/refinement run:
Python 3.12.13, NumPy 2.0.2, pandas 2.3.3, SciPy 1.16.3, scikit-learn 1.9.0,
joblib 1.5.3, Matplotlib 3.10.0, and SHAP 0.51.0.

The canonical code itself keeps its original solver settings (default LR tolerance, i.e. the
canonical numerical protocol). The advanced code fixes C=30 and class_weight=None but explicitly
uses `tol=1e-6`. The refinement re-runs the 18-combination LR grid under `tol=1e-6`.

`legacy_canonical_requirements.txt` is preserved only as provenance from the original public-code package;
it is too loosely pinned for exact reproduction.
