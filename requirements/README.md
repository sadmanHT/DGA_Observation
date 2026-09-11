# Reproduction environment

Use `reproducible.txt` for the project environment used by the successful advanced/refinement experiments.

It pins:

- Python target: 3.12.x (successful run: 3.12.13)
- NumPy 2.0.2
- pandas 2.3.3
- SciPy 1.16.3
- scikit-learn 1.9.0
- joblib 1.5.3
- Matplotlib 3.10.0
- SHAP 0.51.0

The canonical experiment keeps its original logistic-regression numerical settings; the advanced and refinement layers explicitly use `tol=1e-6` where documented. A single pinned package environment is sufficient to execute all three code paths in this repository.
