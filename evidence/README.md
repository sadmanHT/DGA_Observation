# Reference results

This directory contains the machine-readable outputs from the successful experiment runs used to write the current manuscript. They are committed so readers can audit reported values and regenerate figures without rerunning the full dataset pipeline.

- `canonical/results/` — original four-horizon tables and endpoint SHAP summaries.
- `canonical/predictions/` — per-case canonical predictions used by paired comparisons and the formal interaction test.
- `advanced/` — dense-horizon, adaptive-stopping, interaction, conformal, SHAP-stability, ablation, and stress-test outputs.
- `refinement/` — convergence-stable grid, redundancy audit, robust-window selection, and clean/noise results.

These files are reference outputs, not substitutes for the executable pipelines in `canonical/`, `advanced/`, and `refinement/`.

Verify key manuscript values with:

```bash
python tools/verify_paper_evidence.py
```
