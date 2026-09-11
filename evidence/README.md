# Checked-in paper evidence

These files are machine-readable outputs from the successful runs used to write the current paper.
They are included so paper numbers can be audited without rerunning the full dataset pipeline.

- `canonical/results/` - original four-horizon 5-fold paper tables and SHAP summaries.
- `canonical/predictions/` - per-case canonical prediction files used for paired comparisons/formal interaction.
- `advanced/` - dense, adaptive, interaction, conformal, SHAP-stability, ablation, and stress-test outputs.
- `refinement/` - convergence-stable grid, redundancy audit, robust-window selection, clean/noise results.

Run `python tools/verify_paper_evidence.py` to check key headline values.
