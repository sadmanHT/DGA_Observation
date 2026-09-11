# Advanced extension

This is the convergence-stabilized extension used by the updated paper.

It implements:
- dense 10--100% histories at 5-point spacing;
- Statistical-28 vs Temporal-82 dense curves;
- training-OOF adaptive stopping;
- canonical paired difference-of-differences interaction;
- feature-family ablation;
- missing/gap/drift/noise stress tests;
- class-conditional split conformal prediction;
- SHAP at 50/75/100% plus stability analysis.

The classifier is fixed at C=30, class_weight=None, with `tol=1e-6` for these extension analyses.
No GPU is required.

```bash
python advanced/run_advanced.py   --archive data/raw/power_transformers_fdd_and_rul.zip   --canonical-final runs/canonical_final_5fold   --work-dir runs/advanced   --robustness-repeats 20
```
