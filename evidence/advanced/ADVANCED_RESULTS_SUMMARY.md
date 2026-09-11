# Advanced DGA strengthening run summary

Environment: Python `3.12.13`, scikit-learn `1.9.0`, NumPy `2.0.2`, pandas `2.3.3`, SHAP `0.51.0`.

## Formal interaction
Difference-of-differences = **0.0761**, paired 95% CI **[0.0278, 0.1263]**.

## Adaptive stopping
Training-OOF selected threshold = **0.95**, persistence = **3** horizons. Test macro-F1 = **0.8865**, endpoint retention = **98.29%**, mean stop = **37.8%**, median stop = **20.0%**, stopped before 100% = **93.6%**.

## Dense representation curve
Largest observed temporal-minus-statistical test macro-F1 = **0.0636** at **60%** history; paired CI **[0.0177, 0.1104]**.

## Feature-family ablation at 75%
Largest gain when added to Statistical-28: **local_dynamics**, Δ macro-F1 **0.0777**, CI **[0.0388, 0.1191]**.

## Robustness stress test
- contiguous_gap, severity 0.05: macro-F1 0.8971 ± 0.0003, Δ vs clean -0.0001.
- contiguous_gap, severity 0.1: macro-F1 0.8980 ± 0.0028, Δ vs clean 0.0008.
- contiguous_gap, severity 0.2: macro-F1 0.8958 ± 0.0035, Δ vs clean -0.0013.
- linear_sensor_drift, severity 0.05: macro-F1 0.8956 ± 0.0073, Δ vs clean -0.0015.
- linear_sensor_drift, severity 0.1: macro-F1 0.8794 ± 0.0090, Δ vs clean -0.0178.
- linear_sensor_drift, severity 0.2: macro-F1 0.8372 ± 0.0169, Δ vs clean -0.0600.
- random_missing_timestamps, severity 0.05: macro-F1 0.8971 ± 0.0000, Δ vs clean 0.0000.
- random_missing_timestamps, severity 0.1: macro-F1 0.8971 ± 0.0000, Δ vs clean 0.0000.
- random_missing_timestamps, severity 0.2: macro-F1 0.8971 ± 0.0000, Δ vs clean 0.0000.
- relative_gaussian_noise, severity 0.02: macro-F1 0.3179 ± 0.0109, Δ vs clean -0.5792.
- relative_gaussian_noise, severity 0.05: macro-F1 0.2436 ± 0.0068, Δ vs clean -0.6535.
- relative_gaussian_noise, severity 0.1: macro-F1 0.2187 ± 0.0098, Δ vs clean -0.6784.

## Fixed-horizon class-conditional conformal diagnostics
- 50%: coverage **90.44%**, singleton rate **83.33%**, mean set size **1.172**.
- 75%: coverage **90.56%**, singleton rate **94.89%**, mean set size **1.014**.
- 100%: coverage **88.78%**, singleton rate **92.00%**, mean set size **0.978**.

## SHAP stability
- 50% vs 75%: Spearman ρ **0.620**, top-10 Jaccard **0.250**.
- 75% vs 100%: Spearman ρ **0.529**, top-10 Jaccard **0.250**.
- 50% vs 100%: Spearman ρ **0.460**, top-10 Jaccard **0.176**.

## Interpretation warning
Only the formal interaction section can use the existing canonical `final_5fold` predictions directly. Dense-horizon, adaptive, robustness, conformal, ablation, and early-SHAP numbers are final only when the suite is run in the locked canonical environment (scikit-learn 1.9.0) and that environment is archived.