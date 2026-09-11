# Final Compact Refinement Summary

## Convergence-stable model selection
- C: 100.0
- class_weight: balanced
- training-CV macro-F1: 0.915869

## Noise-robust representation selection
- selected causal moving-average window: 11 samples
- clean-retention constraint: 98% of unsmoothed training OOF

## Clean held-out TEST
- Temporal-82 original: macro-F1 = 0.894567
- Temporal-70 unsmoothed: macro-F1 = 0.890695
- Temporal-70 robust w=11: macro-F1 = 0.896385

## Methodological guardrails
- TEST did not select C, class weighting, feature families, or smoothing window.
- Synthetic Gaussian noise is a stress model, not a measured sensor-error distribution.
- Use training-CV evidence before making causal/mechanistic claims from feature-family ablations.