# Changes from the canonical paper

The updated manuscript preserves the original IEEE conference format, dataset description, canonical 5-fold model-selection experiment, fixed 25/50/75/100% results, references, and core engineering background.

Major additions supported by later experiment archives:

1. Formal paired difference-of-differences test for the 75%-to-100% horizon x representation interaction.
2. Dense 10--100% observation-horizon characterization at 5%-point spacing.
3. Training-OOF-selected adaptive stopping with threshold 0.95 and persistence 3.
4. Fixed-horizon class-conditional split-conformal prediction.
5. SHAP stability analysis across 50%, 75%, and 100% history.
6. Redundancy audit showing 82 nominal columns but numerical rank 70.
7. Nonredundant Temporal-70 and feature-family ablation.
8. Controlled missingness, contiguous-gap, drift, and noise stress tests.
9. Training-only causal-smoothing robustness refinement selected from windows {1,3,5,7,11}.
10. Explicit separation of canonical, advanced, and refinement numerical protocols to avoid mixing incompatible scores.

The paper deliberately reports negative results and trade-offs: temporal features are worse at the shortest horizon, some feature families do not help, explanations shift with horizon, drift degrades performance, and the original representation fails severely under the tested high-frequency noise model.
