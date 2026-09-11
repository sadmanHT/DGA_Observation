# Final compact refinement

This is the final experiment layer used by the paper.

It performs only:
1. the exact 18-combination LR training-CV grid at `tol=1e-6`;
2. redundancy-aware Statistical-28 / Temporal-70 / Temporal-82 ablation;
3. training-OOF selection of a causal moving-average window from {1,3,5,7,11};
4. independent held-out noise evaluation at 2%, 5%, and 10%, with 20 paired perturbation repeats.

The exact Kaggle notebook that produced the checked-in refinement evidence is retained under `notebooks/`.
The standalone script is a CLI version of the same computation.

```bash
python refinement/run_refinement.py   --archive data/raw/power_transformers_fdd_and_rul.zip   --canonical-final runs/canonical_final_5fold   --work-dir runs/refinement   --bootstrap 5000   --noise-test-repeats 20
```
