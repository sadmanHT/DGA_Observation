# Canonical experiment

This directory preserves the computational code used for the original four-horizon paper study.
The important property is that its logistic-regression numerical settings are **not retroactively
changed** by the later refinement.

Run:

```bash
python canonical/run_canonical.py   --archive data/raw/power_transformers_fdd_and_rul.zip   --run-dir runs/canonical_final_5fold   --tuning-iterations 20   --bootstrap 5000
```

The LR search space has 9 C values x 2 class-weight settings = 18 unique combinations. The historical
runner asks RandomizedSearchCV for 20 iterations; sklearn therefore evaluates the finite space rather
than creating additional configurations.
