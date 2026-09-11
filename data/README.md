# Dataset

The raw dataset is **not redistributed** in this repository.

Use the public **Power Transformers FDD and RUL** dataset referenced by the paper.
The scripts expect the original archive containing:

- `labels_fdd_train.csv`
- `labels_fdd_test.csv`
- `data_train/`
- `data_test/`

Recommended local path:

```text
data/raw/power_transformers_fdd_and_rul.zip
```

`data/raw/` is ignored by Git.

The paper preserves the supplied split exactly: 2,100 training histories and 900 test histories.
Each history contains 420 observations of H2, CO, C2H4, and C2H2.
