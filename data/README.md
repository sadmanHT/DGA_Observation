# Dataset setup

The raw dataset is **not redistributed** in this repository.

Use the public **Power Transformers FDD and RUL** dataset by I. Katser:

https://www.kaggle.com/datasets/yuriykatser/power-transformers-fdd-and-rul

Download the original archive and place it at:

```text
data/raw/power_transformers_fdd_and_rul.zip
```

`data/raw/` is ignored by Git.

The archive used by the scripts must contain the supplied train/test labels and history folders, including:

```text
labels_fdd_train.csv
labels_fdd_test.csv
data_train/
data_test/
```

The benchmark provides 3,000 histories in the supplied split:

- 2,100 training histories;
- 900 test histories;
- 420 observations per history;
- 12-hour sampling;
- gas channels: H2, CO, C2H4, C2H2.

The code preserves the supplied split exactly. Shorter observation horizons are constructed by truncating each raw history before feature extraction.
