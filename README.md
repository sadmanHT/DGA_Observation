# Adaptive Observation-Efficient Power Transformer Fault Diagnosis from Partial DGA Histories Using Explainable Lightweight Machine Learning

## Setup

Use Python 3.12 and run these commands from the repository directory. No GPU is required.

```bash
python -m venv .venv
```

Activate the environment on Linux/macOS:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements/reproducible.txt
```

Download the [dataset](https://www.kaggle.com/datasets/yuriykatser/power-transformers-fdd-and-rul), create `data/raw/`, and save the archive as `data/raw/power_transformers_fdd_and_rul.zip`. The dataset is not included.

The archive must contain `labels_fdd_train.csv`, `labels_fdd_test.csv`, `data_train/`, and `data_test/`. Label files must have `id` and `category` columns. Each history CSV must contain 420 rows with columns `H2,CO,C2H4,C2H2` in that order. Use the supplied split of 2,100 training and 900 test histories.

## Run all experiments

```bash
python run_full_pipeline.py --archive data/raw/power_transformers_fdd_and_rul.zip --runs-dir runs --bootstrap 5000 --robustness-repeats 20
```

Outputs are saved under `runs/`.

## Run individual stages

Run the canonical stage first:

```bash
python canonical/run_canonical.py --archive data/raw/power_transformers_fdd_and_rul.zip --run-dir runs/canonical_final_5fold --tuning-iterations 20 --bootstrap 5000
```

Then run either or both remaining stages:

```bash
python advanced/run_advanced.py --archive data/raw/power_transformers_fdd_and_rul.zip --canonical-final runs/canonical_final_5fold --work-dir runs/advanced --robustness-repeats 20
python refinement/run_refinement.py --archive data/raw/power_transformers_fdd_and_rul.zip --canonical-final runs/canonical_final_5fold --work-dir runs/refinement --bootstrap 5000 --noise-test-repeats 20
```

## Generate figures or verify bundled results

These commands use the required input files in `evidence/`:

```bash
python generate_figures.py
python tools/verify_paper_evidence.py
```

PDF figures are saved under `figures/`.

## Run on Kaggle

Upload either notebook from `notebooks/` to Kaggle, enable internet access, and select CPU execution. Attach the raw dataset and a completed canonical run containing `features/` and `predictions/`.

In the advanced notebook, set `RAW_SOURCE` and `FINAL_SOURCE` in the first cell to the attached dataset and canonical-run paths. In the refinement notebook, set `RAW_ROOT` and `CANON_ROOT` to their extracted directories. Run all cells in order. Download the result ZIP at the path printed by the last cell.
