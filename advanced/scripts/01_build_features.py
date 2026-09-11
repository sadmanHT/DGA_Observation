#!/usr/bin/env python3
"""Build leakage-safe transformer-level features from the Power Transformers FDD/RUL dataset.

Input: Kaggle archive.zip containing data_train/, data_test/, labels_fdd_train.csv,
       labels_fdd_test.csv.
Output: Transformer-level feature tables for 25%, 50%, 75%, and 100% observation histories.

Only the prefix of each 420-point time series is used for a given observation horizon.
No information from later measurements leaks into shorter-horizon feature tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

GASES = ("H2", "CO", "C2H4", "C2H2")
HORIZONS = {"25": 0.25, "50": 0.50, "75": 0.75, "100": 1.00}
LABEL_MAP = {
    1: "Normal mode",
    2: "Partial discharge",
    3: "Low energy discharge",
    4: "Low-temperature overheating",
}
EPS = 1e-12

STAT_NAMES = ("mean", "std", "min", "max", "median", "q25", "q75")
TEMPORAL_NAMES = (
    "first", "last", "delta", "slope", "trend_r2",
    "mean_diff", "std_diff", "max_abs_diff",
    "early_mean", "late_mean", "late_minus_early", "positive_diff_fraction",
)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < EPS or sy < EPS:
        return 0.0
    val = float(np.corrcoef(x, y)[0, 1])
    return val if math.isfinite(val) else 0.0


def trend_features(x: np.ndarray) -> tuple[float, float]:
    n = len(x)
    t = np.arange(n, dtype=float)
    t_centered = t - t.mean()
    x_centered = x - x.mean()
    denom = float(np.dot(t_centered, t_centered))
    slope = float(np.dot(t_centered, x_centered) / denom) if denom > 0 else 0.0
    fitted = x.mean() + slope * t_centered
    ss_res = float(np.sum((x - fitted) ** 2))
    ss_tot = float(np.sum((x - x.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > EPS else 0.0
    return slope, r2


def one_gas_features(x: np.ndarray, gas: str) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError(f"Expected 1D time series of length >=2 for {gas}")

    diffs = np.diff(x)
    slope, r2 = trend_features(x)
    q25, median, q75 = np.percentile(x, [25, 50, 75])
    edge = max(5, int(round(0.20 * len(x))))

    values = {
        "first": x[0],
        "last": x[-1],
        "mean": np.mean(x),
        "std": np.std(x, ddof=1),
        "min": np.min(x),
        "max": np.max(x),
        "median": median,
        "q25": q25,
        "q75": q75,
        "delta": x[-1] - x[0],
        "slope": slope,
        "trend_r2": r2,
        "mean_diff": np.mean(diffs),
        "std_diff": np.std(diffs, ddof=1) if len(diffs) > 1 else 0.0,
        "max_abs_diff": np.max(np.abs(diffs)),
        "early_mean": np.mean(x[:edge]),
        "late_mean": np.mean(x[-edge:]),
        "late_minus_early": np.mean(x[-edge:]) - np.mean(x[:edge]),
        "positive_diff_fraction": np.mean(diffs > 0),
    }
    return {f"{gas}_{k}": float(v) for k, v in values.items()}


def extract_features(matrix: np.ndarray) -> dict[str, float]:
    if matrix.ndim != 2 or matrix.shape[1] != 4:
        raise ValueError(f"Expected shape (n, 4), got {matrix.shape}")
    feats: dict[str, float] = {}
    for j, gas in enumerate(GASES):
        feats.update(one_gas_features(matrix[:, j], gas))

    for i in range(len(GASES)):
        for j in range(i + 1, len(GASES)):
            g1, g2 = GASES[i], GASES[j]
            feats[f"corr_{g1}_{g2}"] = safe_corr(matrix[:, i], matrix[:, j])
    return feats


def read_labels(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not {"id", "category"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected label columns in {path}: {reader.fieldnames}")
        for row in reader:
            rows.append((row["id"], int(row["category"])))
    return rows


def read_series(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig") as f:
        header = f.readline().strip().split(",")
    if tuple(header) != GASES:
        raise ValueError(f"Unexpected columns in {path.name}: {header}; expected {GASES}")
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=float)
    if arr.shape != (420, 4):
        raise ValueError(f"Unexpected shape in {path.name}: {arr.shape}; expected (420, 4)")
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values found in {path.name}")
    return arr


def locate_dataset_root(root: Path) -> Path:
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]
    for p in candidates:
        if (p / "data_train").is_dir() and (p / "data_test").is_dir() \
           and (p / "labels_fdd_train.csv").exists() and (p / "labels_fdd_test.csv").exists():
            return p
    raise FileNotFoundError("Could not find dataset root with data_train/data_test and FDD label files")


def build_split(dataset_root: Path, split: str, output_dir: Path) -> dict:
    labels_path = dataset_root / f"labels_fdd_{split}.csv"
    data_dir = dataset_root / f"data_{split}"
    labels = read_labels(labels_path)
    expected_n = 2100 if split == "train" else 900
    if len(labels) != expected_n:
        raise ValueError(f"{split}: expected {expected_n} labels, got {len(labels)}")

    cache: dict[str, np.ndarray] = {}
    for idx, (file_id, _) in enumerate(labels, start=1):
        cache[file_id] = read_series(data_dir / file_id)
        if idx % 500 == 0:
            print(f"  loaded {idx}/{len(labels)} {split} series")

    summary = {"n_cases": len(labels), "class_counts": {}}
    for c in LABEL_MAP:
        summary["class_counts"][str(c)] = sum(label == c for _, label in labels)

    for horizon_name, fraction in HORIZONS.items():
        steps = int(round(420 * fraction))
        days = steps * 0.5
        records: list[dict] = []
        for file_id, label in labels:
            feats = extract_features(cache[file_id][:steps, :])
            record = {
                "id": file_id,
                "split": split,
                "label": label,
                "label_name": LABEL_MAP[label],
                "observation_fraction": fraction,
                "observation_steps": steps,
                "observation_days": days,
            }
            record.update(feats)
            records.append(record)

        out_path = output_dir / f"features_{split}_h{horizon_name}.csv"
        fieldnames = list(records[0].keys())
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"  wrote {out_path.name}: {len(records)} rows, {len(fieldnames)} columns")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True, help="Path to Kaggle archive.zip")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for feature CSV files")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="dga_extract_"))
    try:
        print(f"Extracting {args.archive} ...")
        with zipfile.ZipFile(args.archive, "r") as zf:
            zf.extractall(tmpdir)
        dataset_root = locate_dataset_root(tmpdir)
        print(f"Dataset root: {dataset_root}")

        summaries = {}
        for split in ("train", "test"):
            print(f"Building {split} feature tables ...")
            summaries[split] = build_split(dataset_root, split, args.output_dir)

        feature_names = []
        for gas in GASES:
            feature_names.extend([f"{gas}_{n}" for n in ("first", "last") + STAT_NAMES + tuple(n for n in TEMPORAL_NAMES if n not in ("first", "last"))])
        corr_names = [f"corr_{GASES[i]}_{GASES[j]}" for i in range(len(GASES)) for j in range(i + 1, len(GASES))]

        manifest = {
            "dataset": "Power Transformers FDD and RUL",
            "task": "4-class fault detection and diagnosis (FDD)",
            "gases": list(GASES),
            "time_points_full": 420,
            "sampling_interval_hours": 12,
            "label_map": {str(k): v for k, v in LABEL_MAP.items()},
            "horizons": {
                k: {
                    "fraction": v,
                    "steps": int(round(420*v)),
                    "days": int(round(420*v))*0.5,
                }
                for k, v in HORIZONS.items()
            },
            "splits": summaries,
            "feature_groups": {
                "snapshot": [f"{g}_last" for g in GASES],
                "statistical": [f"{g}_{s}" for g in GASES for s in STAT_NAMES],
                "temporal_extra": [f"{g}_{s}" for g in GASES for s in TEMPORAL_NAMES if s not in ("first", "last")],
                "cross_gas_correlation": corr_names,
                "full": feature_names + corr_names,
            },
            "notes": [
                "Shorter horizons use only the prefix of the original 420-point sequence; later measurements are never used.",
                "The supplied train/test partition is preserved exactly.",
                "No SMOTE, scaling, feature selection, or model fitting is performed during feature construction.",
            ],
        }
        with (args.output_dir / "feature_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Verify feature tables contain no NaN/Inf by scanning numeric cells from the full-history files.
        for split in ("train", "test"):
            p = args.output_dir / f"features_{split}_h100.csv"
            with p.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                numeric_cols = [c for c in reader.fieldnames or [] if c not in {"id", "split", "label_name"}]
                for row_idx, row in enumerate(reader, start=2):
                    for c in numeric_cols:
                        val = float(row[c])
                        if not math.isfinite(val):
                            raise ValueError(f"Non-finite feature at {p.name}:{row_idx} column {c}")
        print("Feature build complete and verified.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
