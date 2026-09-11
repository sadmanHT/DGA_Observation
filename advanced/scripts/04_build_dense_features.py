#!/usr/bin/env python3
"""Build leakage-safe features every 5 percentage points from 10% to 100% history.

This extends the original 25/50/75/100 feature builder without changing any feature
formula.  Each horizon is computed from the raw prefix before feature extraction.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

DENSE_HORIZONS = list(range(10, 101, 5))


def load_base_builder(script_path: Path):
    spec = importlib.util.spec_from_file_location("base_builder", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--base-builder", type=Path, default=Path(__file__).with_name("01_build_features.py"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = load_base_builder(args.base_builder)

    tmpdir = Path(tempfile.mkdtemp(prefix="dga_dense_"))
    try:
        with zipfile.ZipFile(args.archive, "r") as zf:
            zf.extractall(tmpdir)
        root = base.locate_dataset_root(tmpdir)
        split_summary = {}
        for split in ("train", "test"):
            labels = base.read_labels(root / f"labels_fdd_{split}.csv")
            data_dir = root / f"data_{split}"
            cache = {}
            for i, (file_id, _) in enumerate(labels, 1):
                cache[file_id] = base.read_series(data_dir / file_id)
                if i % 500 == 0:
                    print(f"loaded {split}: {i}/{len(labels)}")
            split_summary[split] = {
                "n_cases": len(labels),
                "class_counts": {str(c): sum(y == c for _, y in labels) for c in base.LABEL_MAP},
            }
            for h in DENSE_HORIZONS:
                out = args.output_dir / f"features_{split}_h{h}.csv"
                if out.exists() and out.stat().st_size > 0:
                    print(f"skip existing {out.name}")
                    continue
                frac = h / 100.0
                steps = int(round(420 * frac))
                records = []
                for file_id, label in labels:
                    feats = base.extract_features(cache[file_id][:steps, :])
                    rec = {
                        "id": file_id,
                        "split": split,
                        "label": label,
                        "label_name": base.LABEL_MAP[label],
                        "observation_fraction": frac,
                        "observation_steps": steps,
                        "observation_days": steps * 0.5,
                    }
                    rec.update(feats)
                    records.append(rec)
                with out.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(records[0]))
                    w.writeheader(); w.writerows(records)
                print(f"wrote {out.name}: {len(records)} rows")

        sample = next(csv.DictReader((args.output_dir / "features_train_h100.csv").open(encoding="utf-8")))
        metadata = {"id", "split", "label", "label_name", "observation_fraction", "observation_steps", "observation_days"}
        feature_cols = [c for c in sample if c not in metadata]
        manifest = {
            "dataset": "Power Transformers FDD and RUL",
            "dense_horizons_percent": DENSE_HORIZONS,
            "sampling_interval_hours": 12,
            "feature_count": len(feature_cols),
            "features": feature_cols,
            "splits": split_summary,
            "notes": [
                "Every horizon is truncated before feature extraction.",
                "Feature formulas are identical to 01_build_features.py.",
                "observation_days is sample-equivalent duration = number_of_samples * 0.5; elapsed first-to-last timestamp span is 0.5 day shorter.",
            ],
        }
        (args.output_dir / "dense_feature_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
