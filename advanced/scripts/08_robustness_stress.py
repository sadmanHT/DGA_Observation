#!/usr/bin/env python3
"""Stress-test the clean-trained 75%-history temporal model under imperfect DGA data.

Stressors are injected at the RAW-sequence level before feature extraction:
  1) multiplicative Gaussian measurement noise,
  2) randomly missing timestamps + linear interpolation,
  3) one contiguous missing block + linear interpolation,
  4) gradual multiplicative sensor drift on one randomly chosen gas per case.

The classifier is NEVER retrained on corrupted TEST data. This is a deployment-style
stress test, not field validation. Repeated perturbations quantify Monte-Carlo variation.

Feature extraction is vectorized across the 900 test histories so 20+ repeats are practical
on CPU. The vectorized extractor implements the exact formulas in 01_build_features.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from advanced_common import (
    GASES, RANDOM_STATE, class_metrics, classification_metrics,
    feature_groups_from_columns, fixed_lr, load_table, save_environment,
)

EPS = 1e-12


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("base_builder", path)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


def batch_extract(raw: np.ndarray, feature_cols: list[str]) -> np.ndarray:
    """Exact feature formulas, vectorized across cases. raw=(cases,time,4)."""
    x = np.asarray(raw, dtype=float)
    if x.ndim != 3 or x.shape[2] != 4 or x.shape[1] < 2:
        raise ValueError(f"Expected (cases,time,4), got {x.shape}")
    ncase, nt, _ = x.shape
    d = np.diff(x, axis=1)
    q = np.percentile(x, [25, 50, 75], axis=1)  # (3,cases,gases)
    mean = x.mean(axis=1)
    std = x.std(axis=1, ddof=1)
    xmin = x.min(axis=1)
    xmax = x.max(axis=1)
    first = x[:, 0, :]
    last = x[:, -1, :]
    delta = last - first

    t = np.arange(nt, dtype=float)
    tc = t - t.mean()
    denom_t = float(np.dot(tc, tc))
    xc = x - mean[:, None, :]
    slope = np.einsum("t,ntg->ng", tc, xc) / denom_t
    fitted = mean[:, None, :] + slope[:, None, :] * tc[None, :, None]
    ss_res = np.sum((x - fitted) ** 2, axis=1)
    ss_tot = np.sum(xc ** 2, axis=1)
    r2 = np.where(ss_tot > EPS, 1.0 - ss_res / ss_tot, 0.0)

    edge = max(5, int(round(0.20 * nt)))
    early = x[:, :edge, :].mean(axis=1)
    late = x[:, -edge:, :].mean(axis=1)

    vals: dict[str, np.ndarray] = {}
    for j, gas in enumerate(GASES):
        vals[f"{gas}_first"] = first[:, j]
        vals[f"{gas}_last"] = last[:, j]
        vals[f"{gas}_mean"] = mean[:, j]
        vals[f"{gas}_std"] = std[:, j]
        vals[f"{gas}_min"] = xmin[:, j]
        vals[f"{gas}_max"] = xmax[:, j]
        vals[f"{gas}_median"] = q[1, :, j]
        vals[f"{gas}_q25"] = q[0, :, j]
        vals[f"{gas}_q75"] = q[2, :, j]
        vals[f"{gas}_delta"] = delta[:, j]
        vals[f"{gas}_slope"] = slope[:, j]
        vals[f"{gas}_trend_r2"] = r2[:, j]
        vals[f"{gas}_mean_diff"] = d[:, :, j].mean(axis=1)
        vals[f"{gas}_std_diff"] = d[:, :, j].std(axis=1, ddof=1) if d.shape[1] > 1 else np.zeros(ncase)
        vals[f"{gas}_max_abs_diff"] = np.abs(d[:, :, j]).max(axis=1)
        vals[f"{gas}_early_mean"] = early[:, j]
        vals[f"{gas}_late_mean"] = late[:, j]
        vals[f"{gas}_late_minus_early"] = late[:, j] - early[:, j]
        vals[f"{gas}_positive_diff_fraction"] = (d[:, :, j] > 0).mean(axis=1)

    for a in range(4):
        for b in range(a + 1, 4):
            xa = x[:, :, a]
            xb = x[:, :, b]
            ac = xa - xa.mean(axis=1, keepdims=True)
            bc = xb - xb.mean(axis=1, keepdims=True)
            den = np.sqrt(np.sum(ac * ac, axis=1) * np.sum(bc * bc, axis=1))
            corr = np.divide(np.sum(ac * bc, axis=1), den, out=np.zeros(ncase), where=den > EPS)
            vals[f"corr_{GASES[a]}_{GASES[b]}"] = corr

    missing = [c for c in feature_cols if c not in vals]
    if missing:
        raise KeyError(f"Vectorized extractor missing feature columns: {missing}")
    out = np.column_stack([vals[c] for c in feature_cols])
    if not np.isfinite(out).all():
        raise ValueError("Non-finite vectorized features")
    return out


def interpolate_masked_timestamps(x: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Linear interpolation for per-case timestamp masks; endpoints must remain observed."""
    out = x.copy()
    t = np.arange(x.shape[1])
    for i in range(x.shape[0]):
        missing = masks[i]
        if not missing.any():
            continue
        keep = ~missing
        for g in range(x.shape[2]):
            out[i, missing, g] = np.interp(t[missing], t[keep], out[i, keep, g])
    return out


def corrupt_batch(raw: np.ndarray, kind: str, severity: float, rng: np.random.Generator) -> np.ndarray:
    x = raw.copy()
    ncase, nt, ngas = x.shape

    if kind == "relative_gaussian_noise":
        x *= 1.0 + rng.normal(0.0, severity, size=x.shape)
        return np.maximum(x, 0.0)

    if kind == "random_missing_timestamps":
        m = max(1, int(round(nt * severity)))
        masks = np.zeros((ncase, nt), dtype=bool)
        allowed = np.arange(1, nt - 1)
        for i in range(ncase):
            idx = rng.choice(allowed, min(m, len(allowed)), replace=False)
            masks[i, idx] = True
        return interpolate_masked_timestamps(x, masks)

    if kind == "contiguous_gap":
        width = max(1, min(int(round(nt * severity)), nt - 2))
        masks = np.zeros((ncase, nt), dtype=bool)
        starts = rng.integers(1, nt - width, size=ncase)
        for i, start in enumerate(starts):
            masks[i, start : start + width] = True
        return interpolate_masked_timestamps(x, masks)

    if kind == "linear_sensor_drift":
        gases = rng.integers(0, ngas, size=ncase)
        signs = rng.choice(np.array([-1.0, 1.0]), size=ncase)
        ramp = np.linspace(0.0, 1.0, nt)
        for i in range(ncase):
            factor = 1.0 + signs[i] * severity * ramp
            x[i, :, gases[i]] *= factor
        return np.maximum(x, 0.0)

    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--base-builder", type=Path, default=Path(__file__).with_name("01_build_features.py"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_environment(args.output_dir / "environment.json")

    base = load_base(args.base_builder)
    cols_all = pd.read_csv(args.features_dir / "features_train_h75.csv", nrows=1).columns.tolist()
    full = feature_groups_from_columns(cols_all)["full"]
    Xtr, ytr, _, _ = load_table(args.features_dir / "features_train_h75.csv", full)
    model = clone(fixed_lr()).fit(Xtr, ytr)

    tmp = Path(tempfile.mkdtemp(prefix="robust_dga_"))
    try:
        with zipfile.ZipFile(args.archive, "r") as z:
            z.extractall(tmp)
        root = base.locate_dataset_root(tmp)
        labels = base.read_labels(root / "labels_fdd_test.csv")
        raw = np.stack([base.read_series(root / "data_test" / fid)[:315, :] for fid, _ in labels], axis=0)
        y = np.array([lab for _, lab in labels], dtype=int)

        # Validate vectorized feature route against the canonical scalar extractor on a sample.
        fast_sample = batch_extract(raw[:12], full)
        slow_sample = np.array(
            [[base.extract_features(a)[c] for c in full] for a in raw[:12]], dtype=float
        )
        max_abs = float(np.max(np.abs(fast_sample - slow_sample)))
        if not np.allclose(fast_sample, slow_sample, rtol=1e-10, atol=1e-10):
            raise RuntimeError(f"Vectorized extractor disagrees with canonical extractor; max abs={max_abs}")

        cleanX = batch_extract(raw, full)
        clean_pred = model.predict(cleanX)
        clean_m = classification_metrics(y, clean_pred)

        scenarios = {
            "relative_gaussian_noise": [0.02, 0.05, 0.10],
            "random_missing_timestamps": [0.05, 0.10, 0.20],
            "contiguous_gap": [0.05, 0.10, 0.20],
            "linear_sensor_drift": [0.05, 0.10, 0.20],
        }
        rows, crows = [], []
        for kind, severities in scenarios.items():
            for sev in severities:
                for rep in range(args.repeats):
                    seed = RANDOM_STATE + 10000 * rep + int(sev * 1000) + sum(map(ord, kind))
                    rng = np.random.default_rng(seed)
                    t0 = time.perf_counter()
                    damaged = corrupt_batch(raw, kind, sev, rng)
                    X = batch_extract(damaged, full)
                    pred = model.predict(X)
                    m = classification_metrics(y, pred)
                    rows.append({
                        "scenario": kind,
                        "severity": sev,
                        "repeat": rep,
                        "seed": seed,
                        **m,
                        "macro_f1_change_vs_clean": m["macro_f1"] - clean_m["macro_f1"],
                        "seconds": time.perf_counter() - t0,
                    })
                    for cr in class_metrics(y, pred):
                        crows.append({"scenario": kind, "severity": sev, "repeat": rep, **cr})
                    print(kind, sev, rep, m["macro_f1"])

        reps = pd.DataFrame(rows)
        reps.to_csv(args.output_dir / "robustness_replicates.csv", index=False)
        pd.DataFrame(crows).to_csv(args.output_dir / "robustness_classwise_replicates.csv", index=False)
        agg = reps.groupby(["scenario", "severity"]).agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_min=("macro_f1", "min"),
            macro_f1_max=("macro_f1", "max"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            accuracy_mean=("accuracy", "mean"),
            macro_f1_change_mean=("macro_f1_change_vs_clean", "mean"),
            seconds_mean=("seconds", "mean"),
        ).reset_index()
        agg.to_csv(args.output_dir / "robustness_summary.csv", index=False)
        (args.output_dir / "robustness_clean_reference.json").write_text(
            json.dumps({
                **clean_m,
                "vectorized_feature_validation_max_abs_difference": max_abs,
                "interpretation": "Clean-trained 75% model evaluated under controlled test-time corruption; not field validation.",
            }, indent=2),
            encoding="utf-8",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
