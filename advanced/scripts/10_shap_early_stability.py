#!/usr/bin/env python3
"""SHAP at the actual partial-history model plus attribution stability analysis.

Runs 50%, 75%, and 100% temporal-82 models. All 900 TEST cases are explained. The
same TRAIN background row indices are used at every horizon to avoid conflating
horizon effects with different SHAP background samples. Outputs include:
  * global feature importance on all 900 cases,
  * true-class-conditioned importance for the output corresponding to that class,
  * grouped importance by temporal family and by gas,
  * rank stability across horizons.

SHAP is descriptive attribution, not causal transformer physics. The feature set contains
correlated/algebraically related descriptors, so family-level patterns are emphasized over
individual rank positions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.base import clone

from advanced_common import (
    GASES, LABELS, LABEL_NAMES, RANDOM_STATE, feature_groups_from_columns,
    fixed_lr, load_table, save_environment,
)

HORIZONS = [50, 75, 100]


def normalize(values, n, f, k):
    if isinstance(values, list):
        arr = np.stack(values, axis=-1)
    else:
        arr = np.asarray(values)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    elif arr.ndim == 3 and arr.shape == (k, n, f):
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 3 and arr.shape == (n, k, f):
        arr = np.transpose(arr, (0, 2, 1))
    if arr.shape[0] != n or arr.shape[1] != f:
        raise ValueError(arr.shape)
    return arr


def family_for_feature(f):
    if f.startswith("corr_"):
        return "cross_gas_correlation"
    suffix = f.split("_", 1)[1]
    if suffix in {"mean", "std", "min", "max", "median", "q25", "q75"}:
        return "statistical"
    if suffix in {"first", "last", "delta"}:
        return "endpoint_change"
    if suffix in {"slope", "trend_r2"}:
        return "trend"
    if suffix in {"mean_diff", "std_diff", "max_abs_diff", "positive_diff_fraction"}:
        return "local_dynamics"
    if suffix in {"early_mean", "late_mean", "late_minus_early"}:
        return "phase_shift"
    return "other"


def gas_for_feature(f):
    if f.startswith("corr_"):
        return "cross-gas"
    return next((g for g in GASES if f.startswith(g + "_")), "other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--background", type=int, default=200)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_environment(args.output_dir / "environment.json")

    cols_all = pd.read_csv(args.features_dir / "features_train_h100.csv", nrows=1).columns.tolist()
    full = feature_groups_from_columns(cols_all)["full"]

    # All horizons preserve TRAIN row order. Use exactly the same background identities.
    Xtr100, ytr100, ids_tr100, _ = load_table(args.features_dir / "features_train_h100.csv", full)
    rng = np.random.default_rng(RANDOM_STATE)
    bgidx = rng.choice(len(Xtr100), min(args.background, len(Xtr100)), replace=False)
    pd.DataFrame({"row_index": bgidx, "id": ids_tr100[bgidx], "label": ytr100[bgidx]}).to_csv(
        args.output_dir / "shap_background_rows.csv", index=False
    )

    global_by_h = {}
    allrows, classrows, grouprows = [], [], []

    for h in HORIZONS:
        Xtr, ytr, ids_tr, _ = load_table(args.features_dir / f"features_train_h{h}.csv", full)
        Xte, yte, ids_te, _ = load_table(args.features_dir / f"features_test_h{h}.csv", full)
        if not np.array_equal(ids_tr100, ids_tr):
            raise RuntimeError("TRAIN row order mismatch across horizons")

        model = clone(fixed_lr()).fit(Xtr, ytr)
        scaler = model.named_steps["scale"]
        linear = model.named_steps["model"]
        bg = scaler.transform(Xtr[bgidx])
        Xs = scaler.transform(Xte)
        expl = shap.LinearExplainer(linear, bg)
        arr = normalize(expl.shap_values(Xs), len(Xte), len(full), len(linear.classes_))

        imp = np.mean(np.abs(arr), axis=(0, 2))
        global_by_h[h] = imp
        for f, v in zip(full, imp):
            allrows.append({
                "horizon_percent": h,
                "feature": f,
                "mean_abs_shap": float(v),
                "family": family_for_feature(f),
                "gas": gas_for_feature(f),
            })

        # True-class-conditioned attribution for that class's model output.
        for ci, c in enumerate(linear.classes_):
            mask = yte == c
            vals = np.mean(np.abs(arr[mask, :, ci]), axis=0)
            for f, v in zip(full, vals):
                classrows.append({
                    "horizon_percent": h,
                    "class_label": int(c),
                    "class_name": LABEL_NAMES[int(c)],
                    "n_cases": int(mask.sum()),
                    "feature": f,
                    "mean_abs_shap": float(v),
                    "family": family_for_feature(f),
                    "gas": gas_for_feature(f),
                })

        # Both mean-per-feature and total/share are useful: mean avoids automatically
        # favoring larger groups; total share describes aggregate attribution mass.
        d = pd.DataFrame([r for r in allrows if r["horizon_percent"] == h])
        total_imp = float(d.mean_abs_shap.sum())
        for group_type, col in (("family", "family"), ("gas", "gas")):
            for name, sub in d.groupby(col):
                grouprows.append({
                    "horizon_percent": h,
                    "group_type": group_type,
                    "group": name,
                    "n_features": int(len(sub)),
                    "mean_feature_importance": float(sub.mean_abs_shap.mean()),
                    "total_importance": float(sub.mean_abs_shap.sum()),
                    "share_of_total_importance": float(sub.mean_abs_shap.sum() / total_imp),
                })

    pd.DataFrame(allrows).to_csv(args.output_dir / "shap_global_all900.csv", index=False)
    pd.DataFrame(classrows).to_csv(args.output_dir / "shap_true_class_conditioned.csv", index=False)
    pd.DataFrame(grouprows).to_csv(args.output_dir / "shap_grouped.csv", index=False)

    stability = []
    for a, b in ((50, 75), (75, 100), (50, 100)):
        rho, p = spearmanr(global_by_h[a], global_by_h[b])
        ta = set(np.argsort(global_by_h[a])[-10:])
        tb = set(np.argsort(global_by_h[b])[-10:])
        stability.append({
            "horizon_a": a,
            "horizon_b": b,
            "spearman_rho_all_features": float(rho),
            "spearman_p": float(p),
            "top10_jaccard": len(ta & tb) / len(ta | tb),
        })
    pd.DataFrame(stability).to_csv(args.output_dir / "shap_stability.csv", index=False)

    gd = pd.DataFrame(grouprows)
    fam = gd[gd.group_type == "family"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, g in fam.groupby("group"):
        ax.plot(g.horizon_percent, g.share_of_total_importance, marker="o", label=name)
    ax.set_xlabel("History (%)")
    ax.set_ylabel("Share of total |SHAP|")
    ax.set_title("Feature-family attribution share across observation horizons")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "shap_family_stability.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
