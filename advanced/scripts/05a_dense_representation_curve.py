#!/usr/bin/env python3
"""Dense horizon x representation analysis (10%..100% in 5% steps).

For each horizon, fit the paper's fixed logistic-regression configuration separately on
Statistical-28 and Temporal-82 features using TRAIN only, then evaluate the untouched
900-case TEST split.  A paired bootstrap quantifies Temporal-minus-Statistical macro-F1
at each horizon.  The script also writes training 5-fold CV scores so TEST is not the only
view of the dense curve.

This is descriptive benchmarking across multiple horizons. Individual per-horizon CIs are
not interpreted as a multiplicity-adjusted family of hypothesis tests; the pre-specified
75%-vs-100% difference-of-differences remains the primary formal interaction contrast.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score

from advanced_common import (
    DENSE_HORIZONS, RANDOM_STATE, feature_groups_from_columns, fixed_lr,
    load_table, classification_metrics, paired_macro_f1_bootstrap, save_environment,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=5000)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_environment(args.output_dir / "environment.json")

    cols = pd.read_csv(args.features_dir / "features_train_h100.csv", nrows=1).columns.tolist()
    groups = feature_groups_from_columns(cols)
    specs = {"Statistical_28": groups["statistical"], "Temporal_82": groups["full"]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model = fixed_lr()
    rows = []

    for h in DENSE_HORIZONS:
        fitted = {}
        y_ref = ids_ref = None
        for name, fcols in specs.items():
            Xtr, ytr, _, _ = load_table(args.features_dir / f"features_train_h{h}.csv", fcols)
            Xte, yte, ids, _ = load_table(args.features_dir / f"features_test_h{h}.csv", fcols)
            if y_ref is None:
                y_ref, ids_ref = yte.copy(), ids.copy()
            elif not (np.array_equal(y_ref, yte) and np.array_equal(ids_ref, ids)):
                raise RuntimeError(f"TEST pairing mismatch at horizon {h}")
            cvs = cross_val_score(clone(model), Xtr, ytr, cv=cv, scoring="f1_macro", n_jobs=1)
            m = clone(model).fit(Xtr, ytr)
            pred = m.predict(Xte)
            fitted[name] = pred
            met = classification_metrics(yte, pred)
            rows.append({
                "horizon_percent": h,
                "representation": name,
                "n_features": len(fcols),
                "cv_macro_f1_mean": float(cvs.mean()),
                "cv_macro_f1_std": float(cvs.std(ddof=1)),
                **{f"test_{k}": v for k, v in met.items()},
            })

        ci = paired_macro_f1_bootstrap(
            y_ref, fitted["Statistical_28"], fitted["Temporal_82"],
            n_boot=args.bootstrap, seed=RANDOM_STATE + h,
        )
        rows.append({
            "horizon_percent": h,
            "representation": "Temporal_minus_Statistical",
            "n_features": np.nan,
            "cv_macro_f1_mean": np.nan,
            "cv_macro_f1_std": np.nan,
            "test_accuracy": np.nan,
            "test_balanced_accuracy": np.nan,
            "test_macro_f1": ci["difference_b_minus_a"],
            "test_weighted_f1": np.nan,
            "paired_ci95_low": ci["ci95_low"],
            "paired_ci95_high": ci["ci95_high"],
        })
        print(h, "delta", ci["difference_b_minus_a"], ci["ci95_low"], ci["ci95_high"])

    out = pd.DataFrame(rows)
    out.to_csv(args.output_dir / "dense_representation_curve.csv", index=False)

    methods = out[out.representation.isin(["Statistical_28", "Temporal_82"])]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for name, g in methods.groupby("representation"):
        ax.plot(g.horizon_percent, g.test_macro_f1, marker="o", label=name.replace("_", " "))
    ax.set_xlabel("Available DGA history (%)")
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Dense horizon × representation performance")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(args.output_dir / "dense_representation_curve.png", dpi=300)
    plt.close(fig)

    d = out[out.representation == "Temporal_minus_Statistical"].sort_values("horizon_percent")
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    err_low = d.test_macro_f1 - d.paired_ci95_low
    err_hi = d.paired_ci95_high - d.test_macro_f1
    ax.errorbar(d.horizon_percent, d.test_macro_f1, yerr=np.vstack([err_low, err_hi]), marker="o", capsize=3)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Available DGA history (%)")
    ax.set_ylabel("Temporal-82 − Statistical-28 macro-F1")
    ax.set_title("Dense representation effect with paired 95% bootstrap intervals")
    ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(args.output_dir / "dense_temporal_advantage.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
