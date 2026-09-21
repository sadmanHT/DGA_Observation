from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score

from advanced_common import (
    RANDOM_STATE, classification_metrics, feature_groups_from_columns, fixed_lr,
    load_table, paired_macro_f1_bootstrap, save_environment,
)

HORIZONS = [50, 75, 100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=5000)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_environment(args.output_dir / "environment.json")

    cols_all = pd.read_csv(args.features_dir / "features_train_h100.csv", nrows=1).columns.tolist()
    g = feature_groups_from_columns(cols_all)
    base = g["statistical"]
    temporal_groups = [
        "endpoint_change", "trend", "local_dynamics", "phase_shift", "cross_gas_correlation"
    ]

    specs = {"Statistical_28": base, "Full_temporal_82": g["full"]}
    for name in temporal_groups:
        specs[f"Statistical_plus_{name}"] = base + g[name]
    for name in temporal_groups:
        specs[f"Full_minus_{name}"] = [c for c in g["full"] if c not in set(g[name])]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model = fixed_lr()
    rows = []
    preds: dict[tuple[int, str], np.ndarray] = {}
    y_by_h: dict[int, np.ndarray] = {}

    for h in HORIZONS:
        for name, cols in specs.items():
            Xtr, ytr, _, _ = load_table(args.features_dir / f"features_train_h{h}.csv", cols)
            Xte, yte, ids, _ = load_table(args.features_dir / f"features_test_h{h}.csv", cols)
            if h not in y_by_h:
                y_by_h[h] = yte.copy()
            elif not np.array_equal(y_by_h[h], yte):
                raise RuntimeError(f"TEST label order mismatch at h={h}")

            t0 = time.perf_counter()
            scores = cross_val_score(
                clone(model), Xtr, ytr, cv=cv, scoring="f1_macro", n_jobs=1
            )
            fitted = clone(model).fit(Xtr, ytr)
            pred = fitted.predict(Xte)
            preds[(h, name)] = pred.copy()
            rows.append({
                "horizon_percent": h,
                "feature_set": name,
                "n_features": len(cols),
                "cv_macro_f1_mean": float(scores.mean()),
                "cv_macro_f1_std": float(scores.std(ddof=1)),
                **{f"test_{k}": v for k, v in classification_metrics(yte, pred).items()},
                "seconds": time.perf_counter() - t0,
            })
            print(h, name, rows[-1]["test_macro_f1"])

    out = pd.DataFrame(rows)
    out.to_csv(args.output_dir / "feature_family_ablation.csv", index=False)


    h = 75
    y = y_by_h[h]
    stat_pred = preds[(h, "Statistical_28")]
    full_pred = preds[(h, "Full_temporal_82")]
    stat_f1 = float(out[(out.horizon_percent == h) & (out.feature_set == "Statistical_28")].test_macro_f1.iloc[0])
    full_f1 = float(out[(out.horizon_percent == h) & (out.feature_set == "Full_temporal_82")].test_macro_f1.iloc[0])

    summary = []
    for name in temporal_groups:
        plus_name = f"Statistical_plus_{name}"
        minus_name = f"Full_minus_{name}"
        plus_f1 = float(out[(out.horizon_percent == h) & (out.feature_set == plus_name)].test_macro_f1.iloc[0])
        minus_f1 = float(out[(out.horizon_percent == h) & (out.feature_set == minus_name)].test_macro_f1.iloc[0])

        add_ci = paired_macro_f1_bootstrap(
            y, stat_pred, preds[(h, plus_name)], n_boot=args.bootstrap,
            seed=RANDOM_STATE + 100 + temporal_groups.index(name),
        )

        remove_ci = paired_macro_f1_bootstrap(
            y, preds[(h, minus_name)], full_pred, n_boot=args.bootstrap,
            seed=RANDOM_STATE + 200 + temporal_groups.index(name),
        )
        summary.append({
            "group": name,
            "n_group_features": len(g[name]),
            "statistical_only_f1": stat_f1,
            "stat_plus_group_f1": plus_f1,
            "gain_when_added_to_stat": plus_f1 - stat_f1,
            "gain_ci95_low": add_ci["ci95_low"],
            "gain_ci95_high": add_ci["ci95_high"],
            "full_f1": full_f1,
            "full_minus_group_f1": minus_f1,
            "drop_when_removed_from_full": full_f1 - minus_f1,
            "drop_ci95_low": remove_ci["ci95_low"],
            "drop_ci95_high": remove_ci["ci95_high"],
            "bootstrap_samples": args.bootstrap,
        })
    pd.DataFrame(summary).to_csv(args.output_dir / "feature_family_ablation_75_summary.csv", index=False)


if __name__ == "__main__":
    main()
