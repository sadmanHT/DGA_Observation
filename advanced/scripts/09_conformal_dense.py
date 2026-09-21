from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from advanced_common import (
    DENSE_HORIZONS, LABELS, LABEL_NAMES, RANDOM_STATE, fixed_lr,
    feature_groups_from_columns, load_table, save_environment,
)


def conformal_quantile(scores, alpha):
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    rank = int(math.ceil((n + 1) * (1 - alpha)))
    rank = min(max(rank, 1), n)
    return float(s[rank - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--calibration-fraction", type=float, default=0.30)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_environment(args.output_dir / "environment.json")

    cols_all = pd.read_csv(args.features_dir / "features_train_h100.csv", nrows=1).columns.tolist()
    full = feature_groups_from_columns(cols_all)["full"]
    _, y0, ids0, _ = load_table(args.features_dir / "features_train_h100.csv", full)
    idx = np.arange(len(y0))
    proper, cal = train_test_split(
        idx, test_size=args.calibration_fraction, stratify=y0, random_state=RANDOM_STATE
    )

    split_df = pd.DataFrame({
        "row_index": idx,
        "id": ids0,
        "label": y0,
        "role": np.where(np.isin(idx, cal), "calibration", "proper_train"),
    })
    split_df.to_csv(args.output_dir / "conformal_train_split.csv", index=False)

    per_h = []
    class_cov_rows = []
    threshold_rows = []
    sets_by_h, preds_by_h = {}, {}
    ytest = idtest = None

    for h in DENSE_HORIZONS:
        Xtr, ytr, _, _ = load_table(args.features_dir / f"features_train_h{h}.csv", full)
        Xte, yte, ids, _ = load_table(args.features_dir / f"features_test_h{h}.csv", full)
        if ytest is None:
            ytest, idtest = yte.copy(), ids.copy()
        elif not (np.array_equal(ytest, yte) and np.array_equal(idtest, ids)):
            raise RuntimeError("TEST case order mismatch across horizons")

        model = clone(fixed_lr()).fit(Xtr[proper], ytr[proper])
        pcal = model.predict_proba(Xtr[cal])
        pte = model.predict_proba(Xte)
        classes = model.classes_

        q = {}
        for c in LABELS:
            ci = list(classes).index(c)
            mask = ytr[cal] == c
            scores = 1.0 - pcal[mask, ci]
            q[int(c)] = conformal_quantile(scores, args.alpha)
            threshold_rows.append({
                "horizon_percent": h,
                "class_label": int(c),
                "class_name": LABEL_NAMES[int(c)],
                "n_calibration": int(mask.sum()),
                "alpha": args.alpha,
                "q_nonconformity": q[int(c)],
                "probability_inclusion_threshold": 1.0 - q[int(c)],
            })

        sets = []
        sizes = np.empty(len(yte), dtype=int)
        covered = np.empty(len(yte), dtype=int)
        singleton = np.empty(len(yte), dtype=int)
        for i in range(len(yte)):
            s = tuple(
                int(c) for ci, c in enumerate(classes)
                if (1.0 - pte[i, ci]) <= q[int(c)]
            )
            sets.append(s)
            sizes[i] = len(s)
            covered[i] = int(yte[i] in s)
            singleton[i] = int(len(s) == 1)

        sets_by_h[h] = sets
        preds_by_h[h] = classes[np.argmax(pte, axis=1)]
        per_h.append({
            "horizon_percent": h,
            "target_coverage": 1.0 - args.alpha,
            "marginal_coverage": float(np.mean(covered)),
            "mean_set_size": float(np.mean(sizes)),
            "median_set_size": float(np.median(sizes)),
            "singleton_rate": float(np.mean(singleton)),
            "empty_set_rate": float(np.mean(sizes == 0)),
            "multi_label_rate": float(np.mean(sizes > 1)),
        })

        for c in LABELS:
            mask = yte == c
            class_cov_rows.append({
                "horizon_percent": h,
                "class_label": int(c),
                "class_name": LABEL_NAMES[int(c)],
                "n_test": int(mask.sum()),
                "empirical_class_coverage": float(np.mean(covered[mask])),
                "class_singleton_rate": float(np.mean(singleton[mask])),
                "class_mean_set_size": float(np.mean(sizes[mask])),
            })
        print(h, per_h[-1])

    pd.DataFrame(threshold_rows).to_csv(args.output_dir / "conformal_thresholds.csv", index=False)
    pd.DataFrame(per_h).to_csv(args.output_dir / "conformal_by_horizon.csv", index=False)
    pd.DataFrame(class_cov_rows).to_csv(args.output_dir / "conformal_classwise_coverage.csv", index=False)


    stop = np.full(len(ytest), 100, int)
    pred = np.full(len(ytest), -1, int)
    setsize = np.full(len(ytest), -1, int)
    found = np.zeros(len(ytest), bool)
    for i in range(len(ytest)):
        for j in range(1, len(DENSE_HORIZONS)):
            h0, h1 = DENSE_HORIZONS[j - 1], DENSE_HORIZONS[j]
            a, b = sets_by_h[h0][i], sets_by_h[h1][i]
            if len(a) == 1 and len(b) == 1 and a[0] == b[0]:
                stop[i], pred[i], setsize[i], found[i] = h1, b[0], 1, True
                break
        if not found[i]:
            s = sets_by_h[100][i]
            setsize[i] = len(s)
            pred[i] = s[0] if len(s) == 1 else int(preds_by_h[100][i])

    empirical_correct = pred == ytest
    summary = {
        "alpha": args.alpha,
        "target_fixed_horizon_coverage": 1.0 - args.alpha,
        "calibration_fraction": args.calibration_fraction,
        "proper_training_size": int(len(proper)),
        "calibration_size": int(len(cal)),
        "sequential_policy": "first identical singleton conformal set at two consecutive horizons; exploratory only",
        "sequential_warning": "No anytime-valid coverage claim is made under optional stopping.",
        "fraction_stopped_with_stable_singleton": float(found.mean()),
        "mean_stop_percent_all": float(stop.mean()),
        "median_stop_percent_all": float(np.median(stop)),
        "accuracy_all_with_endpoint_fallback": float(empirical_correct.mean()),
        "accuracy_among_stable_singletons": float(empirical_correct[found].mean()) if found.any() else None,
    }
    (args.output_dir / "conformal_sequential_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame({
        "id": idtest,
        "true_label": ytest,
        "stop_horizon_percent": stop,
        "predicted_label": pred,
        "stable_singleton": found.astype(int),
        "correct": empirical_correct.astype(int),
        "set_size_at_stop_or_endpoint": setsize,
    }).to_csv(args.output_dir / "conformal_sequential_per_case.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
