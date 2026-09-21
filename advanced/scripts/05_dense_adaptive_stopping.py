from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from advanced_common import (
    DENSE_HORIZONS, LABELS, RANDOM_STATE, class_metrics,
    classification_metrics, feature_groups_from_columns, fixed_lr,
    load_table, save_environment,
)


def policy_predictions(horizons, preds_by_h, probs_by_h, threshold, persistence):
    n = len(next(iter(preds_by_h.values())))
    stop_h = np.full(n, horizons[-1], dtype=int)
    stop_pred = np.empty(n, dtype=int)
    stop_conf = np.empty(n, dtype=float)
    satisfied = np.zeros(n, dtype=bool)

    for i in range(n):
        chosen = False
        for pos, h in enumerate(horizons):
            p = int(preds_by_h[h][i])
            conf = float(np.max(probs_by_h[h][i]))
            if conf < threshold:
                continue
            if persistence > 1:
                if pos + 1 < persistence:
                    continue
                hs = horizons[pos - persistence + 1 : pos + 1]
                if not all(
                    int(preds_by_h[hh][i]) == p
                    and float(np.max(probs_by_h[hh][i])) >= threshold
                    for hh in hs
                ):
                    continue
            stop_h[i] = h
            stop_pred[i] = p
            stop_conf[i] = conf
            satisfied[i] = True
            chosen = True
            break

        if not chosen:
            h = horizons[-1]
            stop_pred[i] = int(preds_by_h[h][i])
            stop_conf[i] = float(np.max(probs_by_h[h][i]))

    return stop_h, stop_pred, stop_conf, satisfied


def policy_row(y, stop_h, pred, satisfied, threshold, persistence, endpoint_macro):
    m = classification_metrics(y, pred)
    recalls = recall_score(y, pred, labels=LABELS, average=None, zero_division=0)
    row = {
        "threshold": threshold,
        "persistence": persistence,
        **m,
        "endpoint_macro_f1_reference": endpoint_macro,
        "retention_vs_endpoint": m["macro_f1"] / endpoint_macro if endpoint_macro else np.nan,
        "mean_stop_percent": float(np.mean(stop_h)),
        "median_stop_percent": float(np.median(stop_h)),
        "p25_stop_percent": float(np.quantile(stop_h, 0.25)),
        "p75_stop_percent": float(np.quantile(stop_h, 0.75)),
        "fraction_stopped_by_50": float(np.mean(stop_h <= 50)),
        "fraction_stopped_by_75": float(np.mean(stop_h <= 75)),
        "fraction_stopped_before_100": float(np.mean(stop_h < 100)),
        "fraction_threshold_satisfied": float(np.mean(satisfied)),
        "minimum_class_recall": float(np.min(recalls)),
    }
    row.update({f"recall_class_{int(c)}": float(recalls[j]) for j, c in enumerate(LABELS)})
    return row


def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:

    keep = []
    vals = df[["mean_stop_percent", "macro_f1"]].to_numpy(float)
    for i, (stop_i, f1_i) in enumerate(vals):
        dominated = np.any(
            (vals[:, 0] <= stop_i)
            & (vals[:, 1] >= f1_i)
            & ((vals[:, 0] < stop_i) | (vals[:, 1] > f1_i))
        )
        keep.append(not dominated)
    return df.loc[keep].sort_values("mean_stop_percent").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--target-retention", type=float, default=0.97,
                    help="Minimum TRAIN-OOF adaptive macro-F1 / 100%%-history OOF macro-F1.")
    ap.add_argument("--min-class-recall", type=float, default=0.75,
                    help="Minimum TRAIN-OOF recall required for every class.")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_environment(args.output_dir / "environment.json")

    df100 = pd.read_csv(args.features_dir / "features_train_h100.csv", nrows=1)
    cols = feature_groups_from_columns(df100.columns.tolist())["full"]
    model = fixed_lr()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    oof_pred, oof_prob, test_pred, test_prob = {}, {}, {}, {}
    rows = []
    y_train_ref = y_test_ref = ids_train_ref = ids_test_ref = None

    for h in DENSE_HORIZONS:
        print(f"dense horizon {h}%")
        Xtr, ytr, ids_tr, _ = load_table(args.features_dir / f"features_train_h{h}.csv", cols)
        Xte, yte, ids_te, _ = load_table(args.features_dir / f"features_test_h{h}.csv", cols)
        if y_train_ref is None:
            y_train_ref, ids_train_ref = ytr.copy(), ids_tr.copy()
        if y_test_ref is None:
            y_test_ref, ids_test_ref = yte.copy(), ids_te.copy()
        if not (np.array_equal(y_train_ref, ytr) and np.array_equal(ids_train_ref, ids_tr)):
            raise RuntimeError("TRAIN case order differs across dense horizons")
        if not (np.array_equal(y_test_ref, yte) and np.array_equal(ids_test_ref, ids_te)):
            raise RuntimeError("TEST case order differs across dense horizons")

        t0 = time.perf_counter()
        prob_oof = cross_val_predict(clone(model), Xtr, ytr, cv=cv, method="predict_proba", n_jobs=1)
        elapsed_cv = time.perf_counter() - t0

        classes = np.sort(np.unique(ytr))
        pred_oof = classes[np.argmax(prob_oof, axis=1)]

        fitted = clone(model).fit(Xtr, ytr)
        prob_te = fitted.predict_proba(Xte)
        pred_te = fitted.classes_[np.argmax(prob_te, axis=1)]

        oof_pred[h], oof_prob[h] = pred_oof, prob_oof
        test_pred[h], test_prob[h] = pred_te, prob_te
        train_m = classification_metrics(ytr, pred_oof)
        test_m = classification_metrics(yte, pred_te)
        rows.append({
            "horizon_percent": h,
            "observation_steps": int(round(420 * h / 100)),
            "sample_equivalent_days": int(round(420 * h / 100)) * 0.5,
            **{f"oof_{k}": v for k, v in train_m.items()},
            **{f"test_{k}": v for k, v in test_m.items()},
            "oof_cv_seconds": elapsed_cv,
        })

    curve = pd.DataFrame(rows)
    curve.to_csv(args.output_dir / "dense_horizon_curve.csv", index=False)
    endpoint_oof = float(curve.loc[curve.horizon_percent == 100, "oof_macro_f1"].iloc[0])

    candidates = []
    for persistence in (1, 2, 3):
        for threshold in np.round(np.arange(0.50, 0.976, 0.025), 3):
            stop_h, pred, conf, satisfied = policy_predictions(
                DENSE_HORIZONS, oof_pred, oof_prob, float(threshold), persistence
            )
            candidates.append(policy_row(
                y_train_ref, stop_h, pred, satisfied,
                float(threshold), persistence, endpoint_oof,
            ))

    cand = pd.DataFrame(candidates)
    cand["meets_retention_constraint"] = cand.retention_vs_endpoint >= args.target_retention
    cand["meets_class_recall_constraint"] = cand.minimum_class_recall >= args.min_class_recall
    cand["eligible"] = cand.meets_retention_constraint & cand.meets_class_recall_constraint
    cand.to_csv(args.output_dir / "adaptive_policy_oof_grid.csv", index=False)
    pareto_frontier(cand).to_csv(args.output_dir / "adaptive_policy_pareto_frontier.csv", index=False)

    eligible = cand[cand.eligible].copy()
    if eligible.empty:


        chosen = cand.sort_values(
            ["retention_vs_endpoint", "minimum_class_recall", "mean_stop_percent"],
            ascending=[False, False, True],
        ).iloc[0]
        selection_note = (
            "No OOF policy met both preregistered constraints. Fallback shown only for "
            "diagnostic exploration: highest OOF retention, then highest minimum class recall, "
            "then earliest mean stop."
        )
        constraints_met = False
    else:
        chosen = eligible.sort_values(
            ["mean_stop_percent", "macro_f1", "minimum_class_recall"],
            ascending=[True, False, False],
        ).iloc[0]
        selection_note = (
            f"Selected using TRAIN OOF only: earliest mean stopping horizon among policies "
            f"retaining >= {args.target_retention:.3f} of endpoint OOF macro-F1 and "
            f"minimum class recall >= {args.min_class_recall:.3f}."
        )
        constraints_met = True

    threshold = float(chosen.threshold)
    persistence = int(chosen.persistence)
    stop_h, pred, conf, satisfied = policy_predictions(
        DENSE_HORIZONS, test_pred, test_prob, threshold, persistence
    )
    test_row = policy_row(
        y_test_ref, stop_h, pred, satisfied, threshold, persistence,
        float(curve.loc[curve.horizon_percent == 100, "test_macro_f1"].iloc[0]),
    )
    pd.DataFrame([test_row]).to_csv(args.output_dir / "adaptive_policy_test_result.csv", index=False)

    pd.DataFrame({
        "id": ids_test_ref,
        "true_label": y_test_ref,
        "stop_horizon_percent": stop_h,
        "predicted_label": pred,
        "confidence_at_stop": conf,
        "threshold_condition_satisfied": satisfied.astype(int),
        "correct": (y_test_ref == pred).astype(int),
    }).to_csv(args.output_dir / "adaptive_policy_test_per_case.csv", index=False)

    crows = class_metrics(y_test_ref, pred)
    for r in crows:
        mask = y_test_ref == r["class_label"]
        r["mean_stop_percent"] = float(np.mean(stop_h[mask]))
        r["median_stop_percent"] = float(np.median(stop_h[mask]))
        r["fraction_stopped_by_75"] = float(np.mean(stop_h[mask] <= 75))
        r["fraction_stopped_before_100"] = float(np.mean(stop_h[mask] < 100))
    pd.DataFrame(crows).to_csv(args.output_dir / "adaptive_policy_classwise.csv", index=False)

    meta = {
        "selection_note": selection_note,
        "constraints_met_on_oof": constraints_met,
        "target_retention": args.target_retention,
        "minimum_class_recall_constraint": args.min_class_recall,
        "selected_threshold": threshold,
        "selected_persistence": persistence,
        "oof_endpoint_macro_f1": endpoint_oof,
        "selected_oof_policy": chosen.to_dict(),
        "test_result": test_row,
        "important": "Policy selection uses TRAIN OOF only; TEST is used only for final evaluation.",
    }
    (args.output_dir / "adaptive_policy_summary.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(curve.horizon_percent, curve.test_macro_f1, marker="o", label="Test macro-F1")
    ax.plot(curve.horizon_percent, curve.oof_macro_f1, marker="s", label="Training OOF macro-F1")
    ax.set_xlabel("Available history (%)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Dense observation-horizon performance")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "dense_horizon_curve.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    vals, counts = np.unique(stop_h, return_counts=True)
    ax.bar(vals.astype(str), counts)
    ax.set_xlabel("Adaptive stopping horizon (%)")
    ax.set_ylabel("Test cases")
    ax.set_title("Training-selected adaptive stopping distribution")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(args.output_dir / "adaptive_stop_distribution.png", dpi=300)
    plt.close(fig)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
