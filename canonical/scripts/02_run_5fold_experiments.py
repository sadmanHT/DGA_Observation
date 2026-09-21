from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
CV_FOLDS = 5
LABELS = [1, 2, 3, 4]
LABEL_NAMES = {
    1: "Normal mode",
    2: "Partial discharge",
    3: "Low energy discharge",
    4: "Low-temperature overheating",
}
HORIZONS = [25, 50, 75, 100]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(features_dir: Path) -> dict:
    with (features_dir / "feature_manifest.json").open(encoding="utf-8") as f:
        return json.load(f)


def load_table(path: Path, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    missing = sorted(set(feature_cols + ["id", "label"]) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing[:10]}")
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    ids = df["id"].astype(str).to_numpy()
    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite values detected in {path}")
    return X, y, ids


def scorers() -> dict:
    return {"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"}


def base_models(smoke: bool = False) -> dict:
    n_trees = 100 if smoke else 400
    hgb_iter = 100 if smoke else 300
    return {
        "LogisticRegression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "SVM_RBF": Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(
                C=10.0,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                probability=False,
                random_state=RANDOM_STATE,
            )),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=n_trees,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=n_trees,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=hgb_iter,
            learning_rate=0.08,
            max_leaf_nodes=31,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def model_search_space(model_name: str) -> dict:
    if model_name == "LogisticRegression":
        return {
            "model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
            "model__class_weight": ["balanced", None],
        }
    if model_name in {"RandomForest", "ExtraTrees"}:
        return {
            "n_estimators": [300, 500, 800, 1200],
            "max_depth": [None, 8, 12, 16, 24, 32],
            "min_samples_split": [2, 4, 8, 12],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.5, 0.75, 1.0],
            "class_weight": ["balanced", "balanced_subsample", None],
            "criterion": ["gini", "entropy", "log_loss"],
        }
    if model_name == "HistGradientBoosting":
        return {
            "learning_rate": [0.03, 0.05, 0.08, 0.1],
            "max_iter": [150, 250, 350, 500],
            "max_leaf_nodes": [15, 31, 63],
            "max_depth": [None, 4, 8, 12],
            "min_samples_leaf": [10, 20, 30, 50],
            "l2_regularization": [0.0, 0.1, 1.0, 10.0],
            "class_weight": ["balanced", None],
        }
    raise ValueError(f"No tuning space defined for {model_name}")


def evaluate(model, X_train, y_train, X_test, y_test, cv) -> tuple[dict, object, np.ndarray, np.ndarray | None]:
    cvres = cross_validate(
        clone(model),
        X_train,
        y_train,
        cv=cv,
        scoring=scorers(),
        n_jobs=1,
        return_train_score=False,
    )
    fitted = clone(model)
    t0 = time.perf_counter()
    fitted.fit(X_train, y_train)
    fit_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = fitted.predict(X_test)
    infer_sec = time.perf_counter() - t0
    prob = fitted.predict_proba(X_test) if hasattr(fitted, "predict_proba") else None
    row = {
        "cv_macro_f1_mean": float(np.mean(cvres["test_macro_f1"])),
        "cv_macro_f1_std": float(np.std(cvres["test_macro_f1"], ddof=1)),
        "cv_balanced_accuracy_mean": float(np.mean(cvres["test_balanced_accuracy"])),
        "cv_fit_seconds_mean": float(np.mean(cvres["fit_time"])),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "test_macro_f1": float(f1_score(y_test, pred, labels=LABELS, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(y_test, pred, labels=LABELS, average="weighted", zero_division=0)),
        "fit_seconds": float(fit_sec),
        "inference_ms_per_case": float(1000.0 * infer_sec / len(y_test)),
    }
    return row, fitted, pred, prob


def class_report_rows(y_true, y_pred) -> list[dict]:
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=[LABEL_NAMES[c] for c in LABELS],
        output_dict=True,
        zero_division=0,
    )
    return [
        {
            "class_label": cls,
            "class_name": LABEL_NAMES[cls],
            "precision": float(report[LABEL_NAMES[cls]]["precision"]),
            "recall": float(report[LABEL_NAMES[cls]]["recall"]),
            "f1": float(report[LABEL_NAMES[cls]]["f1-score"]),
            "support": int(report[LABEL_NAMES[cls]]["support"]),
        }
        for cls in LABELS
    ]


def classwise_rows(y_true, y_pred, representation: str, horizon: int) -> list[dict]:
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    return [
        {
            "representation": representation,
            "observation_percent": horizon,
            "class_label": cls,
            "class_name": LABEL_NAMES[cls],
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(support[i]),
        }
        for i, cls in enumerate(LABELS)
    ]


def set_class_weight(model, value):
    m = clone(model)
    params = m.get_params(deep=True)
    if "model__class_weight" in params:
        m.set_params(model__class_weight=value)
    elif "class_weight" in params:
        m.set_params(class_weight=value)
    else:
        raise ValueError(f"Model {type(model).__name__} has no class_weight parameter")
    return m


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int, rng: np.random.Generator) -> list[dict]:
    metric_fns = {
        "accuracy": lambda a, b: accuracy_score(a, b),
        "balanced_accuracy": lambda a, b: balanced_accuracy_score(a, b),
        "macro_f1": lambda a, b: f1_score(a, b, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": lambda a, b: f1_score(a, b, labels=LABELS, average="weighted", zero_division=0),
    }
    n = len(y_true)
    idx_all = rng.integers(0, n, size=(n_boot, n))
    rows = []
    for name, fn in metric_fns.items():
        values = np.asarray([fn(y_true[idx], y_pred[idx]) for idx in idx_all], dtype=float)
        rows.append({
            "metric": name,
            "estimate": float(fn(y_true, y_pred)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "bootstrap_samples": n_boot,
        })
    return rows


def paired_bootstrap_difference(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    name_a: str,
    name_b: str,
    n_boot: int,
    rng: np.random.Generator,
) -> list[dict]:
    metric_fns = {
        "accuracy": lambda a, b: accuracy_score(a, b),
        "balanced_accuracy": lambda a, b: balanced_accuracy_score(a, b),
        "macro_f1": lambda a, b: f1_score(a, b, labels=LABELS, average="macro", zero_division=0),
    }
    n = len(y_true)
    idx_all = rng.integers(0, n, size=(n_boot, n))
    rows = []
    for metric_name, fn in metric_fns.items():
        score_a = float(fn(y_true, pred_a))
        score_b = float(fn(y_true, pred_b))
        diffs = np.asarray([
            fn(y_true[idx], pred_b[idx]) - fn(y_true[idx], pred_a[idx])
            for idx in idx_all
        ])
        rows.append({
            "metric": metric_name,
            "comparison_a": name_a,
            "comparison_b": name_b,
            "score_a": score_a,
            "score_b": score_b,
            "difference_b_minus_a": float(score_b - score_a),
            "ci95_low": float(np.quantile(diffs, 0.025)),
            "ci95_high": float(np.quantile(diffs, 0.975)),
            "bootstrap_samples": n_boot,
        })
    return rows


def selective_rows(y_true, y_pred, prob, representation: str, horizon: int) -> list[dict]:
    confidence = np.max(prob, axis=1)
    rows = []
    for threshold in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        accepted = confidence >= threshold
        if not np.any(accepted):
            continue
        yt, yp = y_true[accepted], y_pred[accepted]
        row = {
            "representation": representation,
            "observation_percent": horizon,
            "confidence_threshold": threshold,
            "accepted_cases": int(np.sum(accepted)),
            "rejected_cases": int(len(y_true) - np.sum(accepted)),
            "coverage": float(np.mean(accepted)),
            "accepted_accuracy": float(accuracy_score(yt, yp)),
            "accepted_balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "accepted_macro_f1_fixed4": float(
                f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0)
            ),
            "classes_present_in_accepted": int(len(np.unique(yt))),
        }
        for cls in LABELS:
            class_mask = y_true == cls
            row[f"accepted_class_{cls}"] = int(np.sum(accepted & class_mask))
            row[f"coverage_class_{cls}"] = float(np.mean(accepted[class_mask]))
        rows.append(row)
    return rows


def save_prediction_table(path: Path, ids, y_true, y_pred, prob, classes) -> None:
    data = {
        "id": ids,
        "true_label": y_true,
        "predicted_label": y_pred,
        "correct": (y_true == y_pred).astype(int),
    }
    if prob is not None:
        data["confidence"] = np.max(prob, axis=1)
        for j, cls in enumerate(classes):
            data[f"p_class_{int(cls)}"] = prob[:, j]
    pd.DataFrame(data).to_csv(path, index=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def plot_confusion(y_true, y_pred, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=LABELS,
        display_labels=[LABEL_NAMES[c] for c in LABELS],
        xticks_rotation=25,
        ax=ax,
    )
    ax.set_title("Final 100% History Test-Set Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_horizon_representation(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    labels = {
        "Statistical_28": "Statistical summary (28)",
        "Full_temporal_82": "Full temporal (82)",
    }
    for rep in ["Statistical_28", "Full_temporal_82"]:
        g = df[df["representation"] == rep].sort_values("observation_percent")
        ax.plot(g["observation_percent"], g["test_macro_f1"], marker="o", linewidth=2, label=labels[rep])
    ax.set_xlabel("Available DGA history (%)")
    ax.set_ylabel("Test macro-F1")
    ax.set_xticks(HORIZONS)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_classwise(df: pd.DataFrame, representation: str, out_path: Path) -> None:
    sub = df[df["representation"] == representation]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for class_name, g in sub.groupby("class_name", sort=False):
        g = g.sort_values("observation_percent")
        ax.plot(g["observation_percent"], g["f1"], marker="o", linewidth=2, label=class_name)
    ax.set_xlabel("Available DGA history (%)")
    ax.set_ylabel("Class F1")
    ax.set_xticks(HORIZONS)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_advantage(df: pd.DataFrame, out_path: Path) -> None:
    sub = df[df["metric"] == "macro_f1"].sort_values("observation_percent")
    x = sub["observation_percent"].to_numpy()
    y = sub["difference_b_minus_a"].to_numpy()
    lo = sub["ci95_low"].to_numpy()
    hi = sub["ci95_high"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), marker="o", capsize=4, linewidth=2)
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Available DGA history (%)")
    ax.set_ylabel("Macro-F1 difference\n(temporal 82 - statistical 28)")
    ax.set_xticks(HORIZONS)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_selective(df: pd.DataFrame, representation: str, horizon: int, out_path: Path) -> None:
    sub = df[(df["representation"] == representation) & (df["observation_percent"] == horizon)].sort_values("coverage")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(sub["coverage"], sub["accepted_accuracy"], marker="o", linewidth=2, label="Accepted accuracy")
    ax.plot(sub["coverage"], sub["accepted_macro_f1_fixed4"], marker="s", linewidth=2, label="Accepted macro-F1")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Score on accepted cases")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tuning-iterations", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--smoke", action="store_true", help="Fast code check; still uses 5-fold CV")
    args = parser.parse_args()

    results_dir = args.output_dir / "results"
    figures_dir = args.output_dir / "figures"
    predictions_dir = args.output_dir / "predictions"
    for d in [results_dir, figures_dir, predictions_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.tuning_iterations = min(args.tuning_iterations, 2)
        args.bootstrap = min(args.bootstrap, 200)

    manifest = load_manifest(args.features_dir)
    fg = manifest["feature_groups"]
    full_features = fg["full"]
    stat_features = fg["statistical"]
    if len(full_features) != 82 or len(stat_features) != 28:
        raise ValueError(
            f"Unexpected feature counts: full={len(full_features)}, statistical={len(stat_features)}"
        )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    X_train, y_train, train_ids = load_table(args.features_dir / "features_train_h100.csv", full_features)
    X_test, y_test, test_ids = load_table(args.features_dir / "features_test_h100.csv", full_features)


    print("[1/8] Five-fold TRAIN-CV model benchmark")
    benchmark_rows = []
    for name, model in base_models(args.smoke).items():
        print(f"  - {name}")
        row, _, _, _ = evaluate(model, X_train, y_train, X_test, y_test, cv)
        benchmark_rows.append({"model": name, **row})
    write_rows(results_dir / "01_model_benchmark_5fold.csv", benchmark_rows)


    explainable_names = {"LogisticRegression", "RandomForest", "ExtraTrees", "HistGradientBoosting"}
    explainable_rows = [r for r in benchmark_rows if r["model"] in explainable_names]
    selected_name = max(explainable_rows, key=lambda r: r["cv_macro_f1_mean"])["model"]
    print(f"Selected by TRAIN-CV macro-F1 only: {selected_name}")


    print("[2/8] Five-fold randomized hyperparameter search")
    search = RandomizedSearchCV(
        estimator=base_models(args.smoke)[selected_name],
        param_distributions=model_search_space(selected_name),
        n_iter=args.tuning_iterations,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    best_pred = best_model.predict(X_test)
    best_prob = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else None

    tuning_rows = []
    for i in np.argsort(search.cv_results_["rank_test_score"]):
        tuning_rows.append({
            "rank": int(search.cv_results_["rank_test_score"][i]),
            "mean_cv_macro_f1": float(search.cv_results_["mean_test_score"][i]),
            "std_cv_macro_f1": float(search.cv_results_["std_test_score"][i]),
            "params": json.dumps(search.cv_results_["params"][i], sort_keys=True),
        })
    write_rows(results_dir / "02_hyperparameter_search_5fold.csv", tuning_rows)

    final_metrics = [{
        "model": f"Tuned_{selected_name}",
        "cv_folds": CV_FOLDS,
        "best_cv_macro_f1": float(search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, best_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, best_pred)),
        "test_macro_f1": float(f1_score(y_test, best_pred, labels=LABELS, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(y_test, best_pred, labels=LABELS, average="weighted", zero_division=0)),
        "best_params": json.dumps(search.best_params_, sort_keys=True),
    }]
    write_rows(results_dir / "03_final_model_metrics_5fold.csv", final_metrics)
    write_rows(results_dir / "04_final_classification_report.csv", class_report_rows(y_test, best_pred))
    write_rows(results_dir / "05_bootstrap_ci.csv", bootstrap_ci(y_test, best_pred, args.bootstrap, rng))
    plot_confusion(y_test, best_pred, figures_dir / "confusion_matrix_final.png")
    joblib.dump(best_model, results_dir / "best_model.joblib")
    (results_dir / "best_model_features.json").write_text(json.dumps(full_features, indent=2), encoding="utf-8")
    save_prediction_table(
        predictions_dir / "final_full82_h100.csv",
        test_ids,
        y_test,
        best_pred,
        best_prob,
        best_model.classes_ if hasattr(best_model, "classes_") else LABELS,
    )


    print("[3/8] Endpoint feature ablation")
    ablation_specs = {
        "Snapshot_final_value": fg["snapshot"],
        "Statistical_summary_28": stat_features,
        "Full_temporal_82": full_features,
    }
    ablation_rows = []
    for name, cols in ablation_specs.items():
        Xa_tr, ya_tr, _ = load_table(args.features_dir / "features_train_h100.csv", cols)
        Xa_te, ya_te, _ = load_table(args.features_dir / "features_test_h100.csv", cols)
        row, _, _, _ = evaluate(best_model, Xa_tr, ya_tr, Xa_te, ya_te, cv)
        ablation_rows.append({"feature_set": name, "n_features": len(cols), **row})
    write_rows(results_dir / "06_feature_ablation_5fold.csv", ablation_rows)


    print("[4/8] Representation x observation-history experiment")
    representations = {
        "Statistical_28": stat_features,
        "Full_temporal_82": full_features,
    }
    horizon_rows = []
    class_rows = []
    prediction_store = {}

    for rep_name, cols in representations.items():
        for h in HORIZONS:
            print(f"  - {rep_name}, {h}%")
            Xtr, ytr, _ = load_table(args.features_dir / f"features_train_h{h}.csv", cols)
            Xte, yte, ids = load_table(args.features_dir / f"features_test_h{h}.csv", cols)
            row, fitted, pred, prob = evaluate(best_model, Xtr, ytr, Xte, yte, cv)
            horizon_rows.append({
                "representation": rep_name,
                "n_features": len(cols),
                "observation_percent": h,
                "observation_steps": manifest["horizons"][str(h)]["steps"],
                "observation_days": manifest["horizons"][str(h)]["days"],
                **row,
            })
            class_rows.extend(classwise_rows(yte, pred, rep_name, h))
            prediction_store[(rep_name, h)] = {
                "y": yte.copy(),
                "pred": pred.copy(),
                "prob": None if prob is None else prob.copy(),
            }
            save_prediction_table(
                predictions_dir / f"pred_{rep_name}_h{h}.csv",
                ids,
                yte,
                pred,
                prob,
                fitted.classes_ if hasattr(fitted, "classes_") else LABELS,
            )

    horizon_df = pd.DataFrame(horizon_rows)
    class_df = pd.DataFrame(class_rows)
    horizon_df.to_csv(results_dir / "07_horizon_representation_5fold.csv", index=False)
    class_df.to_csv(results_dir / "08_classwise_by_horizon.csv", index=False)
    plot_horizon_representation(horizon_df, figures_dir / "horizon_representation.png")
    plot_classwise(class_df, "Full_temporal_82", figures_dir / "classwise_full_temporal.png")
    plot_classwise(class_df, "Statistical_28", figures_dir / "classwise_statistical.png")


    print("[5/8] Paired bootstrap comparisons")
    h_boot_rows = []
    for rep_name in representations:
        d75 = prediction_store[(rep_name, 75)]
        d100 = prediction_store[(rep_name, 100)]
        if not np.array_equal(d75["y"], d100["y"]):
            raise RuntimeError("Test order differs across horizons; paired comparison invalid")
        rows = paired_bootstrap_difference(
            d100["y"], d75["pred"], d100["pred"],
            f"{rep_name}_75pct", f"{rep_name}_100pct",
            args.bootstrap, rng,
        )
        for row in rows:
            row["representation"] = rep_name
            row["retention_75_over_100"] = (
                row["score_a"] / row["score_b"] if row["metric"] == "macro_f1" and row["score_b"] else np.nan
            )
            h_boot_rows.append(row)
    h_boot_df = pd.DataFrame(h_boot_rows)
    h_boot_df.to_csv(results_dir / "09_paired_bootstrap_75_vs_100.csv", index=False)


    rep_boot_rows = []
    for h in HORIZONS:
        stat = prediction_store[("Statistical_28", h)]
        full = prediction_store[("Full_temporal_82", h)]
        if not np.array_equal(stat["y"], full["y"]):
            raise RuntimeError("Test order differs across representations; paired comparison invalid")
        rows = paired_bootstrap_difference(
            full["y"], stat["pred"], full["pred"],
            f"Statistical_28_h{h}", f"Full_temporal_82_h{h}",
            args.bootstrap, rng,
        )
        for row in rows:
            row["observation_percent"] = h
            rep_boot_rows.append(row)
    rep_boot_df = pd.DataFrame(rep_boot_rows)
    rep_boot_df.to_csv(results_dir / "10_paired_bootstrap_representation_by_horizon.csv", index=False)
    plot_temporal_advantage(rep_boot_df, figures_dir / "temporal_advantage_by_horizon.png")


    print("[6/8] Class-weight sensitivity")
    imbalance_rows = []
    for strategy in ["none", "balanced"]:
        try:
            m = set_class_weight(best_model, None if strategy == "none" else "balanced")
        except ValueError:
            continue
        row, _, _, _ = evaluate(m, X_train, y_train, X_test, y_test, cv)
        imbalance_rows.append({"class_weight_strategy": strategy, **row})
    write_rows(results_dir / "11_imbalance_strategy_5fold.csv", imbalance_rows)


    print("[7/8] Confidence-based selective diagnosis")
    selective = []
    for rep_name in representations:
        for h in [75, 100]:
            d = prediction_store[(rep_name, h)]
            if d["prob"] is None:
                continue
            selective.extend(selective_rows(d["y"], d["pred"], d["prob"], rep_name, h))
    selective_df = pd.DataFrame(selective)
    selective_df.to_csv(results_dir / "12_selective_diagnosis.csv", index=False)
    if not selective_df.empty:
        plot_selective(selective_df, "Full_temporal_82", 75, figures_dir / "selective_diagnosis_full82_h75.png")
        plot_selective(selective_df, "Full_temporal_82", 100, figures_dir / "selective_diagnosis_full82_h100.png")


    print("[8/8] Reproducibility metadata")
    fingerprint_rows = []
    paths = [args.features_dir / "feature_manifest.json"]
    for h in HORIZONS:
        paths += [
            args.features_dir / f"features_train_h{h}.csv",
            args.features_dir / f"features_test_h{h}.csv",
        ]
    for path in paths:
        fingerprint_rows.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    pd.DataFrame(fingerprint_rows).to_csv(results_dir / "13_input_fingerprints.csv", index=False)

    summary = {
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "tuning_iterations": args.tuning_iterations,
        "paired_bootstrap_samples": args.bootstrap,
        "primary_metric": "macro-F1",
        "model_selection": "highest 5-fold TRAIN-CV macro-F1 among explainable candidates; held-out TEST metrics excluded from selection",
        "selected_model": selected_name,
        "best_cv_macro_f1": float(search.best_score_),
        "final_full82_h100_test_macro_f1": float(final_metrics[0]["test_macro_f1"]),
        "final_full82_h100_test_accuracy": float(final_metrics[0]["test_accuracy"]),
        "best_params": search.best_params_,
        "smoke_mode": bool(args.smoke),
        "notes": [
            "The supplied train/test split is preserved exactly.",
            "All hyperparameter tuning occurs on the training split via stratified 5-fold CV.",
            "Observation-horizon comparisons use prefix-only feature tables.",
            "Paired bootstrap comparisons resample the same test cases.",
            "Selective diagnosis uses native model confidence and makes no probability-calibration claim.",
        ],
    }
    (results_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


    lines = [
        "DGA OBSERVATION-EFFICIENT FDD — FINAL 5-FOLD RUN",
        "=" * 60,
        f"Selected model: {selected_name}",
        f"Best 5-fold TRAIN-CV macro-F1: {search.best_score_:.6f}",
        f"Full-82 / 100% TEST macro-F1: {final_metrics[0]['test_macro_f1']:.6f}",
        f"Full-82 / 100% TEST accuracy: {final_metrics[0]['test_accuracy']:.6f}",
        "",
        "Results are saved in this output directory.",
    ]
    (args.output_dir / "READ_ME_FIRST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    zip_base = args.output_dir.parent / args.output_dir.name
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=args.output_dir)
    print(f"Output ZIP: {zip_path}")


if __name__ == "__main__":
    main()
