from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
LABEL_NAMES = {
    1: "Normal mode",
    2: "Partial discharge",
    3: "Low energy discharge",
    4: "Low-temperature overheating",
}


def load_table(path: Path, feature_cols: list[str]):
    X, y, ids = [], [], []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row[c]) for c in feature_cols])
            y.append(int(row["label"]))
            ids.append(row["id"])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), ids


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def normalize_shap(values, n_samples: int, n_features: int, n_classes: int) -> np.ndarray:

    if isinstance(values, list):
        arr = np.stack(values, axis=-1)
    else:
        arr = np.asarray(values)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        elif arr.ndim == 3:
            if arr.shape == (n_classes, n_samples, n_features):
                arr = np.transpose(arr, (1, 2, 0))
            elif arr.shape == (n_samples, n_classes, n_features):
                arr = np.transpose(arr, (0, 2, 1))
    if arr.shape[0] != n_samples or arr.shape[1] != n_features:
        raise ValueError(f"Unexpected SHAP shape: {arr.shape}")
    return arr


def bar_plot(names, vals, title, out_path, top_n=20):
    order = np.argsort(vals)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(np.asarray(names)[order], np.asarray(vals)[order])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_explainer(model, X_background):
    if isinstance(model, Pipeline) and isinstance(model.named_steps.get("model"), LogisticRegression):
        scaler = model.named_steps["scale"]
        linear = model.named_steps["model"]
        bg = scaler.transform(X_background)
        return shap.LinearExplainer(linear, bg), scaler.transform, linear.classes_
    return shap.TreeExplainer(model), (lambda x: x), model.classes_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--max-test-samples", type=int, default=500)
    ap.add_argument("--background-samples", type=int, default=200)
    args = ap.parse_args()

    results_dir = args.run_dir / "results"
    figures_dir = args.run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(results_dir / "best_model.joblib")
    features = json.loads((results_dir / "best_model_features.json").read_text(encoding="utf-8"))
    X_train, _, _ = load_table(args.features_dir / "features_train_h100.csv", features)
    X_test, y_test, ids = load_table(args.features_dir / "features_test_h100.csv", features)

    rng = np.random.default_rng(RANDOM_STATE)
    bg_n = min(args.background_samples, len(X_train))
    bg_idx = rng.choice(np.arange(len(X_train)), size=bg_n, replace=False)
    X_background = X_train[bg_idx]

    selected = []
    per_class = max(1, args.max_test_samples // 4)
    for cls in [1, 2, 3, 4]:
        idx = np.where(y_test == cls)[0]
        take = min(len(idx), per_class)
        selected.extend(rng.choice(idx, size=take, replace=False).tolist())
    selected = np.asarray(sorted(selected), dtype=int)
    Xs = X_test[selected]

    explainer, transform, classes = make_explainer(model, X_background)
    values = explainer.shap_values(transform(Xs))
    arr = normalize_shap(values, len(Xs), len(features), len(classes))

    global_imp = np.mean(np.abs(arr), axis=(0, 2))
    global_rows = [
        {"feature": f, "mean_abs_shap": float(v)}
        for f, v in sorted(zip(features, global_imp), key=lambda z: z[1], reverse=True)
    ]
    write_rows(results_dir / "14_shap_global_importance.csv", global_rows)
    bar_plot(features, global_imp, "Global Feature Importance (SHAP)", figures_dir / "shap_global_importance.png")

    class_rows = []
    for ci, cls in enumerate(classes):
        vals = np.mean(np.abs(arr[:, :, ci]), axis=0)
        for f, v in sorted(zip(features, vals), key=lambda z: z[1], reverse=True):
            class_rows.append({
                "class": int(cls),
                "class_name": LABEL_NAMES[int(cls)],
                "feature": f,
                "mean_abs_shap": float(v),
            })
        bar_plot(features, vals, f"SHAP Importance — {LABEL_NAMES[int(cls)]}", figures_dir / f"shap_class_{int(cls)}.png")
    write_rows(results_dir / "15_shap_class_importance.csv", class_rows)

    pred = model.predict(X_test)
    local_rows = []
    for cls in [1, 2, 3, 4]:
        candidates = np.where((y_test == cls) & (pred == cls))[0]
        if len(candidates) == 0:
            continue
        oi = int(candidates[0])
        lv = explainer.shap_values(transform(X_test[oi:oi + 1]))
        local_arr = normalize_shap(lv, 1, len(features), len(classes))[0]
        ci = list(classes).index(cls)
        sv = local_arr[:, ci]
        top = np.argsort(np.abs(sv))[-10:][::-1]
        for rank, fi in enumerate(top, start=1):
            local_rows.append({
                "id": ids[oi],
                "class": cls,
                "class_name": LABEL_NAMES[cls],
                "rank": rank,
                "feature": features[fi],
                "feature_value": float(X_test[oi, fi]),
                "shap_value": float(sv[fi]),
            })
    write_rows(results_dir / "16_shap_local_examples.csv", local_rows)
    print("SHAP outputs complete.")


if __name__ == "__main__":
    main()
