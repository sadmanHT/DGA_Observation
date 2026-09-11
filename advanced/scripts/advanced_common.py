from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
LABELS = np.array([1, 2, 3, 4], dtype=int)
LABEL_NAMES = {
    1: "Normal mode",
    2: "Partial discharge",
    3: "Low energy discharge",
    4: "Low-temperature overheating",
}
GASES = ("H2", "CO", "C2H4", "C2H2")
STAT_NAMES = ("mean", "std", "min", "max", "median", "q25", "q75")
DENSE_HORIZONS = list(range(10, 101, 5))


def fixed_lr() -> Pipeline:
    """Canonical classifier configuration from the final 5-fold paper.

    The original canonical run was serialized by scikit-learn 1.9.0.  Keep C and
    class weighting fixed for all strengthening analyses so the new experiments
    study observation/representation effects rather than re-selecting a model on TEST.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=30.0,
            class_weight=None,
            max_iter=5000,
            tol=1e-6,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )),
    ])


def load_table(path: Path, cols: list[str]):
    df = pd.read_csv(path)
    X = df[cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    ids = df["id"].astype(str).to_numpy()
    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite values in {path}")
    return X, y, ids, df


def feature_groups_from_columns(columns: list[str]) -> dict[str, list[str]]:
    cols = set(columns)
    groups: dict[str, list[str]] = {}
    groups["statistical"] = [f"{g}_{s}" for g in GASES for s in STAT_NAMES if f"{g}_{s}" in cols]
    groups["endpoint_change"] = [f"{g}_{s}" for g in GASES for s in ("first", "last", "delta") if f"{g}_{s}" in cols]
    groups["trend"] = [f"{g}_{s}" for g in GASES for s in ("slope", "trend_r2") if f"{g}_{s}" in cols]
    groups["local_dynamics"] = [f"{g}_{s}" for g in GASES for s in ("mean_diff", "std_diff", "max_abs_diff", "positive_diff_fraction") if f"{g}_{s}" in cols]
    groups["phase_shift"] = [f"{g}_{s}" for g in GASES for s in ("early_mean", "late_mean", "late_minus_early") if f"{g}_{s}" in cols]
    groups["cross_gas_correlation"] = sorted([c for c in columns if c.startswith("corr_")])
    groups["full"] = [c for c in columns if c not in {"id", "split", "label", "label_name", "observation_fraction", "observation_steps", "observation_days"}]
    return groups


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
    }


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return [
        {
            "class_label": int(cls),
            "class_name": LABEL_NAMES[int(cls)],
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(n[i]),
        }
        for i, cls in enumerate(LABELS)
    ]


def environment_info() -> dict:
    try:
        import shap
        shap_ver = shap.__version__
    except Exception:
        shap_ver = None
    try:
        import matplotlib
        mpl_ver = matplotlib.__version__
    except Exception:
        mpl_ver = None
    return {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
        "shap": shap_ver,
        "matplotlib": mpl_ver,
    }


def save_environment(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(environment_info(), indent=2), encoding="utf-8")


def _macro_f1_fast(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 over the fixed four labels using a 4x4 confusion matrix.

    This is numerically equivalent to sklearn's macro F1 for this project but avoids
    metric-construction overhead inside thousands of bootstrap iterations.
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    cat = (yt - 1) * 4 + (yp - 1)
    cm = np.bincount(cat, minlength=16).reshape(4, 4)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    den = 2.0 * tp + fp + fn
    per_class = np.divide(2.0 * tp, den, out=np.zeros(4, dtype=float), where=den > 0)
    return float(per_class.mean())


def paired_macro_f1_bootstrap(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = 5000, seed: int = RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)
    n = len(y)
    point_a = _macro_f1_fast(y, pred_a)
    point_b = _macro_f1_fast(y, pred_b)
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = _macro_f1_fast(y[idx], pred_b[idx]) - _macro_f1_fast(y[idx], pred_a[idx])
    return {
        "score_a": float(point_a),
        "score_b": float(point_b),
        "difference_b_minus_a": float(point_b - point_a),
        "ci95_low": float(np.quantile(diffs, .025)),
        "ci95_high": float(np.quantile(diffs, .975)),
        "bootstrap_samples": int(n_boot),
    }
