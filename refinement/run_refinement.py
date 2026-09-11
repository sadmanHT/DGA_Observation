#!/usr/bin/env python3
"""Final compact robustness/refinement experiment used by the updated paper.

This is the script form of the successful Kaggle notebook preserved under
notebooks/DGA_Final_Compact_Robustness_Refinement_Kaggle.ipynb.

Selection discipline:
- LR hyperparameters: training-only stratified 5-fold CV.
- Robust smoothing window: training OOF only under predeclared stress.
- Held-out TEST never selects C, class weight, feature family, or smoothing window.
- TEST noise seeds are disjoint from training-side selection seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix,
)

REQUIRED_SKLEARN = "1.9.0"
SEED = 42
N_SPLITS = 5
NOISE_SELECTION_LEVELS = [0.02, 0.05]
NOISE_SELECTION_REPEATS = 3
NOISE_TEST_LEVELS = [0.02, 0.05, 0.10]
CLEAN_RETENTION_CONSTRAINT = 0.98
SMOOTHING_WINDOWS = [1, 3, 5, 7, 11]


def extract_archive(archive: Path, work: Path) -> Path:
    """Extract the raw Kaggle ZIP and return the directory containing label CSVs."""
    raw_extract = work / "_raw_extract"
    if raw_extract.exists():
        shutil.rmtree(raw_extract)
    raw_extract.mkdir(parents=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_extract)
    hits = list(raw_extract.rglob("labels_fdd_train.csv"))
    if not hits:
        raise FileNotFoundError("labels_fdd_train.csv not found after extracting archive")
    return hits[0].parent


ap = argparse.ArgumentParser()
ap.add_argument("--archive", type=Path, required=True, help="Original Power Transformers FDD/RUL ZIP")
ap.add_argument("--canonical-final", type=Path, required=True, help="Canonical run directory containing features/")
ap.add_argument("--work-dir", type=Path, default=Path("runs/refinement"))
ap.add_argument("--bootstrap", type=int, default=5000)
ap.add_argument("--noise-test-repeats", type=int, default=20)
ap.add_argument("--zip-output", type=Path, default=None)
args = ap.parse_args()

if sklearn.__version__ != REQUIRED_SKLEARN:
    raise SystemExit(
        f"Refinement requires scikit-learn {REQUIRED_SKLEARN}; found {sklearn.__version__}. "
        "Install requirements/reproducible.txt."
    )

RAW_ARCHIVE = args.archive.resolve()
CANON_ROOT = args.canonical_final.resolve()
WORK = args.work_dir.resolve()
OUT = WORK / "outputs"
BOOTSTRAP = args.bootstrap
NOISE_TEST_REPEATS = args.noise_test_repeats

if WORK.exists():
    shutil.rmtree(WORK)
OUT.mkdir(parents=True, exist_ok=True)
RAW_ROOT = extract_archive(RAW_ARCHIVE, WORK)

print("Python:", sys.version)
print("scikit-learn:", sklearn.__version__)
print("Raw root:", RAW_ROOT)
print("Canonical root:", CANON_ROOT)
print("Output:", OUT)


# 3. Resolve canonical feature tables and raw dataset structure

def find_required(root: Path, relative_candidates):
    for rel in relative_candidates:
        p = root / rel
        if p.exists():
            return p
    # Nested folder fallback
    target_name = Path(relative_candidates[0]).name
    hits = list(root.rglob(target_name))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Could not find any of {relative_candidates} under {root}")

TRAIN75 = find_required(CANON_ROOT, ["features/features_train_h75.csv"])
TEST75  = find_required(CANON_ROOT, ["features/features_test_h75.csv"])
TRAIN100 = find_required(CANON_ROOT, ["features/features_train_h100.csv"])
TEST100  = find_required(CANON_ROOT, ["features/features_test_h100.csv"])

LABEL_TRAIN = find_required(RAW_ROOT, ["labels_fdd_train.csv"])
LABEL_TEST  = find_required(RAW_ROOT, ["labels_fdd_test.csv"])

# Find folders by one known file relationship.
DATA_TRAIN = LABEL_TRAIN.parent / "data_train"
DATA_TEST = LABEL_TEST.parent / "data_test"
assert DATA_TRAIN.is_dir()
assert DATA_TEST.is_dir()

print("Canonical feature tables:")
for p in [TRAIN75, TEST75, TRAIN100, TEST100]:
    print(" ", p)
print("Raw:")
print(" ", LABEL_TRAIN)
print(" ", LABEL_TEST)
print(" ", DATA_TRAIN)
print(" ", DATA_TEST)

tr75 = pd.read_csv(TRAIN75)
te75 = pd.read_csv(TEST75)
tr100 = pd.read_csv(TRAIN100)
te100 = pd.read_csv(TEST100)

print("Shapes:", tr75.shape, te75.shape, tr100.shape, te100.shape)
assert len(tr75) == 2100 and len(te75) == 900

META = {
    "id", "split", "label", "label_name",
    "observation_fraction", "observation_steps", "observation_days"
}
FEATURE82 = [c for c in tr75.columns if c not in META]
assert len(FEATURE82) == 82, len(FEATURE82)
print("Temporal feature count:", len(FEATURE82))

# 4. Feature families and shared helpers

GASES = ["H2", "CO", "C2H4", "C2H2"]

STAT28 = [
    f"{g}_{s}"
    for g in GASES
    for s in ["mean", "std", "min", "max", "median", "q25", "q75"]
]

# Nonredundant temporal additions.
# Deliberately omitted exact algebraic redundancies:
#   delta = last - first
#   mean_diff = delta / (N - 1)
#   late_minus_early = late_mean - early_mean
ENDPOINT = [f"{g}_{s}" for g in GASES for s in ["first", "last"]]
TREND = [f"{g}_{s}" for g in GASES for s in ["slope", "trend_r2"]]
ROUGHNESS = [f"{g}_{s}" for g in GASES for s in [
    "std_diff", "max_abs_diff", "positive_diff_fraction"
]]
SEGMENT = [f"{g}_{s}" for g in GASES for s in ["early_mean", "late_mean"]]
CORR = [c for c in FEATURE82 if c.startswith("corr_")]

FAMILIES = {
    "endpoint_level": ENDPOINT,
    "trend": TREND,
    "local_roughness": ROUGHNESS,
    "segment_level": SEGMENT,
    "cross_gas_correlation": CORR,
}
TEMP70 = STAT28 + ENDPOINT + TREND + ROUGHNESS + SEGMENT + CORR

assert len(STAT28) == 28
assert len(ENDPOINT) == 8
assert len(TREND) == 8
assert len(ROUGHNESS) == 12
assert len(SEGMENT) == 8
assert len(CORR) == 6
assert len(TEMP70) == 70
assert len(set(TEMP70)) == 70
assert set(TEMP70).issubset(FEATURE82)

CV = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

def make_lr(C=30.0, class_weight=None, tol=1e-6):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=float(C),
            class_weight=class_weight,
            solver="lbfgs",
            max_iter=5000,
            tol=float(tol),
            random_state=SEED,
        )),
    ])

def metrics(y, pred):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
    }

def paired_bootstrap_delta(y, pred_a, pred_b, n=BOOTSTRAP, seed=SEED):
    # Delta = A - B
    y = np.asarray(y)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    N = len(y)
    vals = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, N, N)
        vals[i] = (
            f1_score(y[idx], a[idx], average="macro")
            - f1_score(y[idx], b[idx], average="macro")
        )
    return {
        "delta": float(f1_score(y, a, average="macro") - f1_score(y, b, average="macro")),
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
    }

print("Feature families:")
for k, v in FAMILIES.items():
    print(k, len(v))
print("Nonredundant Temporal-70:", len(TEMP70))

# 5. Exact 18-configuration training-only grid under tol=1e-6

X100 = tr100[FEATURE82].to_numpy(float)
y100 = tr100["label"].to_numpy()
X100_test = te100[FEATURE82].to_numpy(float)
y100_test = te100["label"].to_numpy()

base = make_lr(C=1.0, class_weight=None, tol=1e-6)
param_grid = {
    "lr__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
    "lr__class_weight": [None, "balanced"],
}

search = GridSearchCV(
    base,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=CV,
    n_jobs=-1,
    refit=True,
    return_train_score=False,
)
t0 = time.time()
search.fit(X100, y100)
print("Grid seconds:", round(time.time() - t0, 2))

grid = pd.DataFrame(search.cv_results_)
grid_out = grid[[
    "param_lr__C", "param_lr__class_weight",
    "mean_test_score", "std_test_score", "rank_test_score"
]].copy()
grid_out = grid_out.sort_values(["rank_test_score", "param_lr__C"])
grid_out.to_csv(OUT / "01_tol1e6_training_grid.csv", index=False)

BEST_C = float(search.best_params_["lr__C"])
BEST_WEIGHT = search.best_params_["lr__class_weight"]
BEST_CV = float(search.best_score_)

pred_test = search.best_estimator_.predict(X100_test)
best_test = metrics(y100_test, pred_test)

selection = {
    "tol": 1e-6,
    "C": BEST_C,
    "class_weight": BEST_WEIGHT,
    "training_cv_macro_f1": BEST_CV,
    "test_metrics_descriptive_only_after_selection": best_test,
}
(OUT / "01_selected_model.json").write_text(json.dumps(selection, indent=2))

print(grid_out.to_string(index=False))
print("\nSelected from TRAINING CV only:")
print(json.dumps(selection, indent=2))

# 6. Redundancy-aware ablation at 75%

representations = {
    "Statistical-28": STAT28,
    "Stat28 + endpoint": STAT28 + ENDPOINT,
    "Stat28 + trend": STAT28 + TREND,
    "Stat28 + local roughness": STAT28 + ROUGHNESS,
    "Stat28 + segment level": STAT28 + SEGMENT,
    "Stat28 + correlations": STAT28 + CORR,
    "Temporal-70 nonredundant": TEMP70,
    "Temporal-82 original": FEATURE82,
}
for fam_name, fam_cols in FAMILIES.items():
    representations[f"T70 minus {fam_name}"] = [c for c in TEMP70 if c not in set(fam_cols)]

ytr = tr75["label"].to_numpy()
yte = te75["label"].to_numpy()

ablation_rows = []
test_predictions = {}

for name, cols in representations.items():
    Xtr = tr75[cols].to_numpy(float)
    Xte = te75[cols].to_numpy(float)
    model = make_lr(BEST_C, BEST_WEIGHT, tol=1e-6)

    cv_scores = cross_val_score(
        model, Xtr, ytr,
        cv=CV, scoring="f1_macro", n_jobs=-1
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    test_predictions[name] = pred

    m = metrics(yte, pred)
    scaled = StandardScaler().fit_transform(Xtr)
    rank = int(np.linalg.matrix_rank(scaled))

    ablation_rows.append({
        "representation": name,
        "n_features": len(cols),
        "training_cv_macro_f1_mean": float(cv_scores.mean()),
        "training_cv_macro_f1_std": float(cv_scores.std(ddof=1)),
        "test_macro_f1": m["macro_f1"],
        "test_balanced_accuracy": m["balanced_accuracy"],
        "test_accuracy": m["accuracy"],
        "training_matrix_rank_after_scaling": rank,
    })
    print(name, "CV", round(cv_scores.mean(), 5), "TEST", round(m["macro_f1"], 5), "rank", rank)

abl = pd.DataFrame(ablation_rows)

# Paired TEST uncertainty versus Statistical-28.
base_pred = test_predictions["Statistical-28"]
boot_rows = []
for name, pred in test_predictions.items():
    if name == "Statistical-28":
        continue
    b = paired_bootstrap_delta(yte, pred, base_pred, n=BOOTSTRAP, seed=SEED + 71)
    boot_rows.append({"representation": name, **b})
boot = pd.DataFrame(boot_rows)

abl = abl.merge(boot, on="representation", how="left")
abl.to_csv(OUT / "02_redundancy_aware_ablation.csv", index=False)

pred_df = pd.DataFrame({"id": te75["id"], "label": yte})
for name, pred in test_predictions.items():
    pred_df["pred_" + name.replace(" ", "_").replace("+", "plus").replace("-", "_")] = pred
pred_df.to_csv(OUT / "02_ablation_test_predictions.csv", index=False)

print("\nSorted by TRAINING CV:")
print(abl.sort_values("training_cv_macro_f1_mean", ascending=False).to_string(index=False))

# 7. Raw loading + exact vectorized feature extraction

EPS = 1e-12

def load_raw_in_feature_order(feature_df, data_dir, labels_csv, steps=315):
    labels = pd.read_csv(labels_csv).set_index("id")["category"].to_dict()
    raws = []
    ys = []
    for fid, expected_label in zip(feature_df["id"], feature_df["label"]):
        p = data_dir / fid
        arr = pd.read_csv(p)[GASES].to_numpy(float)
        assert arr.shape[0] >= steps
        lab = int(labels[fid])
        assert lab == int(expected_label), (fid, lab, expected_label)
        raws.append(arr[:steps])
        ys.append(lab)
    return np.stack(raws), np.asarray(ys)

raw_train, raw_y = load_raw_in_feature_order(tr75, DATA_TRAIN, LABEL_TRAIN, steps=315)
raw_test, test_y = load_raw_in_feature_order(te75, DATA_TEST, LABEL_TEST, steps=315)
assert np.array_equal(raw_y, ytr)
assert np.array_equal(test_y, yte)

print("Raw train:", raw_train.shape)
print("Raw test :", raw_test.shape)

def causal_moving_average(x, window):
    x = np.asarray(x, float)
    if int(window) <= 1:
        return x.copy()
    w = int(window)
    cs = np.cumsum(x, axis=1)
    out = np.empty_like(x)
    for t in range(x.shape[1]):
        start = t - w + 1
        if start <= 0:
            out[:, t, :] = cs[:, t, :] / float(t + 1)
        else:
            out[:, t, :] = (cs[:, t, :] - cs[:, start - 1, :]) / float(w)
    return out

def all_feature_dict(x):
    x = np.asarray(x, float)
    ncase, nt, ngas = x.shape
    assert ngas == 4
    d = np.diff(x, axis=1)

    q = np.percentile(x, [25, 50, 75], axis=1)
    mean = x.mean(axis=1)
    std = x.std(axis=1, ddof=1)
    xmin = x.min(axis=1)
    xmax = x.max(axis=1)
    first = x[:, 0, :]
    last = x[:, -1, :]
    delta = last - first

    t = np.arange(nt, dtype=float)
    tc = t - t.mean()
    denom = float(np.dot(tc, tc))
    xc = x - mean[:, None, :]
    slope = np.einsum("t,ntg->ng", tc, xc) / denom
    fitted = mean[:, None, :] + slope[:, None, :] * tc[None, :, None]
    ss_res = np.sum((x - fitted) ** 2, axis=1)
    ss_tot = np.sum(xc ** 2, axis=1)
    r2 = np.divide(
        1.0 * (ss_tot - ss_res), ss_tot,
        out=np.zeros_like(ss_tot), where=ss_tot > EPS
    )

    edge = max(5, int(round(0.20 * nt)))
    early = x[:, :edge, :].mean(axis=1)
    late = x[:, -edge:, :].mean(axis=1)

    vals = {}
    for j, g in enumerate(GASES):
        vals[f"{g}_first"] = first[:, j]
        vals[f"{g}_last"] = last[:, j]
        vals[f"{g}_mean"] = mean[:, j]
        vals[f"{g}_std"] = std[:, j]
        vals[f"{g}_min"] = xmin[:, j]
        vals[f"{g}_max"] = xmax[:, j]
        vals[f"{g}_median"] = q[1, :, j]
        vals[f"{g}_q25"] = q[0, :, j]
        vals[f"{g}_q75"] = q[2, :, j]
        vals[f"{g}_delta"] = delta[:, j]
        vals[f"{g}_slope"] = slope[:, j]
        vals[f"{g}_trend_r2"] = r2[:, j]
        vals[f"{g}_mean_diff"] = d[:, :, j].mean(axis=1)
        vals[f"{g}_std_diff"] = d[:, :, j].std(axis=1, ddof=1)
        vals[f"{g}_max_abs_diff"] = np.abs(d[:, :, j]).max(axis=1)
        vals[f"{g}_early_mean"] = early[:, j]
        vals[f"{g}_late_mean"] = late[:, j]
        vals[f"{g}_late_minus_early"] = late[:, j] - early[:, j]
        vals[f"{g}_positive_diff_fraction"] = (d[:, :, j] > 0).mean(axis=1)

    for a in range(4):
        for b in range(a + 1, 4):
            xa = x[:, :, a]
            xb = x[:, :, b]
            ac = xa - xa.mean(axis=1, keepdims=True)
            bc = xb - xb.mean(axis=1, keepdims=True)
            den = np.sqrt(np.sum(ac * ac, axis=1) * np.sum(bc * bc, axis=1))
            corr = np.divide(
                np.sum(ac * bc, axis=1), den,
                out=np.zeros(ncase), where=den > EPS
            )
            vals[f"corr_{GASES[a]}_{GASES[b]}"] = corr
    return vals

def matrix_from_dict(vals, cols):
    return np.column_stack([vals[c] for c in cols]).astype(float)

def extract82(x):
    return matrix_from_dict(all_feature_dict(x), FEATURE82)

def extract70_hybrid(raw_x, smoothing_window):
    # Statistical levels/distributions stay on raw observations.
    raw_vals = all_feature_dict(raw_x)
    smooth_x = causal_moving_average(raw_x, smoothing_window)
    dyn_vals = all_feature_dict(smooth_x)

    cols = []
    arrays = []
    stat_set = set(STAT28)
    for c in TEMP70:
        if c in stat_set:
            arrays.append(raw_vals[c])
        else:
            arrays.append(dyn_vals[c])
        cols.append(c)
    assert cols == TEMP70
    return np.column_stack(arrays).astype(float)

# Validate exact clean extraction against canonical CSVs.
x82_check = extract82(raw_test[:20])
x82_canon = te75.loc[:19, FEATURE82].to_numpy(float)
max_abs82 = float(np.max(np.abs(x82_check - x82_canon)))
print("82-feature extractor max abs difference:", max_abs82)
assert np.allclose(x82_check, x82_canon, rtol=1e-9, atol=1e-10)

x70_check = extract70_hybrid(raw_test[:20], 1)
x70_canon = te75.loc[:19, TEMP70].to_numpy(float)
max_abs70 = float(np.max(np.abs(x70_check - x70_canon)))
print("70-feature extractor max abs difference:", max_abs70)
assert np.allclose(x70_check, x70_canon, rtol=1e-9, atol=1e-10)

validation = {"max_abs_82": max_abs82, "max_abs_70": max_abs70}
(OUT / "03_feature_extractor_validation.json").write_text(json.dumps(validation, indent=2))

# 8. Precompute clean candidate features and training-only noisy stress features

def add_relative_gaussian_noise(raw, severity, seed):
    rng = np.random.default_rng(seed)
    damaged = raw * (1.0 + rng.normal(0.0, severity, size=raw.shape))
    return np.maximum(damaged, 0.0)

clean70_by_w = {}
for w in SMOOTHING_WINDOWS:
    print("Clean feature candidate window", w)
    clean70_by_w[w] = extract70_hybrid(raw_train, w)

# Use exactly the same noisy raw realization across all smoothing candidates.
noisy_train_features = {}
for severity in NOISE_SELECTION_LEVELS:
    for rep in range(NOISE_SELECTION_REPEATS):
        seed = 100000 + int(severity * 10000) + rep
        print("Training stress", severity, "rep", rep, "seed", seed)
        damaged = add_relative_gaussian_noise(raw_train, severity, seed)
        for w in SMOOTHING_WINDOWS:
            noisy_train_features[(severity, rep, w)] = extract70_hybrid(damaged, w)

print("Precomputation complete.")

# 9. OOF robust-window selection using TRAINING data only

candidate_rows = []
oof_detail_rows = []

for w in SMOOTHING_WINDOWS:
    clean_oof = np.empty_like(raw_y)
    stress_oof = {
        (sev, rep): np.empty_like(raw_y)
        for sev in NOISE_SELECTION_LEVELS
        for rep in range(NOISE_SELECTION_REPEATS)
    }

    Xclean = clean70_by_w[w]

    for fold, (fit_idx, val_idx) in enumerate(CV.split(Xclean, raw_y), start=1):
        model = make_lr(BEST_C, BEST_WEIGHT, tol=1e-6)
        model.fit(Xclean[fit_idx], raw_y[fit_idx])

        clean_oof[val_idx] = model.predict(Xclean[val_idx])

        for sev in NOISE_SELECTION_LEVELS:
            for rep in range(NOISE_SELECTION_REPEATS):
                Xstress = noisy_train_features[(sev, rep, w)]
                stress_oof[(sev, rep)][val_idx] = model.predict(Xstress[val_idx])

    clean_f1 = float(f1_score(raw_y, clean_oof, average="macro"))
    stress_scores = []
    for sev in NOISE_SELECTION_LEVELS:
        for rep in range(NOISE_SELECTION_REPEATS):
            s = float(f1_score(raw_y, stress_oof[(sev, rep)], average="macro"))
            stress_scores.append(s)
            oof_detail_rows.append({
                "window": w,
                "severity": sev,
                "repeat": rep,
                "macro_f1": s,
            })

    candidate_rows.append({
        "window": w,
        "clean_oof_macro_f1": clean_f1,
        "stressed_oof_macro_f1_mean": float(np.mean(stress_scores)),
        "stressed_oof_macro_f1_std": float(np.std(stress_scores, ddof=1)),
        "stressed_oof_macro_f1_min": float(np.min(stress_scores)),
        "stressed_oof_macro_f1_max": float(np.max(stress_scores)),
    })

cand = pd.DataFrame(candidate_rows)
baseline_clean = float(cand.loc[cand["window"] == 1, "clean_oof_macro_f1"].iloc[0])
threshold = CLEAN_RETENTION_CONSTRAINT * baseline_clean
cand["clean_retention_vs_unsmoothed"] = cand["clean_oof_macro_f1"] / baseline_clean
cand["eligible"] = cand["clean_oof_macro_f1"] >= threshold

eligible = cand[cand["eligible"]].copy()
assert len(eligible) > 0

# Predeclared ordering:
# 1) highest stressed OOF mean
# 2) highest clean OOF
# 3) smaller smoothing window
eligible = eligible.sort_values(
    ["stressed_oof_macro_f1_mean", "clean_oof_macro_f1", "window"],
    ascending=[False, False, True],
)

SELECTED_WINDOW = int(eligible.iloc[0]["window"])

cand.to_csv(OUT / "03_noise_robust_window_selection.csv", index=False)
pd.DataFrame(oof_detail_rows).to_csv(OUT / "03_noise_selection_oof_replicates.csv", index=False)

selection_noise = {
    "selection_data": "training OOF only",
    "noise_model": "independent multiplicative Gaussian, clipped nonnegative",
    "selection_noise_levels": NOISE_SELECTION_LEVELS,
    "selection_repeats_per_level": NOISE_SELECTION_REPEATS,
    "clean_retention_constraint": CLEAN_RETENTION_CONSTRAINT,
    "unsmoothed_clean_oof_macro_f1": baseline_clean,
    "minimum_eligible_clean_oof_macro_f1": threshold,
    "selected_causal_moving_average_window": SELECTED_WINDOW,
}
(OUT / "03_noise_robust_selection.json").write_text(json.dumps(selection_noise, indent=2))

print(cand.to_string(index=False))
print("\nSelected robust window:", SELECTED_WINDOW)
print(json.dumps(selection_noise, indent=2))

# 10. Fit clean training models and evaluate clean TEST performance

# Original Temporal-82
X82_train = tr75[FEATURE82].to_numpy(float)
X82_test_clean = te75[FEATURE82].to_numpy(float)
m82 = make_lr(BEST_C, BEST_WEIGHT, tol=1e-6).fit(X82_train, ytr)

# Unsmoothed nonredundant Temporal-70
X70_train = clean70_by_w[1]
X70_test_clean = extract70_hybrid(raw_test, 1)
m70 = make_lr(BEST_C, BEST_WEIGHT, tol=1e-6).fit(X70_train, ytr)

# Training-selected robust Temporal-70
Xrob_train = clean70_by_w[SELECTED_WINDOW]
Xrob_test_clean = extract70_hybrid(raw_test, SELECTED_WINDOW)
mrob = make_lr(BEST_C, BEST_WEIGHT, tol=1e-6).fit(Xrob_train, ytr)

models = {
    "Temporal-82 original": (m82, X82_test_clean),
    "Temporal-70 unsmoothed": (m70, X70_test_clean),
    f"Temporal-70 robust w={SELECTED_WINDOW}": (mrob, Xrob_test_clean),
}

clean_rows = []
clean_preds = {}
for name, (model, Xtest) in models.items():
    pred = model.predict(Xtest)
    clean_preds[name] = pred
    clean_rows.append({"model": name, **metrics(yte, pred)})
clean_df = pd.DataFrame(clean_rows)
clean_df.to_csv(OUT / "04_clean_test_comparison.csv", index=False)

# Pair robust clean prediction against original 82 and unsmoothed 70.
clean_boot = []
rob_name = f"Temporal-70 robust w={SELECTED_WINDOW}"
for base_name in ["Temporal-82 original", "Temporal-70 unsmoothed"]:
    b = paired_bootstrap_delta(
        yte, clean_preds[rob_name], clean_preds[base_name],
        n=BOOTSTRAP, seed=SEED + 901
    )
    clean_boot.append({"comparison": f"{rob_name} minus {base_name}", **b})
pd.DataFrame(clean_boot).to_csv(OUT / "04_clean_paired_bootstrap.csv", index=False)

print("Clean TEST:")
print(clean_df.to_string(index=False))
print(pd.DataFrame(clean_boot).to_string(index=False))

# 11. Independent TEST noise stress: paired realizations for all three models

noise_rows = []
class_rows = []

for severity in NOISE_TEST_LEVELS:
    for rep in range(NOISE_TEST_REPEATS):
        # Deliberately disjoint seed range from training selection.
        seed = 900000 + int(severity * 10000) + rep
        damaged = add_relative_gaussian_noise(raw_test, severity, seed)

        X82_noise = extract82(damaged)
        X70_noise = extract70_hybrid(damaged, 1)
        Xrob_noise = extract70_hybrid(damaged, SELECTED_WINDOW)

        scenario_data = [
            ("Temporal-82 original", m82, X82_noise),
            ("Temporal-70 unsmoothed", m70, X70_noise),
            (rob_name, mrob, Xrob_noise),
        ]

        for name, model, Xnoise in scenario_data:
            pred = model.predict(Xnoise)
            m = metrics(yte, pred)
            noise_rows.append({
                "severity": severity,
                "repeat": rep,
                "seed": seed,
                "model": name,
                **m,
            })

            pr, rc, f1, sup = precision_recall_fscore_support(
                yte, pred, labels=sorted(np.unique(yte)), zero_division=0
            )
            for cls, p, r, ff, n in zip(sorted(np.unique(yte)), pr, rc, f1, sup):
                class_rows.append({
                    "severity": severity,
                    "repeat": rep,
                    "seed": seed,
                    "model": name,
                    "class": int(cls),
                    "precision": float(p),
                    "recall": float(r),
                    "f1": float(ff),
                    "n": int(n),
                })

noise_reps = pd.DataFrame(noise_rows)
noise_reps.to_csv(OUT / "04_noise_test_replicates.csv", index=False)
pd.DataFrame(class_rows).to_csv(OUT / "04_noise_test_classwise_replicates.csv", index=False)

agg = noise_reps.groupby(["severity", "model"]).agg(
    macro_f1_mean=("macro_f1", "mean"),
    macro_f1_std=("macro_f1", "std"),
    macro_f1_min=("macro_f1", "min"),
    macro_f1_max=("macro_f1", "max"),
    balanced_accuracy_mean=("balanced_accuracy", "mean"),
    accuracy_mean=("accuracy", "mean"),
).reset_index()

# Paired robust gains on identical TEST perturbations.
wide = noise_reps.pivot_table(
    index=["severity", "repeat", "seed"],
    columns="model",
    values="macro_f1"
).reset_index()

for base_name in ["Temporal-82 original", "Temporal-70 unsmoothed"]:
    wide[f"robust_minus_{base_name}"] = wide[rob_name] - wide[base_name]

gain_rows = []
for severity in NOISE_TEST_LEVELS:
    sub = wide[wide["severity"] == severity]
    for base_name in ["Temporal-82 original", "Temporal-70 unsmoothed"]:
        col = f"robust_minus_{base_name}"
        vals = sub[col].to_numpy()
        gain_rows.append({
            "severity": severity,
            "comparison": f"{rob_name} minus {base_name}",
            "mean_delta_macro_f1": float(vals.mean()),
            "std_delta_macro_f1": float(vals.std(ddof=1)),
            "min_delta_macro_f1": float(vals.min()),
            "max_delta_macro_f1": float(vals.max()),
        })

gain_df = pd.DataFrame(gain_rows)
agg.to_csv(OUT / "04_noise_test_summary.csv", index=False)
gain_df.to_csv(OUT / "04_noise_paired_gain_summary.csv", index=False)

print("Noise summary:")
print(agg.to_string(index=False))
print("Paired robust gains:")
print(gain_df.to_string(index=False))

# 12. Plots and decision-oriented summary

# Ablation plot: training CV is primary.
plot_abl = abl.sort_values("training_cv_macro_f1_mean", ascending=True)
plt.figure(figsize=(9, 6))
plt.barh(plot_abl["representation"], plot_abl["training_cv_macro_f1_mean"])
plt.xlabel("Training 5-fold CV macro-F1")
plt.title("Redundancy-aware feature-family ablation at 75% history")
plt.tight_layout()
plt.savefig(OUT / "fig_ablation_training_cv.png", dpi=180)
plt.close()

# Noise plot
plt.figure(figsize=(8, 5))
for name in agg["model"].unique():
    sub = agg[agg["model"] == name].sort_values("severity")
    plt.errorbar(
        sub["severity"] * 100,
        sub["macro_f1_mean"],
        yerr=sub["macro_f1_std"],
        marker="o",
        label=name,
    )
plt.xlabel("Independent multiplicative Gaussian noise (%)")
plt.ylabel("TEST macro-F1")
plt.title("Noise robustness after training-only robust-representation selection")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "fig_noise_robustness.png", dpi=180)
plt.close()

# Candidate selection plot
plt.figure(figsize=(7, 5))
plt.plot(cand["window"], cand["clean_oof_macro_f1"], marker="o", label="Clean OOF")
plt.plot(cand["window"], cand["stressed_oof_macro_f1_mean"], marker="o", label="Stressed OOF mean")
plt.axvline(SELECTED_WINDOW, linestyle="--", label=f"Selected w={SELECTED_WINDOW}")
plt.xlabel("Causal moving-average window (samples)")
plt.ylabel("Training OOF macro-F1")
plt.title("Training-only robust-window selection")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "fig_robust_window_selection.png", dpi=180)
plt.close()

summary = {
    "selected_convergence_stable_model": selection,
    "redundancy_aware_feature_counts": {
        "Statistical-28": 28,
        "Temporal-70": 70,
        "Temporal-82": 82,
    },
    "noise_robust_selection": selection_noise,
    "clean_test_comparison": clean_df.to_dict(orient="records"),
    "noise_test_summary": agg.to_dict(orient="records"),
    "noise_paired_gain_summary": gain_df.to_dict(orient="records"),
    "interpretation_guardrails": [
        "Hyperparameters and smoothing window were selected without TEST labels.",
        "The Gaussian-noise experiment is a controlled stress test, not field sensor validation.",
        "Feature-family mechanism claims should prioritize training-CV patterns.",
        "TEST results are held-out confirmation/descriptive evaluation after training-side selection."
    ],
}
(OUT / "FINAL_REFINEMENT_SUMMARY.json").write_text(json.dumps(summary, indent=2))

# Human-readable markdown
lines = [
    "# Final Compact Refinement Summary",
    "",
    "## Convergence-stable model selection",
    f"- C: {BEST_C}",
    f"- class_weight: {BEST_WEIGHT}",
    f"- training-CV macro-F1: {BEST_CV:.6f}",
    "",
    "## Noise-robust representation selection",
    f"- selected causal moving-average window: {SELECTED_WINDOW} samples",
    f"- clean-retention constraint: {CLEAN_RETENTION_CONSTRAINT:.0%} of unsmoothed training OOF",
    "",
    "## Clean held-out TEST",
]
for _, r in clean_df.iterrows():
    lines.append(f"- {r['model']}: macro-F1 = {r['macro_f1']:.6f}")
lines += [
    "",
    "## Methodological guardrails",
    "- TEST did not select C, class weighting, feature families, or smoothing window.",
    "- Synthetic Gaussian noise is a stress model, not a measured sensor-error distribution.",
    "- Use training-CV evidence before making causal/mechanistic claims from feature-family ablations.",
]
(OUT / "FINAL_REFINEMENT_SUMMARY.md").write_text("\n".join(lines))

print((OUT / "FINAL_REFINEMENT_SUMMARY.md").read_text())

# 13. Integrity audit and ZIP export

def sha256(path, block=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

audit = {
    "raw_root": str(RAW_ROOT),
    "canonical_root": str(CANON_ROOT),
    "sklearn_version": sklearn.__version__,
    "sklearn_file": sklearn.__file__,
    "seed": SEED,
    "cv_splits": N_SPLITS,
    "bootstrap": BOOTSTRAP,
    "noise_selection_levels": NOISE_SELECTION_LEVELS,
    "noise_selection_repeats": NOISE_SELECTION_REPEATS,
    "noise_test_levels": NOISE_TEST_LEVELS,
    "noise_test_repeats": NOISE_TEST_REPEATS,
    "clean_retention_constraint": CLEAN_RETENTION_CONSTRAINT,
    "candidate_smoothing_windows": SMOOTHING_WINDOWS,
    "selected_window": SELECTED_WINDOW,
    "selected_C": BEST_C,
    "selected_class_weight": BEST_WEIGHT,
}
(OUT / "RUN_AUDIT.json").write_text(json.dumps(audit, indent=2))

expected = [
    "01_tol1e6_training_grid.csv",
    "01_selected_model.json",
    "02_redundancy_aware_ablation.csv",
    "03_noise_robust_window_selection.csv",
    "03_noise_robust_selection.json",
    "04_clean_test_comparison.csv",
    "04_noise_test_replicates.csv",
    "04_noise_test_summary.csv",
    "04_noise_paired_gain_summary.csv",
    "FINAL_REFINEMENT_SUMMARY.json",
    "FINAL_REFINEMENT_SUMMARY.md",
    "RUN_AUDIT.json",
]
missing = [name for name in expected if not (OUT / name).exists()]
assert not missing, f"Missing outputs: {missing}"

ZIP_OUT = args.zip_output or (WORK.parent / f"{WORK.name}_results.zip")
if ZIP_OUT.exists():
    ZIP_OUT.unlink()

shutil.make_archive(
    str(ZIP_OUT.with_suffix("")),
    "zip",
    root_dir=WORK,
    base_dir="outputs",
)

print("\nSUCCESS")
print("Download and upload this file:")
print(ZIP_OUT)
print("Size MB:", round(ZIP_OUT.stat().st_size / 1024**2, 2))
print("SHA256:", sha256(ZIP_OUT))