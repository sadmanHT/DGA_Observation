#!/usr/bin/env python3
"""Run all computational layers that contribute to the updated paper."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import sklearn

REQUIRED = "1.9.0"

def run(cmd):
    print("\n+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)

ap = argparse.ArgumentParser()
ap.add_argument("--archive", type=Path, required=True)
ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
ap.add_argument("--bootstrap", type=int, default=5000)
ap.add_argument("--robustness-repeats", type=int, default=20)
args = ap.parse_args()

if sklearn.__version__ != REQUIRED:
    raise SystemExit(f"Exact reproduction requires scikit-learn {REQUIRED}; found {sklearn.__version__}")

root = Path(__file__).resolve().parent
runs = args.runs_dir.resolve()
runs.mkdir(parents=True, exist_ok=True)
canon = runs / "canonical_final_5fold"
advanced = runs / "advanced"
refine = runs / "refinement"

# 1) Canonical paper experiment: original numerical protocol.
run([
    sys.executable, root/"canonical/run_canonical.py",
    "--archive", args.archive,
    "--run-dir", canon,
    "--tuning-iterations", "20",
    "--bootstrap", str(args.bootstrap),
])

# 2) Advanced extension: dense horizons, adaptive stopping, interaction,
#    ablation, stress tests, conformal prediction, horizon-specific SHAP.
run([
    sys.executable, root/"advanced/run_advanced.py",
    "--archive", args.archive,
    "--canonical-final", canon,
    "--work-dir", advanced,
    "--robustness-repeats", str(args.robustness_repeats),
])

# 3) Final compact refinement: convergence-stable grid, nonredundant features,
#    training-selected causal smoothing, held-out noise evaluation.
run([
    sys.executable, root/"refinement/run_refinement.py",
    "--archive", args.archive,
    "--canonical-final", canon,
    "--work-dir", refine,
    "--bootstrap", str(args.bootstrap),
    "--noise-test-repeats", str(args.robustness_repeats),
])

print("\nAll paper experiments completed under:", runs)
