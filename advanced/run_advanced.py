#!/usr/bin/env python3
"""Run the thesis-strengthening suite. CPU is sufficient; no GPU is required."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import sklearn

RECOMMENDED_SKLEARN = "1.9.0"


def run(cmd):
    print("\n+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--canonical-final", type=Path, required=True,
                    help="Unzipped final_5fold directory containing predictions/")
    ap.add_argument("--work-dir", type=Path, default=Path("advanced_run"))
    ap.add_argument("--robustness-repeats", type=int, default=20)
    ap.add_argument("--allow-version-mismatch", action="store_true",
                    help="Allow a smoke/preliminary run outside scikit-learn 1.9.0. Do not mix those numbers with the canonical thesis run.")
    args = ap.parse_args()

    if sklearn.__version__ != RECOMMENDED_SKLEARN:
        msg = (f"This environment has scikit-learn {sklearn.__version__}, but the canonical final paper model "
               f"was serialized with scikit-learn {RECOMMENDED_SKLEARN}. Exact advanced thesis numbers must be "
               "generated in the locked canonical environment.")
        if not args.allow_version_mismatch:
            raise SystemExit(msg + "\nInstall requirements/reproducible.txt or pass --allow-version-mismatch for code-path testing only.")
        print("WARNING:", msg, flush=True)

    root = Path(__file__).resolve().parent
    scripts = root / "scripts"
    feat = args.work_dir / "dense_features"
    out = args.work_dir / "advanced_results"
    feat.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    run([py, scripts/"04_build_dense_features.py", "--archive", args.archive, "--output-dir", feat])
    run([py, scripts/"05a_dense_representation_curve.py", "--features-dir", feat, "--output-dir", out/"dense_representation"])
    run([py, scripts/"05_dense_adaptive_stopping.py", "--features-dir", feat, "--output-dir", out/"adaptive"])
    # This one intentionally uses the already-canonical final_5fold predictions.
    run([py, scripts/"06_formal_interaction.py", "--predictions-dir", args.canonical_final/"predictions", "--output-dir", out/"interaction"])
    run([py, scripts/"07_feature_family_ablation.py", "--features-dir", feat, "--output-dir", out/"ablation"])
    run([py, scripts/"08_robustness_stress.py", "--archive", args.archive, "--features-dir", feat, "--output-dir", out/"robustness", "--repeats", args.robustness_repeats])
    run([py, scripts/"09_conformal_dense.py", "--features-dir", feat, "--output-dir", out/"conformal"])
    run([py, scripts/"10_shap_early_stability.py", "--features-dir", feat, "--output-dir", out/"shap"])
    run([py, scripts/"11_summarize_advanced_results.py", "--results-dir", out])
    print(f"\nAdvanced suite complete: {out}")


if __name__ == "__main__":
    main()
