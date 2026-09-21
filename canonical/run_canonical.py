from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True, help="Path to the original Kaggle archive.zip")
    ap.add_argument("--run-dir", type=Path, default=Path("runs/final_5fold"))
    ap.add_argument("--tuning-iterations", type=int, default=20)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--smoke", action="store_true", help="Fast code check; CV remains 5-fold")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    run_dir = args.run_dir.resolve()
    features_dir = run_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    run([
        sys.executable,
        str(root / "scripts" / "01_build_features.py"),
        "--archive", str(args.archive.resolve()),
        "--output-dir", str(features_dir),
    ])

    cmd = [
        sys.executable,
        str(root / "scripts" / "02_run_5fold_experiments.py"),
        "--features-dir", str(features_dir),
        "--output-dir", str(run_dir),
        "--tuning-iterations", str(args.tuning_iterations),
        "--bootstrap", str(args.bootstrap),
    ]
    if args.smoke:
        cmd.append("--smoke")
    run(cmd)

    run([
        sys.executable,
        str(root / "scripts" / "03_shap_explanations.py"),
        "--features-dir", str(features_dir),
        "--run-dir", str(run_dir),
    ])


    zip_base = run_dir.parent / run_dir.name
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=run_dir))
    print(f"\nFINAL OUTPUT ZIP: {zip_path}")
    print("Run complete.")


if __name__ == "__main__":
    main()
