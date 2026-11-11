"""Full pipeline automation script.

Downloads SpamAssassin data (if not already downloaded), parses and prepares data,
trains and evaluates models, and generates explainability artifacts.

Usage:
    python scripts/build_all.py          # Runs full end-to-end pipeline
    python scripts/build_all.py --prepare
    python scripts/build_all.py --train
    python scripts/build_all.py --explain
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_step(step_name: str, cmd: list[str]) -> None:
    print(f"\n{'='*70}\n[STEP] {step_name}\n{'='*70}", flush=True)
    start = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"[ERROR] Step '{step_name}' failed with return code {result.returncode}", flush=True)
        sys.exit(result.returncode)
    print(f"[SUCCESS] Step '{step_name}' completed in {elapsed:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and reproduce all Spam Detector artifacts.")
    parser.add_argument("--prepare", action="store_true", help="Download and parse SpamAssassin corpus")
    parser.add_argument("--train", action="store_true", help="Train and evaluate all classifiers")
    parser.add_argument("--explain", action="store_true", help="Generate Explainable AI artifacts")
    parser.add_argument("--all", action="store_true", help="Run the full pipeline end-to-end (default)")

    args = parser.parse_args()

    # Default to running all steps if no specific step is requested
    run_all = args.all or (not args.prepare and not args.train and not args.explain)

    py_exe = sys.executable

    if args.prepare or run_all:
        run_step("Data Preparation", [py_exe, "src/prepare_data.py"])

    if args.train or run_all:
        run_step("Model Training & Evaluation", [py_exe, "src/train_evaluate.py"])

    if args.explain or run_all:
        run_step("Explainability & Faithfulness", [py_exe, "-m", "src.explain.runner"])

    print(f"\n{'='*70}\nPipeline build complete!\n{'='*70}", flush=True)


if __name__ == "__main__":
    main()
