"""Command-line runner to generate all explainability, faithfulness, and error artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from .error_analysis import run_error_analysis
from .faithfulness import run_faithfulness_suite
from .global_terms import export_all_global_terms
from .utils import EXPLAIN_DIR


def main():
    parser = argparse.ArgumentParser(description="Generate all explainability artifacts.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPLAIN_DIR,
        help="Directory to save explanation artifacts",
    )
    args = parser.parse_args()

    start_time = time.time()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("STEP 1: Exporting global terms for all 5 models...")
    print("==================================================================")
    global_files = export_all_global_terms(out_dir, top_k=25)
    for model, path in global_files.items():
        print(f"  Saved {model} global terms -> {path}")

    print("\n==================================================================")
    print("STEP 2: Running faithfulness deletion tests & method agreement...")
    print("==================================================================")
    run_faithfulness_suite(out_dir)

    print("\n==================================================================")
    print("STEP 3: Running aggregate error analysis...")
    print("==================================================================")
    run_error_analysis(out_dir)

    duration = time.time() - start_time
    print(f"\n[DONE] All explainability artifacts generated in {duration:.2f}s.")


if __name__ == "__main__":
    main()
