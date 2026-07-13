#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umec.pipeline.runner import run_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate UMEC ensemble.")
    parser.add_argument("--config", required=True, help="Path to config directory.")
    args = parser.parse_args()

    results = run_evaluate(args.config)
    print("Evaluation complete")
    print(f"Report: {results['report_path']}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Top-2 accuracy: {results['top2_accuracy']:.4f}")


if __name__ == "__main__":
    main()
