#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umec.pipeline.runner import run_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UMEC base classifiers and ensemble.")
    parser.add_argument("--config", required=True, help="Path to config directory.")
    args = parser.parse_args()

    artifacts = run_train(args.config)
    umec = artifacts["umec"]
    print(f"UMEC trained with {len(umec.classes)} classes and {len(umec.bit_labels)} ECOC bits")


if __name__ == "__main__":
    main()
