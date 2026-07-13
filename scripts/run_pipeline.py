#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umec.pipeline.runner import run_evaluate, run_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UMEC train + evaluate pipeline.")
    parser.add_argument("--config", required=True, help="Path to config directory.")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after training.")
    args = parser.parse_args()

    run_train(args.config)
    if args.evaluate:
        run_evaluate(args.config)


if __name__ == "__main__":
    main()
