#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umec.explainability.explain import explain_record
from umec.pipeline.runner import run_train, _load_and_preprocess
from umec.utils.config import load_config
from umec.utils.serialization import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain a record with UMEC.")
    parser.add_argument("--config", required=True, help="Path to config directory.")
    parser.add_argument("--index", type=int, required=True, help="Record index to explain.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k scores to display.")
    parser.add_argument("--use-saved", action="store_true", help="Load saved models instead of retraining.")
    args = parser.parse_args()

    if args.use_saved:
        cfg = load_config(args.config)
        df = _load_and_preprocess(cfg)
        token_clf = load_model(Path(cfg.project.model_dir) / "token_matching.joblib")
        equipment_clf = load_model(Path(cfg.project.model_dir) / "equipment_based.joblib")
        semantic_clf = load_model(Path(cfg.project.model_dir) / "semantic_similarity.joblib")
        umec = load_model(Path(cfg.project.model_dir) / "umec.joblib")
    else:
        artifacts = run_train(args.config)
        cfg = artifacts["config"]
        df = artifacts["data"]
        token_clf = artifacts["token_clf"]
        equipment_clf = artifacts["equipment_clf"]
        semantic_clf = artifacts["semantic_clf"]
        umec = artifacts["umec"]

    text = explain_record(
        df=df,
        index=args.index,
        text_column=cfg.data.text_column,
        label_column=cfg.data.label_column,
        base_classifiers=[token_clf, equipment_clf, semantic_clf],
        umec=umec,
        top_k=args.top_k,
    )
    print(text)


if __name__ == "__main__":
    main()
