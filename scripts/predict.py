#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umec.data.io import read_data, save_data
from umec.data.preprocessing import preprocess_dataframe, normalize_tokens
from umec.data.resources import load_failure_keywords, load_token_mappings
from umec.data.validation import validate_columns
from umec.utils.config import load_config
from umec.utils.serialization import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UMEC inference from saved models.")
    parser.add_argument("--config", required=True, help="Path to config directory.")
    parser.add_argument("--input", help="Optional input file path (overrides config).")
    parser.add_argument("--output", default="reports/predictions.csv", help="Output CSV path.")
    parser.add_argument("--include-xai", action="store_true", help="Include confidence and keyword XAI columns.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k keywords for XAI.")
    parser.add_argument("--batch-size", type=int, default=5000, help="Batch size for inference.")
    parser.add_argument("--scores-output", help="Optional CSV path for per-class UMEC scores.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_path = args.input or cfg.data.path
    

    read_kwargs = cfg.data.read_kwargs or {}
    df = read_data(input_path, file_format=cfg.data.format, **read_kwargs)
    print(f"Input path used: {input_path}")
    print(f"Loaded shape: {df.shape}")
    print(f"Loaded columns: {list(df.columns)}")
    source_text = cfg.data.source_text_column
    if source_text is None and cfg.data.text_column.startswith("processed_"):
        source_text = cfg.data.text_column.replace("processed_", "", 1)
    source_text = source_text or cfg.data.text_column

    validate_columns(df, [source_text])

    if cfg.data.preprocess.get("enabled", True):
        token_map = load_token_mappings(cfg.data.resources["token_mappings"])
        df = preprocess_dataframe(
            df,
            text_column=source_text,
            output_column=cfg.data.text_column,
            preprocess_cfg=cfg.data.preprocess,
            token_map=token_map,
        )

    umec = load_model(Path(cfg.project.model_dir) / "umec.joblib")

    total = len(df)
    batch_size = max(1, args.batch_size)

    pred_parts = []
    reduction_parts = []
    token_xai_parts = []
    umec_xai_parts = []
    score_parts = []

    if args.include_xai:
        token_clf = load_model(Path(cfg.project.model_dir) / "token_matching.joblib")
        failure_keywords = load_failure_keywords(cfg.data.resources["failure_keywords"])
        token_map = load_token_mappings(cfg.data.resources["token_mappings"])
        token_vocab = token_clf.vectorizer.vocabulary_

        def normalize_keyword(keyword: str) -> str:
            keyword = keyword.lower().strip()
            return normalize_tokens(keyword, token_map)

        keyword_index = {}
        for label, keywords in failure_keywords.items():
            idxs = []
            for kw in keywords:
                norm_kw = normalize_keyword(kw)
                if norm_kw in token_vocab:
                    idxs.append((kw, token_vocab[norm_kw]))
            keyword_index[label] = idxs

    steps = range(0, total, batch_size)
    for start in tqdm(steps, desc="Predicting", total=len(steps)):
        end = min(start + batch_size, total)
        chunk = df.iloc[start:end]

        preds, reduction = umec.predict(chunk, column_name=cfg.data.text_column)
        pred_parts.append(preds)
        reduction_parts.append(reduction)

        if args.include_xai or args.scores_output:
            class_scores = umec.class_score_df(reduction)
            score_parts.append(class_scores)

        if args.include_xai:
            score_vals = class_scores.values
            score_vals = score_vals - score_vals.max(axis=1, keepdims=True)
            exp_scores = pd.DataFrame(
                data=np.exp(score_vals),
                index=class_scores.index,
                columns=class_scores.columns,
            )
            prob_scores = exp_scores.div(exp_scores.sum(axis=1), axis=0)

            tfidf_matrix = token_clf.vectorizer.transform(
                chunk[cfg.data.text_column].fillna("").astype(str)
            )

            confidence = []
            keyword_summaries = []
            for row_idx, label in preds.items():
                if label not in prob_scores.columns:
                    confidence.append(0.0)
                    keyword_summaries.append("NA")
                    continue

                row_probs = prob_scores.loc[row_idx].sort_values(ascending=False)
                top1 = float(row_probs.iloc[0])
                top2 = float(row_probs.iloc[1]) if len(row_probs) > 1 else 0.0
                prob = (top1 / max(top1 + top2, 1e-9)) * 100.0
                confidence.append(round(prob, 2))

                kw_idxs = keyword_index.get(label, [])
                if not kw_idxs:
                    keyword_summaries.append("NA")
                    continue

                row = tfidf_matrix[chunk.index.get_loc(row_idx)]
                weights = []
                for kw, idx in kw_idxs:
                    val = row[0, idx]
                    if val > 0:
                        weights.append((kw, float(val)))

                if not weights:
                    keyword_summaries.append("NA")
                    continue

                weights.sort(key=lambda x: x[1], reverse=True)
                top = weights[: args.top_k]
                total_w = sum(w for _, w in top) or 1.0
                formatted = "|".join(
                    f"{kw}={w / total_w * 100.0:.1f}%" for kw, w in top
                )
                keyword_summaries.append(formatted)

            token_xai_parts.append(pd.Series(confidence, index=chunk.index))
            umec_xai_parts.append(pd.Series(keyword_summaries, index=chunk.index))

    preds = pd.concat(pred_parts)
    reduction = pd.concat(reduction_parts)
    class_scores = pd.concat(score_parts) if score_parts else None
    out_df = df.copy()

    actual_col = None

    if "PartCondition" in df.columns:
        actual_col = "PartCondition"
    elif cfg.data.label_column and cfg.data.label_column in df.columns:
        actual_col = cfg.data.label_column

    if actual_col:
        out_df["actual_label"] = df[actual_col]

    out_df["umec_prediction"] = preds

    if args.include_xai:
        out_df["xai_confidence_pct"] = pd.concat(token_xai_parts)
        out_df["xai_keywords"] = pd.concat(umec_xai_parts)

        print("Created XAI cols:", ["xai_confidence_pct", "xai_keywords"])

    cols = []
    source_text_col = source_text if source_text in out_df.columns else cfg.data.text_column
    cols.append(source_text_col)
    if actual_col:
        cols.append("actual_label")
    cols.append("umec_prediction")
    if args.include_xai:
        cols.extend(["xai_confidence_pct", "xai_keywords"])

    save_data(out_df[cols], args.output, file_format=Path(args.output).suffix.lstrip("."))

    if args.scores_output and class_scores is not None:
        score_df = class_scores.copy()
        score_df.insert(0, source_text_col, out_df[source_text_col])
        score_df.insert(1, "umec_prediction", out_df["umec_prediction"])
        save_data(score_df, args.scores_output, file_format=Path(args.scores_output).suffix.lstrip("."))

    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
