import os
from pathlib import Path

import pandas as pd
from flask import Blueprint, jsonify, request

from umec.data.keyword_generation import generate_keywords_for_labels, resolve_all_category_keywords
from umec.data.resources import load_label_mappings, load_token_mappings
from umec.prediction.keywords import find_overlapping_keywords

tokens_bp = Blueprint("tokens", __name__)

_DEFAULT_LABEL_MAPPINGS = Path("configs/mappings/label_mappings.json")
_DEFAULT_TOKEN_MAPPINGS = Path("configs/mappings/token_mappings.json")


def _label_mappings_from_request(data: dict) -> dict[str, str]:
    path = data.get("label_mappings_path") or os.getenv(
        "UMEC_LABEL_MAPPINGS",
        str(_DEFAULT_LABEL_MAPPINGS),
    )
    try:
        return load_label_mappings(path)
    except Exception:
        return {}


@tokens_bp.route("/api/generate_tokens", methods=["POST"])
def generate_tokens():
    data = request.get_json(silent=True) or {}
    labels = data.get("labels", [])
    corpus = data.get("corpus", [])

    if not isinstance(labels, list) or not labels:
        return jsonify({"error": "Body must include a non-empty 'labels' list."}), 400

    if isinstance(corpus, list):
        corpus = [str(t) for t in corpus if str(t).strip()]
    else:
        corpus = []

    rows = data.get("rows")
    text_column = data.get("text_column")
    label_column = data.get("label_column")
    part_column = data.get("part_column")
    custom_categories = data.get("custom_categories")

    df = None
    if isinstance(rows, list) and rows and text_column:
        df = pd.DataFrame(rows)

    categories: list[dict] = []
    if isinstance(custom_categories, list) and custom_categories:
        for item in custom_categories:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()
            if not label:
                continue
            categories.append(
                {
                    "label": label,
                    "keywords": item.get("keywords") or "",
                }
            )
    if not categories:
        categories = [{"label": str(lb).strip(), "keywords": ""} for lb in labels if str(lb).strip()]

    label_mappings = _label_mappings_from_request(data)
    token_map: dict[str, str] = {}
    token_path = data.get("token_mappings_path") or os.getenv(
        "UMEC_TOKEN_MAPPINGS",
        str(_DEFAULT_TOKEN_MAPPINGS),
    )
    try:
        token_map = load_token_mappings(token_path)
    except Exception:
        pass

    if df is not None and categories:
        generated = resolve_all_category_keywords(
            categories,
            corpus=corpus or None,
            df=df,
            text_column=text_column,
            label_column=label_column,
            label_mappings=label_mappings,
            token_map=token_map,
            part_column=part_column,
        )
        if labels:
            generated = {
                k: v
                for k, v in generated.items()
                if any(str(lb).strip().lower() == k for lb in labels)
            }
    else:
        generated = generate_keywords_for_labels(
            [str(lb).strip() for lb in labels if str(lb).strip()],
            corpus=corpus or None,
            categories=categories,
            df=df,
            text_column=text_column,
            label_column=label_column,
            label_mappings=label_mappings,
            token_map=token_map,
            part_column=part_column,
        )

    overlaps = find_overlapping_keywords(generated)
    return jsonify({"keywords": generated, "overlapping_keywords": overlaps})
