"""Suggest category keyword updates from engineer corrections (active learning)."""

from __future__ import annotations

import re
from typing import Any

from umec.evaluation.category_matching import (
    build_category_matcher,
    normalize_label,
    parse_keywords,
)

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "was",
    "were",
    "from",
    "that",
    "this",
    "have",
    "has",
    "not",
    "are",
    "but",
    "into",
    "during",
    "found",
    "observed",
    "note",
    "item",
    "part",
    "unit",
    "left",
    "right",
    "also",
    "due",
    "per",
    "via",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]{3,}", normalize_label(text)) if t not in _STOPWORDS]


def suggest_keywords_from_edits(
    edits: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    *,
    max_per_label: int = 8,
) -> dict[str, Any]:
    """
    When engineers change Final away from Predicted, mine discrepancy tokens
    as keyword candidates for the corrected label.
    """
    match = build_category_matcher(categories)
    existing: dict[str, set[str]] = {}
    for cat in categories:
        label = normalize_label(cat.get("label"))
        if not label:
            continue
        existing[label] = set(parse_keywords(cat)) | {label}

    suggestions: dict[str, set[str]] = {}
    edit_count = 0

    for row in edits or []:
        predicted = normalize_label(row.get("predicted_condition") or row.get("predicted"))
        final_raw = row.get("final_condition") or row.get("final") or ""
        final = match(final_raw) or normalize_label(final_raw)
        if not final or final == predicted:
            continue
        edit_count += 1
        text = str(row.get("discrepancy") or row.get("text") or "")
        known = existing.get(final, set())
        for token in _tokens(text):
            if token in known:
                continue
            suggestions.setdefault(final, set()).add(token)

    by_label = {
        label: sorted(tokens)[:max_per_label]
        for label, tokens in suggestions.items()
        if tokens
    }

    return {
        "edit_count": edit_count,
        "suggestions": by_label,
    }


def merge_keyword_suggestions(
    categories: list[dict[str, Any]],
    suggestions: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Append suggested tokens to category keyword lists (deduped, case-insensitive)."""
    out: list[dict[str, Any]] = []
    for cat in categories:
        label = normalize_label(cat.get("label"))
        current = parse_keywords(cat)
        current_set = set(current)
        extra = suggestions.get(label) or suggestions.get(cat.get("label", "")) or []
        merged = list(current)
        for token in extra:
            tok = normalize_label(token)
            if tok and tok not in current_set:
                merged.append(tok)
                current_set.add(tok)
        out.append({**cat, "label": cat.get("label", ""), "keywords": ", ".join(merged)})
    return out
