"""Resolve shared keywords across categories using per-upload association scores only."""

from __future__ import annotations

import re
from collections import Counter

from umec.data.keyword_enricher import token_relates_to_label
from umec.prediction.keywords import find_overlapping_keywords

_TERM_RE = re.compile(r"[a-z][a-z0-9]{2,}")

# Generic tokens that should not float across unrelated classes without strong association.
_GENERIC_SHARED = frozenset(
    {
        "fault",
        "faults",
        "faulted",
        "faulting",
        "found",
        "noted",
        "observed",
        "checked",
        "panel",
        "unit",
        "item",
        "area",
        "left",
        "right",
        "during",
        "maintenance",
        "action",
        "requires",
        "required",
    }
)


def _normalize(term: str) -> str:
    return re.sub(r"\s+", " ", str(term or "").strip().lower())


def _term_count_in_texts(term: str, texts: list[str]) -> int:
    term = _normalize(term)
    if not term:
        return 0
    count = 0
    if " " in term:
        for text in texts:
            if term in _normalize(text):
                count += 1
    else:
        pattern = re.compile(rf"\b{re.escape(term)}\b")
        for text in texts:
            if pattern.search(_normalize(text)):
                count += 1
    return count


def _association_scores(
    term: str,
    labels: list[str],
    positive_texts_by_label: dict[str, list[str]],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for label in labels:
        texts = positive_texts_by_label.get(label) or positive_texts_by_label.get(_normalize(label)) or []
        scores[label] = float(_term_count_in_texts(term, texts))
        if token_relates_to_label(term, label):
            scores[label] *= 2.0
    return scores


def dedupe_cross_class_keywords(
    keywords: dict[str, list[str]],
    positive_texts_by_label: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """
    Assign each overlapping token to at most one category for this upload.

    Uses only text buckets from the current dataframe/corpus — no global dataset file.
    """
    if not keywords:
        return keywords

    positive_texts_by_label = positive_texts_by_label or {}
    result: dict[str, list[str]] = {
        _normalize(label): [_normalize(t) for t in terms if _normalize(t)]
        for label, terms in keywords.items()
        if _normalize(label)
    }

    overlaps = find_overlapping_keywords(result)
    if not overlaps:
        return result

    for term, labels in overlaps.items():
        term = _normalize(term)
        labels = [_normalize(lb) for lb in labels if _normalize(lb)]
        if len(labels) < 2:
            continue

        morph_winners = [lb for lb in labels if token_relates_to_label(term, lb)]
        if len(morph_winners) == 1:
            winner = morph_winners[0]
        else:
            scores = _association_scores(term, labels, positive_texts_by_label)
            winner = max(labels, key=lambda lb: scores.get(lb, 0.0))
            if scores.get(winner, 0.0) <= 0 and term in _GENERIC_SHARED:
                for lb in labels:
                    if term in result.get(lb, []):
                        result[lb] = [t for t in result[lb] if t != term]
                continue

        for lb in labels:
            if lb == winner:
                continue
            if term in result.get(lb, []):
                result[lb] = [t for t in result[lb] if t != term]

    return result
