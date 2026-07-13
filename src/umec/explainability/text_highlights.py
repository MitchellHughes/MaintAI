"""Map classifier evidence to spans in the original discrepancy text."""

from __future__ import annotations

import re
from typing import Any, Iterable

from umec.data.keyword_generation import is_noise_keyword
from umec.prediction.keywords import SEMANTIC_STOPWORDS, is_strong_keyword_hit

_HIGHLIGHT_STOPWORDS = SEMANTIC_STOPWORDS | frozenset(
    {
        "see",
        "full",
        "details",
        "detail",
        "attachment",
        "attachments",
        "investigation",
        "investigate",
        "found",
        "noted",
        "damaged",
        "damage",
        "panel",
        "seal",
        "wing",
        "fuel",
        "tank",
        "surge",
    }
)


def filter_highlight_terms(terms: Iterable[str], *, label: str | None = None) -> list[str]:
    """Drop discourse noise and label-inappropriate tokens from UI keyword chips."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in terms:
        term = str(raw).strip().lower()
        if not term or term in seen:
            continue
        if len(term) < 4 and term not in {"leak", "leaks"}:
            continue
        if term in _HIGHLIGHT_STOPWORDS or is_noise_keyword(term):
            continue
        if label and not is_strong_keyword_hit(label, term):
            continue
        out.append(term)
        seen.add(term)
    return out


def _word_pattern(term: str) -> re.Pattern[str]:
    return re.compile(r"(?i)\b" + re.escape(str(term).strip()) + r"\b")


def surface_terms_in_text(text: str, candidates: Iterable[str]) -> list[str]:
    """Return terms (or their tokens) that appear as words in ``text``."""
    if not text:
        return []
    blob = str(text).lower()
    found: list[str] = []
    seen: set[str] = set()

    for raw in candidates:
        cand = str(raw).strip().lower()
        if not cand or cand in seen:
            continue
        tokens = re.findall(r"[a-z]{3,}", cand)
        if len(tokens) > 1:
            for tok in tokens:
                if tok in seen:
                    continue
                if _word_pattern(tok).search(blob):
                    found.append(tok)
                    seen.add(tok)
            continue
        if _word_pattern(cand).search(blob):
            found.append(cand)
            seen.add(cand)
    return found


def find_non_overlapping_spans(
    text: str,
    term_roles: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """
    Find non-overlapping highlight spans in ``text``.

    ``term_roles`` is a list of (term, role) where role is support | alternate | other.
    Longer terms are matched first; support wins over alternate on overlap.
    """
    if not text or not term_roles:
        return []

    priority = {"support": 0, "alternate": 1, "other": 2}
    ordered = sorted(
        term_roles,
        key=lambda item: (priority.get(item[1], 9), -len(str(item[0]))),
    )

    spans: list[dict[str, Any]] = []
    occupied = [False] * len(text)

    for term, role in ordered:
        term = str(term).strip()
        if len(term) < 3:
            continue
        for match in _word_pattern(term).finditer(text):
            start, end = match.start(), match.end()
            if any(occupied[start:end]):
                continue
            for idx in range(start, end):
                occupied[idx] = True
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                    "role": role,
                    "term": term.lower(),
                }
            )

    return sorted(spans, key=lambda s: s["start"])


def _keywords_for_label(failure_keywords: dict[str, list[str]], label: str) -> list[str]:
    label_key = str(label).strip().lower()
    for key, words in (failure_keywords or {}).items():
        if str(key).strip().lower() == label_key:
            return [str(w).strip() for w in (words or []) if str(w).strip()]
    return []


def build_text_highlights(
    text: str,
    *,
    predicted_label: str,
    top_predictions: list[dict[str, Any]] | None = None,
    failure_keywords: dict[str, list[str]] | None = None,
    support_terms: Iterable[str] | None = None,
    actual_label: str | None = None,
) -> dict[str, Any]:
    """
    Build inline highlight spans for the review UI.

    - support (green): terms in text backing the predicted label
    - alternate (red): terms in text backing runner-up / top-k alternatives
    - other (amber): salient text terms not mapped to user categories (e.g. deteriorated)
    """
    predicted = str(predicted_label or "").strip().lower()
    failure_keywords = failure_keywords or {}
    top_predictions = top_predictions or []

    support_pool: list[str] = list(support_terms or [])
    support_pool.extend(_keywords_for_label(failure_keywords, predicted))
    support_found = surface_terms_in_text(text, support_pool)

    alternate_labels = [
        str(item.get("label", "")).strip()
        for item in top_predictions
        if str(item.get("label", "")).strip().lower() not in ("", predicted, "unclassified")
    ][:3]

    alternate_pool: list[str] = []
    for label in alternate_labels:
        alternate_pool.extend(_keywords_for_label(failure_keywords, label))
    alternate_found = [
        t for t in surface_terms_in_text(text, alternate_pool) if t not in support_found
    ]

    other_terms: list[str] = []
    actual = str(actual_label or "").strip().lower()
    if actual and actual not in (predicted, "unclassified"):
        other_terms.extend(surface_terms_in_text(text, [actual]))
    # Words like "deteriorated" in narrative when not a defined category keyword.
    for word in re.findall(r"[A-Za-z]{5,}", text):
        low = word.lower()
        if low in support_found or low in alternate_found:
            continue
        if low == predicted or low in {a.lower() for a in alternate_labels}:
            continue
        if low.endswith("ed") and low not in failure_keywords:
            if _word_pattern(low).search(text) and low not in other_terms:
                other_terms.append(low)

    support_found = filter_highlight_terms(support_found, label=predicted)
    alternate_found = filter_highlight_terms(alternate_found)
    other_terms = filter_highlight_terms(
        [t for t in other_terms if t not in support_found and t not in alternate_found]
    )

    term_roles = []
    for term in support_found:
        term_roles.append((term, "support"))
    for term in alternate_found:
        term_roles.append((term, "alternate"))
    for term in other_terms:
        term_roles.append((term, "other"))

    spans = find_non_overlapping_spans(text, term_roles)
    return {
        "spans": spans,
        "support_terms": support_found,
        "alternate_terms": alternate_found,
        "other_terms": other_terms,
    }
