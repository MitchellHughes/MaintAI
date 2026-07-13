"""Strict per-label keyword checks (avoid shared-token leakage across classes)."""

from __future__ import annotations

import re

from umec.data.preprocessing import normalize_tokens

# Tokens that should not count as semantic evidence (discourse noise).
SEMANTIC_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)


def normalized_keywords_for_label(
    failure_keywords: dict[str, list[str]],
    label: str,
    token_map: dict[str, str] | None = None,
    *,
    normalize: bool = True,
) -> set[str]:
    words = failure_keywords.get(str(label)) or failure_keywords.get(str(label).lower()) or []
    out: set[str] = set()
    for raw in words:
        term = str(raw).strip().lower()
        if not term:
            continue
        if normalize and token_map is not None:
            term = normalize_tokens(term, token_map)
        elif normalize:
            term = term.lower().strip()
        if term:
            out.add(term)
    return out


# Generic fluid/system words — not enough alone to claim «leaking» (avoids HYD/no-leak false positives).
# Other failure mechanisms that should not ride along in mined «leaking» phrases.
_LEAKING_COLLISION_TERMS = frozenset(
    {
        "broken",
        "contaminated",
        "corroded",
        "cracked",
        "damaged",
        "defective",
        "deteriorated",
        "failed",
        "odour",
        "odor",
        "smoke",
        "worn",
    }
)

_LEAKING_WEAK_TERMS = frozenset(
    {
        "hydraulic",
        "fluid",
        "oil",
        "sys",
        "system",
        "reservoir",
        "pump",
        "blue",
        "green",
        "yellow",
        "pressure",
        "connection",
        "connections",
        "serviced",
        "service",
        "amm",
        "ref",
        "noted",
        "accomplished",
        "required",
        "inspection",
        "chapter",
        "defects",
        "continue",
        "okay",
        "ops",
        "burst",
        "tire",
        "brake",
        "assy",
        "mlg",
    }
)

_NEGATION_PATTERNS: dict[str, list[str]] = {
    "leaking": [
        r"\bno\s+leaks?\b",
        r"\bno\s+leakage\b",
        r"\bwithout\s+leaks?\b",
        r"\bnot\s+leaking\b",
        r"\bno\s+evidence\s+of\s+leaks?\b",
        r"\bleak\s*[- ]?free\b",
        r"\bno\s+hydraulic\s+leaks?\b",
        r"\bno\s+leaks?\s+of\s+hydraulic\b",
        r"\bno\s+leaks?\s+noted\b",
        r"\bleak\s+check\s+good\b",
        r"\bleak\s+checks?\s+good\b",
        r"\bops\s+and\s+leak\s+check\s+good\b",
        r"\bno\s+defects?\s+were\s+noted\b",
        r"\bno\s+defects?\s+noted\b",
        r"\bac\s+is\s+okay\b",
        r"\bokay\s+to\s+continue\b",
    ],
}


def label_negated_in_text(text: str, label: str) -> bool:
    """True when the text explicitly denies this failure mechanism."""
    label_key = str(label).strip().lower()
    patterns = _NEGATION_PATTERNS.get(label_key, [])
    if not patterns or not text:
        return False
    blob = str(text).lower()
    return any(re.search(pat, blob) for pat in patterns)


def is_strong_keyword_hit(label: str, term: str) -> bool:
    """
    For «leaking», only terms that mention leak/leaking/leakage count as evidence.

    Stops collateral hits from mined phrases (attachment, deteriorated, investigation, …).
    """
    label_key = str(label).strip().lower()
    term_key = str(term).strip().lower()
    if not term_key:
        return False
    if label_key != "leaking":
        return True
    if term_key in _LEAKING_WEAK_TERMS:
        return False
    if "leak" not in term_key:
        return False
    phrase_tokens = set(re.findall(r"[a-z]+", term_key))
    if phrase_tokens & _LEAKING_COLLISION_TERMS:
        return False
    return True


def keyword_hit_counts_as_evidence(label: str, term: str, text: str) -> bool:
    """Combine negation + strong-keyword rules for evidence gating."""
    if label_negated_in_text(text, label):
        return False
    return is_strong_keyword_hit(label, term)


def find_overlapping_keywords(failure_keywords: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return {token: [labels...]} for tokens assigned to more than one class."""
    index: dict[str, list[str]] = {}
    for label, words in failure_keywords.items():
        for raw in words or []:
            term = str(raw).strip().lower()
            if not term:
                continue
            index.setdefault(term, []).append(str(label))
    return {term: labels for term, labels in index.items() if len(labels) > 1}
