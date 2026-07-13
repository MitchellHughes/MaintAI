"""
Corpus vocabulary mining (classical NLP, no LLM) — always from the current upload only.

- c-TF-IDF: class pseudo-document vs upload corpus IDF
- PMI collocation: class vs rest log-odds on n-grams
- Prefix families + chi-squared feature selection
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2

from umec.utils.io import load_json_or_yaml

_ACTION_NOISE = frozenset(
    {
        "noted",
        "showed",
        "added",
        "worked",
        "found",
        "checking",
        "checked",
        "observed",
        "action",
        "corrective",
        "requires",
        "required",
        "accomplished",
        "repaired",
        "inspection",
        "attachment",
    }
)
_TOKEN_RE = re.compile(r"[a-z][a-z0-9]{2,}")
_DIGIT_RUN = re.compile(r"\d{3,}")


def _is_mined_noise(term: str) -> bool:
    term = _normalize(term)
    if not term or term in _ACTION_NOISE:
        return True
    if _DIGIT_RUN.search(term):
        return True
    if sum(ch.isdigit() for ch in term) / max(len(term), 1) >= 0.35:
        return True
    return term in {"amm", "era", "ref", "ship", "fr", "lhs", "rhs", "action", "svc", "flt", "ac"}


def _normalize(term: str) -> str:
    return re.sub(r"\s+", " ", str(term or "").strip().lower())


def label_prefix_stem(label: str) -> str:
    """4-char morphological anchor (matches experiments/failure_keyword_enricher.ipynb)."""
    label = _normalize(label)
    if not label:
        return ""
    core = label.split()[0]
    if len(core) >= 4:
        return core[:4]
    return label[:4] if len(label) >= 4 else label


def token_relates_to_label(token: str, label: str) -> bool:
    token = _normalize(token)
    label = _normalize(label)
    if not token or not label:
        return False
    stem = label_prefix_stem(label)
    if stem and (token.startswith(stem) or stem in token):
        return True
    if label in token or token in label:
        return True
    return False


def _apply_token_map(term: str, token_map: dict[str, str] | None) -> str:
    if not token_map:
        return term
    parts = term.split()
    return " ".join(token_map.get(p, p) for p in parts)


def _load_iso_hints(path: str | Path | None = None) -> dict[str, list[str]]:
    hint_path = path or Path("configs/mappings/iso_token_hints.yaml")
    try:
        raw = load_json_or_yaml(str(hint_path))
    except Exception:
        return {}
    hints = raw.get("hints") if isinstance(raw, dict) else None
    if not isinstance(hints, dict):
        return {}
    out: dict[str, list[str]] = {}
    for label, terms in hints.items():
        clean = _normalize(label)
        if not clean:
            continue
        if isinstance(terms, list):
            out[clean] = [_normalize(t) for t in terms if _normalize(t)]
    return out


def _corpus_vocabulary(
    texts: Iterable[str],
    *,
    min_df: int = 3,
    max_features: int = 8000,
) -> tuple[list[str], np.ndarray]:
    corpus = [_normalize(t) for t in texts if _normalize(t)]
    if not corpus:
        return [], np.array([])
    adaptive_min_df = max(2, min(min_df, len(corpus) // 200 or 2))
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=adaptive_min_df,
        max_features=max_features,
        stop_words="english",
        token_pattern=r"(?u)\b[a-z][a-z0-9]{2,}(?:\s[a-z][a-z0-9]{2,})?\b",
    )
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return [], np.array([])
    names = list(vectorizer.get_feature_names_out())
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    return names, counts


def prefix_family_keywords(
    label: str,
    feature_names: list[str],
    doc_counts: np.ndarray,
    *,
    total_docs: int,
    top_n: int = 18,
) -> list[str]:
    """Force engineering tokens sharing the label stem (corrosion, corroded, …)."""
    if not feature_names or total_docs <= 0:
        return []
    stem = label_prefix_stem(label)
    if not stem:
        return []

    scored: list[tuple[str, float, bool]] = []
    for idx, token in enumerate(feature_names):
        words = token.split()
        is_core = len(words) == 1 and token.startswith(stem)
        prefix_hit = any(w.startswith(stem) for w in words)
        if not is_core and not prefix_hit:
            continue
        popularity = float(doc_counts[idx]) / total_docs if idx < len(doc_counts) else 0.0
        specificity = 1.0 - min(popularity, 0.95)
        score = (2.5 if is_core else 1.2) * specificity
        scored.append((token, score, is_core))

    scored.sort(key=lambda x: (x[2], x[1]), reverse=True)
    out: list[str] = []
    seen: set[str] = set()
    for token, _, _ in scored:
        if token in seen:
            continue
        words = token.split()
        if len(words) > 1 and any(w in _ACTION_NOISE for w in words):
            if not any(w.startswith(stem) for w in words if w not in _ACTION_NOISE):
                continue
        seen.add(token)
        out.append(token)
        if len(out) >= top_n:
            break
    return out


def chi2_class_terms(
    positive_texts: list[str],
    negative_texts: list[str],
    *,
    top_n: int = 12,
    min_df: int = 2,
) -> list[str]:
    """
    Chi-squared ranking of n-grams in class vs rest-of-corpus documents.
    Standard filter for text categorization vocabularies.
    """
    pos = [_normalize(t) for t in positive_texts if _normalize(t)]
    neg = [_normalize(t) for t in negative_texts if _normalize(t)]
    if len(pos) < 2 or len(neg) < 2:
        return []

    docs = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg))
    adaptive_min_df = max(2, min(min_df, len(docs) // 100 or 2))
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        min_df=adaptive_min_df,
        max_features=6000,
        stop_words="english",
        token_pattern=r"(?u)\b[a-z][a-z0-9]{2,}(?:\s[a-z][a-z0-9]{2,})?\b",
    )
    try:
        matrix = vectorizer.fit_transform(docs)
    except ValueError:
        return []

    scores, _ = chi2(matrix, y)
    names = vectorizer.get_feature_names_out()
    ranked = sorted(
        zip(names, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [str(term) for term, score in ranked[:top_n] if float(score) > 0]


def ctfidf_class_terms(
    label: str,
    positive_texts: list[str],
    corpus_texts: list[str],
    *,
    top_n: int = 14,
) -> list[str]:
    """Class-focused TF-IDF (c-TF-IDF): one pseudo-doc per class vs this upload's IDF."""
    pos = [_normalize(t) for t in positive_texts if _normalize(t)]
    corpus = [_normalize(t) for t in corpus_texts if _normalize(t)]
    if not pos or len(corpus) < 2:
        return []

    class_doc = " ".join(pos)
    adaptive_min_df = max(1, min(3, len(corpus) // 300 or 1))
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=adaptive_min_df,
        max_features=6000,
        stop_words="english",
        token_pattern=r"(?u)\b[a-z][a-z0-9]{2,}(?:\s[a-z][a-z0-9]{2,})?\b",
    )
    try:
        vectorizer.fit(corpus)
        weights = vectorizer.transform([class_doc]).toarray().ravel()
    except ValueError:
        return []

    names = list(vectorizer.get_feature_names_out())
    stem = label_prefix_stem(label)
    ranked = sorted(
        zip(names, weights),
        key=lambda item: (
            0 if (stem and str(item[0]).startswith(stem)) else 1,
            -float(item[1]),
        ),
    )
    out: list[str] = []
    for term, weight in ranked:
        if float(weight) <= 0:
            continue
        out.append(str(term))
        if len(out) >= top_n:
            break
    return out


def pmi_class_terms(
    positive_texts: list[str],
    negative_texts: list[str],
    *,
    label: str = "",
    top_n: int = 12,
    min_count: int = 2,
) -> list[str]:
    """PMI-style collocation scores for n-grams in class texts vs rest of upload."""
    pos = [_normalize(t) for t in positive_texts if _normalize(t)]
    neg = [_normalize(t) for t in negative_texts if _normalize(t)]
    if len(pos) < 2:
        return []

    def _count_terms(texts: list[str]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for text in texts:
            tokens = _TOKEN_RE.findall(_normalize(text))
            for i, tok in enumerate(tokens):
                counts[tok] += 1
                if i + 1 < len(tokens):
                    counts[f"{tokens[i]} {tokens[i + 1]}"] += 1
        return counts

    pos_c = _count_terms(pos)
    neg_c = _count_terms(neg) if neg else Counter()
    n_pos = max(len(pos), 1)
    n_neg = max(len(neg), 1)
    scores: dict[str, float] = {}

    for term, pos_raw in pos_c.items():
        if pos_raw < min_count and " " not in term:
            continue
        pos_rate = pos_raw / n_pos
        neg_rate = neg_c.get(term, 0) / n_neg
        scores[term] = math.log((pos_rate + 1e-4) / (neg_rate + 1e-4))
        if label and token_relates_to_label(term, label):
            scores[term] += 1.5

    ranked = sorted(scores.items(), key=lambda item: (-item[1], -pos_c[item[0]], item[0]))
    return [t for t, _ in ranked[:top_n] if _ > 0]


def corpus_validated_hints(
    label: str,
    corpus_terms: set[str],
    *,
    hints_path: str | Path | None = None,
) -> list[str]:
    """ISO-style hints only when they actually appear in this upload."""
    hints = _load_iso_hints(hints_path).get(_normalize(label), [])
    validated: list[str] = []
    for hint in hints:
        if any(hint in term or term.startswith(hint) for term in corpus_terms):
            validated.append(hint)
    return validated


def enrich_label_keywords(
    label: str,
    base_terms: list[str],
    *,
    corpus: list[str] | None,
    positive_texts: list[str] | None = None,
    negative_texts: list[str] | None = None,
    token_map: dict[str, str] | None = None,
    max_terms: int = 28,
) -> list[str]:
    """
    Merge discriminative base terms with c-TF-IDF, PMI, and prefix-family terms (this upload only).
    """
    label = _normalize(label)
    pos = list(positive_texts or [])
    full_corpus = list(corpus or []) or pos
    mining_corpus = pos + [t for t in full_corpus if t not in pos]
    if not mining_corpus:
        return base_terms[:max_terms]

    neg = list(negative_texts or [])
    if not neg and full_corpus:
        pos_set = set(pos)
        neg = [t for t in full_corpus if t not in pos_set]

    ctfidf = ctfidf_class_terms(label, pos or mining_corpus[:500], full_corpus, top_n=max_terms)
    pmi = pmi_class_terms(pos, neg, label=label, top_n=max_terms // 2)

    feature_names, doc_counts = _corpus_vocabulary(mining_corpus)
    total_docs = len(mining_corpus)
    family = prefix_family_keywords(
        label,
        feature_names,
        doc_counts,
        total_docs=total_docs,
        top_n=max_terms,
    )

    corpus_terms = set(feature_names) | set(base_terms) | set(ctfidf) | set(pmi)
    hints = corpus_validated_hints(label, corpus_terms)

    merged: list[str] = []
    seen: set[str] = set()
    for term in ctfidf + pmi + family + hints + base_terms:
        term = _apply_token_map(_normalize(term), token_map)
        if not term or term in seen or _is_mined_noise(term):
            continue
        seen.add(term)
        merged.append(term)

    return merged[:max_terms]
