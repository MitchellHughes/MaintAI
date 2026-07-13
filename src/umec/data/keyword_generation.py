"""
Dataset-only keyword generation for failure-mechanism classes.

Unsupervised / weakly-supervised NLP (no LLM):
- Discriminative log-odds term scoring (one-vs-rest pseudo-documents)
- TF-IDF corpus vocabulary with morphological prefix families (corr* → corrosion)
- Chi-squared feature selection when positive/negative buckets exist
- N-gram collocation mining from class-associated maintenance narrative

See also experiments/failure_keyword_enricher.ipynb (same prefix-family idea).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from umec.data.keyword_enricher import chi2_class_terms, enrich_label_keywords, token_relates_to_label
from umec.data.keyword_overlap import dedupe_cross_class_keywords
from umec.utils.io import load_json_or_yaml
from umec.evaluation.category_matching import build_category_matcher, normalize_label

_CORPUS_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "was",
        "were",
        "has",
        "have",
        "not",
        "per",
        "also",
        "into",
        "during",
        "found",
        "noted",
        "observed",
        "checked",
        "action",
        "associated",
        "complied",
        "require",
        "requires",
        "required",
        "needs",
        "need",
        "proper",
        "corective",
        "corrective",
        "serviced",
        "service",
        "item",
        "unit",
        "left",
        "right",
        "chk",
        "emer",
        "assit",
        "assyst",
        "cids",
        "pack",
        "level",
        "area",
        "panel",
        "door",
    }
)

_TOKEN_RE = re.compile(r"[a-z][a-z0-9]{2,}")
_DIGIT_RUN = re.compile(r"\d{3,}")
# CMMS / AMM noise — part numbers, ERA refs, ship IDs (not failure-mechanism terms).
_NOISE_TOKEN_RE = re.compile(
    r"^("
    r"\d{3,}[a-z]*|"  # 1750640, 5310, 3024
    r"[a-z]*\d{4,}[a-z\d]*|"  # 14ad, c20225510
    r"\d+[a-z]{1,3}\d*"
    r")$",
    re.I,
)


def is_noise_keyword(term: str) -> bool:
    """Reject part numbers, document IDs, and other maintenance metadata tokens."""
    term = normalize_keyword_term(term)
    if not term:
        return True
    if term in _CORPUS_STOPWORDS:
        return True
    if _NOISE_TOKEN_RE.match(term):
        return True
    if _DIGIT_RUN.search(term):
        return True
    digit_count = sum(ch.isdigit() for ch in term)
    if digit_count and digit_count / max(len(term), 1) >= 0.35:
        return True
    if term in {"amm", "era", "ref", "ship", "fr", "lhs", "rhs", "mlg", "flt", "ac", "svc", "rpt"}:
        return True
    return False


def normalize_keyword_term(term: str) -> str:
    return re.sub(r"\s+", " ", str(term or "").strip().lower())


def is_valid_keyword(term: str) -> bool:
    term = normalize_keyword_term(term)
    if len(term) < 3 or len(term) > 48:
        return False
    if is_noise_keyword(term):
        return False
    # Broken -ed truncation (corroded -> corrod) — real forms come from corpus mining.
    if term.endswith("rod") and term.endswith(("corrod", "erod")):
        return False
    parts = term.split()
    if all(p in _CORPUS_STOPWORDS for p in parts):
        return False
    return True


def _label_roots(label: str) -> list[str]:
    """Conservative label forms — morphological variants come from corpus mining."""
    label = normalize_keyword_term(label)
    if not label:
        return []
    roots: list[str] = [label]
    spaced = label.replace("_", " ")
    if spaced != label:
        roots.append(spaced)
    if label.endswith("ged") and len(label) > 5:
        roots.append(label[:-1])  # damaged -> damage
    elif label.endswith("cked") and len(label) > 5:
        roots.append(label[:-3] + "k")  # cracked -> crack
    if label.endswith("ing") and len(label) > 5:
        stem = label[:-3]
        if len(stem) >= 3:
            roots.append(stem)  # leaking -> leak
    out: list[str] = []
    seen: set[str] = set()
    for r in roots:
        if is_valid_keyword(r) and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _load_label_mappings(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    try:
        raw = load_json_or_yaml(str(path))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {normalize_keyword_term(k): normalize_keyword_term(v) for k, v in raw.items()}


def _reference_maps_to_category(
    raw_value: str,
    category_label: str,
    *,
    label_mappings: dict[str, str],
    matcher=None,
) -> bool:
    """Map noisy CMMS reference values to a user category without pre-existing keywords."""
    target = normalize_keyword_term(category_label)
    raw = normalize_keyword_term(raw_value)
    if not raw or not target:
        return False
    if raw == target:
        return True
    if label_mappings.get(raw) == target:
        return True
    if matcher:
        mapped = matcher(raw_value)
        if mapped == target:
            return True
    if target in raw or raw in target:
        return True
    for root in _label_roots(target):
        if len(root) >= 4 and re.search(rf"\b{re.escape(root)}\b", raw):
            return True
    if token_relates_to_label(raw, target):
        return True
    return False


def _extract_terms(text: str) -> list[str]:
    blob = normalize_keyword_term(text)
    if not blob:
        return []
    tokens = _TOKEN_RE.findall(blob)
    terms: list[str] = []
    for i, tok in enumerate(tokens):
        if is_valid_keyword(tok):
            terms.append(tok)
        if i + 1 < len(tokens):
            bg = f"{tokens[i]} {tokens[i + 1]}"
            if is_valid_keyword(bg):
                terms.append(bg)
        if i + 2 < len(tokens):
            tg = f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}"
            if is_valid_keyword(tg):
                terms.append(tg)
    return terms


def _text_mentions_category(text: str, label: str) -> bool:
    blob = normalize_keyword_term(text)
    if not blob:
        return False
    for root in _label_roots(label):
        if " " in root:
            if root in blob:
                return True
        elif re.search(rf"\b{re.escape(root)}\b", blob):
            return True
    for tok in _TOKEN_RE.findall(blob):
        if token_relates_to_label(tok, label):
            return True
    return False


def _compose_narrative_row(
    row: pd.Series,
    text_column: str,
    part_column: str | None = None,
) -> str:
    text = str(row.get(text_column, "")).strip()
    part = str(row.get(part_column, "")).strip() if part_column else ""
    if part:
        part_lower = part.lower()
        text_lower = text.lower()
        if not text_lower or part_lower not in text_lower:
            return f"{part} — {text}".strip() if text else part
    return text


def _labeled_texts_for_category(
    df: pd.DataFrame | None,
    label: str,
    *,
    text_column: str,
    label_column: str | None,
    categories: list[dict] | None,
    label_mappings: dict[str, str],
    part_column: str | None = None,
) -> list[str]:
    if df is None or not label_column or label_column not in df.columns:
        return []
    if text_column not in df.columns:
        return []

    matcher = build_category_matcher(categories or []) if categories else None
    texts: list[str] = []
    for _, row in df.iterrows():
        raw_label = row.get(label_column)
        if not _reference_maps_to_category(
            str(raw_label or ""),
            label,
            label_mappings=label_mappings,
            matcher=matcher,
        ):
            continue
        text = _compose_narrative_row(row, text_column, part_column)
        if text:
            texts.append(text)
    return texts


def _discriminative_terms(
    positive_texts: list[str],
    negative_texts: list[str],
    *,
    label: str = "",
    max_terms: int = 24,
    min_count: int = 2,
) -> list[str]:
    """
    Terms that appear more often in this category's texts than in others (handles noisy data).
    """
    if not positive_texts:
        return []

    pos_counts: Counter[str] = Counter()
    neg_counts: Counter[str] = Counter()

    for text in positive_texts:
        for term in _extract_terms(text):
            pos_counts[term] += 1
    for text in negative_texts:
        for term in _extract_terms(text):
            neg_counts[term] += 1

    n_pos = max(len(positive_texts), 1)
    n_neg = max(len(negative_texts), 1)
    scores: dict[str, float] = {}

    for term, pos_raw in pos_counts.items():
        if pos_raw < min_count and " " not in term:
            continue
        if not is_valid_keyword(term):
            continue
        pos_rate = pos_raw / n_pos
        neg_rate = neg_counts.get(term, 0) / n_neg
        # Log-odds style score with smoothing
        score = math.log1p(pos_raw) * math.log((pos_rate + 1e-3) / (neg_rate + 1e-3) + 1.0)
        if label and token_relates_to_label(term, label):
            score *= 2.0
        scores[term] = score

    ranked = sorted(scores.items(), key=lambda item: (-item[1], -pos_counts[item[0]], item[0]))
    return [t for t, _ in ranked[:max_terms]]


def _frequency_terms(texts: list[str], *, max_terms: int = 16) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        for term in _extract_terms(text):
            counts[term] += 1
    return [t for t, c in counts.most_common(max_terms * 2) if c >= 2 and is_valid_keyword(t)][:max_terms]


def _build_text_buckets(
    label: str,
    *,
    corpus: list[str] | None,
    df: pd.DataFrame | None,
    text_column: str | None,
    label_column: str | None,
    categories: list[dict] | None,
    label_mappings: dict[str, str],
    all_labels: list[str] | None,
    part_column: str | None = None,
) -> tuple[list[str], list[str]]:
    """Positive texts for this label vs negative (other categories + unrelated rows)."""
    positive: list[str] = []

    if df is not None and text_column and label_column:
        positive.extend(
            _labeled_texts_for_category(
                df,
                label,
                text_column=text_column,
                label_column=label_column,
                categories=categories,
                label_mappings=label_mappings,
                part_column=part_column,
            )
        )

    if corpus:
        for text in corpus:
            if _text_mentions_category(text, label):
                positive.append(text)

    # Dedupe while preserving order
    seen: set[str] = set()
    pos_unique: list[str] = []
    for t in positive:
        key = t[:200]
        if key not in seen:
            seen.add(key)
            pos_unique.append(t)
    positive = pos_unique

    negative: list[str] = []
    if df is not None and text_column and label_column and all_labels:
        for other in all_labels:
            if normalize_label(other) == normalize_label(label):
                continue
            negative.extend(
                _labeled_texts_for_category(
                    df,
                    other,
                    text_column=text_column,
                    label_column=label_column,
                    categories=categories,
                    label_mappings=label_mappings,
                    part_column=part_column,
                )
            )

    if corpus:
        pos_set = set(positive)
        for text in corpus:
            if text in pos_set:
                continue
            if not _text_mentions_category(text, label):
                negative.append(text)

    if not negative and corpus:
        negative = [t for t in corpus if t not in set(positive)]

    return positive, negative


def resolve_category_keywords(
    label: str,
    *,
    user_keywords: list[str] | None = None,
    corpus: list[str] | None = None,
    df: pd.DataFrame | None = None,
    text_column: str | None = None,
    label_column: str | None = None,
    categories: list[dict] | None = None,
    label_mappings: dict[str, str] | None = None,
    all_labels: list[str] | None = None,
    token_map: dict[str, str] | None = None,
    part_column: str | None = None,
    max_terms: int = 28,
) -> list[str]:
    """
    Keywords mined from this dataset using discriminative term scoring.

    Uses reference column + label_mappings to find rows for each category, then
    extracts words/phrases that distinguish that category from others in the upload.
    """
    label = normalize_keyword_term(label)
    if not label:
        return []

    explicit = [
        normalize_keyword_term(k)
        for k in (user_keywords or [])
        if is_valid_keyword(normalize_keyword_term(k))
    ]
    if explicit:
        seen: set[str] = set()
        out: list[str] = []
        for term in explicit:
            if term not in seen:
                seen.add(term)
                out.append(term)
        for root in _label_roots(label):
            if root not in seen:
                seen.add(root)
                out.append(root)
        return out[:max_terms]

    mappings = label_mappings or {}
    labels_list = all_labels or [label]
    positive, negative = _build_text_buckets(
        label,
        corpus=corpus,
        df=df,
        text_column=text_column,
        label_column=label_column,
        categories=categories,
        label_mappings=mappings,
        all_labels=labels_list,
        part_column=part_column,
    )

    mined: list[str] = []
    if positive:
        mined = _discriminative_terms(
            positive,
            negative,
            label=label,
            max_terms=max_terms - len(_label_roots(label)),
        )
        if len(mined) < 6:
            for term in _frequency_terms(positive, max_terms=max_terms):
                if term not in mined:
                    mined.append(term)
                if len(mined) >= max_terms - 4:
                    break

    if not mined and corpus:
        narrative = [t for t in corpus if _text_mentions_category(t, label)]
        mined = _frequency_terms(narrative or corpus[:3000], max_terms=max_terms)

    roots = _label_roots(label)
    # Prefer short discriminative unigrams, then phrases, then label stems
    unigrams = [t for t in mined if " " not in t]
    phrases = [t for t in mined if " " in t]

    result: list[str] = []
    seen: set[str] = set()
    for term in unigrams + phrases + roots:
        if term and term not in seen and is_valid_keyword(term):
            seen.add(term)
            result.append(term)

    if positive and negative:
        for term in chi2_class_terms(positive, negative, top_n=12):
            if term not in result and is_valid_keyword(term):
                result.append(term)

    enriched = enrich_label_keywords(
        label,
        result,
        corpus=corpus,
        positive_texts=positive,
        negative_texts=negative,
        token_map=token_map,
        max_terms=max_terms,
    )

    return enriched[:max_terms] if enriched else roots


def collect_positive_texts_by_label(
    labels: list[str],
    *,
    corpus: list[str] | None,
    df: pd.DataFrame | None = None,
    text_column: str | None = None,
    label_column: str | None = None,
    categories: list[dict] | None = None,
    label_mappings: dict[str, str] | None = None,
    part_column: str | None = None,
) -> dict[str, list[str]]:
    """Per-class narrative buckets from this upload only (for overlap resolution)."""
    mappings = label_mappings or {}
    out: dict[str, list[str]] = {}
    for label in labels:
        clean = normalize_keyword_term(label)
        if not clean:
            continue
        positive, _ = _build_text_buckets(
            clean,
            corpus=corpus,
            df=df,
            text_column=text_column,
            label_column=label_column,
            categories=categories,
            label_mappings=mappings,
            all_labels=labels,
            part_column=part_column,
        )
        out[clean] = positive
    return out


def resolve_all_category_keywords(
    categories: list[dict],
    *,
    corpus: list[str] | None = None,
    df: pd.DataFrame | None = None,
    text_column: str | None = None,
    label_column: str | None = None,
    label_mappings: dict[str, str] | None = None,
    token_map: dict[str, str] | None = None,
    part_column: str | None = None,
    max_terms: int = 28,
) -> dict[str, list[str]]:
    """Resolve keywords for every category in one pass (shared negative corpus)."""
    labels = [
        normalize_keyword_term(c.get("label"))
        for c in categories
        if normalize_keyword_term(c.get("label"))
    ]
    out: dict[str, list[str]] = {}
    for cat in categories:
        label = normalize_keyword_term(cat.get("label"))
        if not label:
            continue
        raw_kw = cat.get("keywords") or []
        if isinstance(raw_kw, str):
            user_kw = [k.strip() for k in raw_kw.split(",") if k.strip()]
        else:
            user_kw = [str(k).strip() for k in raw_kw if str(k).strip()]
        out[label] = resolve_category_keywords(
            label,
            user_keywords=user_kw,
            corpus=corpus,
            df=df,
            text_column=text_column,
            label_column=label_column,
            categories=categories,
            label_mappings=label_mappings,
            all_labels=labels,
            token_map=token_map,
            part_column=part_column,
            max_terms=max_terms,
        )

    positive_by_label = collect_positive_texts_by_label(
        labels,
        corpus=corpus,
        df=df,
        text_column=text_column,
        label_column=label_column,
        categories=categories,
        label_mappings=label_mappings,
        part_column=part_column,
    )
    return dedupe_cross_class_keywords(out, positive_by_label)


def generate_dataset_keywords_for_label(
    label: str,
    corpus: list[str] | None = None,
    max_terms: int = 28,
    **kwargs,
) -> list[str]:
    return resolve_category_keywords(
        label,
        corpus=corpus,
        max_terms=max_terms,
        df=kwargs.get("df"),
        text_column=kwargs.get("text_column"),
        label_column=kwargs.get("label_column"),
        categories=kwargs.get("categories"),
        label_mappings=kwargs.get("label_mappings"),
        all_labels=kwargs.get("all_labels"),
    )


def generate_keywords_for_label(
    label: str,
    corpus: list[str] | None = None,
    existing: dict[str, list[str]] | None = None,
    max_terms: int = 28,
    *,
    dataset_only: bool = True,
    **kwargs,
) -> list[str]:
    _ = existing, dataset_only
    return generate_dataset_keywords_for_label(label, corpus=corpus, max_terms=max_terms, **kwargs)


def generate_keywords_for_labels(
    labels: list[str],
    corpus: list[str] | None = None,
    failure_keywords_path: str | None = None,
    *,
    dataset_only: bool = True,
    **kwargs,
) -> dict[str, list[str]]:
    _ = failure_keywords_path, dataset_only
    categories = kwargs.get("categories") or [{"label": lb, "keywords": []} for lb in labels]
    mappings = kwargs.get("label_mappings")
    if categories and kwargs.get("df") is not None:
        return resolve_all_category_keywords(
            categories,
            corpus=corpus,
            df=kwargs.get("df"),
            text_column=kwargs.get("text_column"),
            label_column=kwargs.get("label_column"),
            label_mappings=mappings,
            token_map=kwargs.get("token_map"),
            max_terms=kwargs.get("max_terms", 28),
        )
    out: dict[str, list[str]] = {}
    all_labels = labels
    for label in labels:
        clean = normalize_keyword_term(label)
        if not clean:
            continue
        out[clean] = generate_dataset_keywords_for_label(
            clean,
            corpus=corpus,
            all_labels=all_labels,
            **kwargs,
        )
    return out
