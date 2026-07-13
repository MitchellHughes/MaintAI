"""Inference helpers: per-model prediction and explainability."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

from umec.data.preprocessing import normalize_tokens, preprocess_dataframe
from umec.models.equipment_based import EquipmentBasedClassifier
from umec.models.semantic_similarity import SemanticSimilarityClassifier
from umec.models.token_matching import TokenMatchingClassifier
from umec.models.umec import UMECClassifier
from umec.prediction.decode import (
    MIN_CLASS_SCORE,
    UNCLASSIFIED_LABEL,
    decode_from_scores,
    ensemble_is_ambiguous,
    has_label_evidence,
    pick_label_with_evidence,
    review_confidence,
    scores_row_to_probs,
    tier_display_label,
    token_clear_winner,
    top_k_predictions,
)
from umec.explainability.text_highlights import (
    build_text_highlights,
    filter_highlight_terms,
    surface_terms_in_text,
)
from umec.prediction.keywords import (
    SEMANTIC_STOPWORDS,
    is_strong_keyword_hit,
    keyword_hit_counts_as_evidence,
    label_negated_in_text,
    normalized_keywords_for_label,
)

logger = logging.getLogger(__name__)

MODEL_TOKEN = "TokenMatchingClassifier"
MODEL_EQUIPMENT = "EquipmentBasedClassifier"
MODEL_SEMANTIC = "SemanticSimilarityClassifier"
MODEL_UMEC = "UMECClassifier"

_XAI_TOP_K = 3


def _empty_xai() -> dict[str, Any]:
    return {"keyword": "n/a", "explanation": "n/a", "contributions": []}


def _slim_row_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Drop heavy per-row fields to keep large-job memory bounded."""
    xai = row.get("xai") or {}
    simple = xai.get("simple") or {}
    slim_simple = {
        "tier": simple.get("tier"),
        "tier_label": simple.get("tier_label"),
        "confidence": simple.get("confidence"),
        "one_liner": simple.get("one_liner"),
        "keywords": simple.get("keywords"),
        "runner_up": simple.get("runner_up"),
        "top_ranked": simple.get("top_ranked"),
        "top_k_details": simple.get("top_k_details"),
        "text_spans": simple.get("text_spans"),
        "models_agree": simple.get("models_agree"),
        "models_total": simple.get("models_total"),
    }
    return {
        "row_id": row.get("row_id"),
        "discrepancy": row.get("discrepancy"),
        "component": row.get("component"),
        "predicted_condition": row.get("predicted_condition"),
        "top_predictions": row.get("top_predictions"),
        "confidence": row.get("confidence"),
        "confidence_tier": row.get("confidence_tier"),
        "runner_up": row.get("runner_up"),
        "models_agree": row.get("models_agree"),
        "models_total": row.get("models_total"),
        "actual_label": row.get("actual_label"),
        "model": row.get("model"),
        "xai": {
            "simple": slim_simple,
            "explanation": simple.get("one_liner") or xai.get("explanation"),
        },
    }


def _token_classifier_fitted(token_clf: TokenMatchingClassifier | None) -> bool:
    return (
        token_clf is not None
        and token_clf.vectorizer is not None
        and getattr(token_clf.vectorizer, "vocabulary_", None) is not None
    )


def _no_evidence_xai() -> dict[str, Any]:
    return {
        "keyword": "n/a",
        "explanation": "No keyword evidence for this label in the text.",
        "contributions": [],
    }


def _build_keyword_index(token_clf: TokenMatchingClassifier, failure_keywords: dict) -> dict[str, list[tuple[str, int]]]:
    if not _token_classifier_fitted(token_clf):
        return {}
    token_vocab = token_clf.vectorizer.vocabulary_
    token_map = token_clf.token_map
    keyword_index: dict[str, list[tuple[str, int]]] = {}
    for label, keywords in failure_keywords.items():
        idxs = []
        for kw in keywords:
            norm_kw = normalize_tokens(str(kw).lower().strip(), token_map)
            if norm_kw in token_vocab:
                idxs.append((kw, token_vocab[norm_kw]))
        keyword_index[str(label)] = idxs
    return keyword_index


def _filter_xai_to_text(xai: dict[str, Any], text: str) -> dict[str, Any]:
    """Keep only contribution terms that appear as words in the source discrepancy."""
    contributions = xai.get("contributions") or []
    if not text or not contributions:
        return xai
    surfaced = set(surface_terms_in_text(text, [str(c.get("term", "")) for c in contributions]))
    if not surfaced:
        return {**xai, "contributions": []}
    filtered = [c for c in contributions if str(c.get("term", "")).lower() in surfaced]
    if not filtered:
        return {**xai, "contributions": []}
    return _pack_xai([(str(c["term"]), float(c.get("weight") or 0.0)) for c in filtered])


def _pack_xai(weights: list[tuple[str, float]], *, prefix: str = "") -> dict[str, Any]:
    if not weights:
        return _empty_xai()
    merged: dict[str, float] = {}
    for term, weight in weights:
        key = str(term).strip()
        if not key:
            continue
        merged[key] = merged.get(key, 0.0) + float(weight)
    weights = list(merged.items())
    weights.sort(key=lambda x: x[1], reverse=True)
    top = weights[:_XAI_TOP_K]
    total_w = sum(w for _, w in top) or 1.0
    contributions = [
        {"term": term, "weight": w, "pct": round(w / total_w * 100.0, 1)} for term, w in top
    ]
    formatted = "|".join(f"{c['term']}={c['pct']}%" for c in contributions)
    if prefix:
        formatted = f"{prefix}{formatted}"
    return {
        "keyword": str(top[0][0]),
        "explanation": formatted,
        "contributions": contributions,
    }


def _format_class_scores(scores_row: pd.Series | None, *, highlight: str | None = None) -> str:
    if scores_row is None or scores_row.empty:
        return ""
    probs = scores_row_to_probs(scores_row)
    parts = []
    for cls, val in sorted(probs.items(), key=lambda item: item[1], reverse=True):
        mark = "*" if highlight and str(cls).lower() == str(highlight).lower() else ""
        parts.append(f"{mark}{cls}={val:.1%}")
    return "|".join(parts)


def _xai_from_keyword_weights(
    keyword_index: dict[str, list[tuple[str, int]]],
    row_vec,
    label: str,
) -> dict[str, Any]:
    kw_idxs = keyword_index.get(str(label), [])
    if not kw_idxs:
        return _empty_xai()

    weights: list[tuple[str, float]] = []
    for kw, kw_idx in kw_idxs:
        val = row_vec[0, kw_idx]
        if val > 0:
            weights.append((kw, float(val)))
    return _pack_xai(weights)


def _label_keyword_set(token_clf: TokenMatchingClassifier, label: str) -> set[str]:
    return normalized_keywords_for_label(
        token_clf.failure_keywords,
        label,
        token_clf.token_map,
        normalize=token_clf.config.normalize_tokens,
    )


def _token_strict_support(
    token_clf: TokenMatchingClassifier,
    tfidf_matrix,
    row_pos: int,
    label: str,
    text: str = "",
) -> bool:
    """True only if this label's own keywords appear in the row (not shared-token bleed)."""
    if label_negated_in_text(text, label):
        return False
    allowed = _label_keyword_set(token_clf, label)
    if not allowed or token_clf.vectorizer is None:
        return False
    vocab = token_clf.vectorizer.vocabulary_
    row = tfidf_matrix[row_pos]
    for kw in allowed:
        if not is_strong_keyword_hit(label, kw):
            continue
        idx = vocab.get(kw)
        if idx is not None and float(row[0, idx]) > 0:
            return True
    return False


def _semantic_strict_support(semantic_clf: SemanticSimilarityClassifier, text: str, label: str) -> bool:
    if label_negated_in_text(text, label):
        return False
    allowed = normalized_keywords_for_label(semantic_clf.failure_keywords, label, normalize=False)
    if not allowed or semantic_clf.embedding_model is None:
        return False
    for token in semantic_clf._tokenize(text):
        if token in SEMANTIC_STOPWORDS or len(token) < 2:
            continue
        if token in allowed and token in semantic_clf.embedding_model.wv:
            if keyword_hit_counts_as_evidence(label, token, text):
                return True
    return False


def _equipment_strict_support(
    equipment_clf: EquipmentBasedClassifier,
    equip_tfidf,
    row_pos: int,
    label: str,
    text: str,
) -> bool:
    if label_negated_in_text(text, label):
        return False
    xai = explain_equipment_based_row(equipment_clf, equip_tfidf, row_pos, label)
    if not xai.get("contributions"):
        return False
    if str(label).lower() == "leaking":
        terms = [str(c.get("term", "")) for c in xai.get("contributions") or []]
        return any(is_strong_keyword_hit(label, t) for t in terms)
    return True


def explain_token_matching_row(
    keyword_index: dict[str, list[tuple[str, int]]],
    tfidf_matrix,
    row_pos: int,
    label: str,
    token_clf: TokenMatchingClassifier | None = None,
    text: str = "",
) -> dict[str, Any]:
    if label_negated_in_text(text, label):
        return _no_evidence_xai()

    row_vec = tfidf_matrix[row_pos]
    xai = _xai_from_keyword_weights(keyword_index, row_vec, label)
    if xai.get("contributions"):
        filtered = [
            c
            for c in xai["contributions"]
            if is_strong_keyword_hit(label, str(c.get("term", "")))
        ]
        if filtered:
            xai = _pack_xai([(c["term"], c["weight"]) for c in filtered])
        else:
            xai = _empty_xai()
    if xai.get("contributions"):
        return _filter_xai_to_text(xai, text)
    # Only show terms from this label's keyword list (never cross-class mapping).
    if token_clf is not None and _token_strict_support(
        token_clf, tfidf_matrix, row_pos, label, text=text
    ):
        allowed = _label_keyword_set(token_clf, label)
        row = row_vec
        vocab = token_clf.vectorizer.vocabulary_
        weights = []
        for kw in allowed:
            if not is_strong_keyword_hit(label, kw):
                continue
            idx = vocab.get(kw)
            if idx is not None:
                val = float(row[0, idx])
                if val > 0:
                    weights.append((kw, val))
        packed = _pack_xai(weights)
        if packed.get("contributions"):
            return _filter_xai_to_text(packed, text)
    return _no_evidence_xai()


def explain_equipment_based_row(
    equipment_clf: EquipmentBasedClassifier,
    tfidf_matrix,
    row_pos: int,
    label: str,
    scores_row: pd.Series | None = None,
    text: str = "",
) -> dict[str, Any]:
    if label_negated_in_text(text, label):
        return _no_evidence_xai()

    if str(label) in (UNCLASSIFIED_LABEL, ""):
        return _no_evidence_xai()

    if str(label) not in equipment_clf.classes:
        return _no_evidence_xai()

    part_act = equipment_clf._part_activation(tfidf_matrix[row_pos])
    class_idx = equipment_clf.classes.index(str(label))
    weights: list[tuple[str, float]] = []
    for part_idx, part in enumerate(equipment_clf.part_names):
        prominence = equipment_clf.part_to_class_weights[part_idx, class_idx]
        contrib = float(part_act[part_idx] * prominence)
        if contrib > 0:
            weights.append((part, contrib))

    if str(label).lower() == "leaking":
        weights = [(t, w) for t, w in weights if is_strong_keyword_hit(label, t)]

    xai = _pack_xai(weights)
    if not xai.get("contributions"):
        return _no_evidence_xai()
    return _filter_xai_to_text(xai, text)


def explain_semantic_similarity(
    semantic_clf: SemanticSimilarityClassifier,
    text: str,
    label: str,
    scores_row: pd.Series | None = None,
) -> dict[str, Any]:
    if str(label) in (UNCLASSIFIED_LABEL, ""):
        return _no_evidence_xai()

    if label_negated_in_text(text, label):
        return _no_evidence_xai()

    if semantic_clf.embedding_model is None or semantic_clf.class_prototypes is None:
        return _no_evidence_xai()

    prototype = semantic_clf.class_prototypes.get(str(label))
    if prototype is None:
        return _no_evidence_xai()

    allowed = normalized_keywords_for_label(semantic_clf.failure_keywords, label, normalize=False)
    tokens = semantic_clf._tokenize(text)
    weights: list[tuple[str, float]] = []
    for token in tokens:
        if not token or not token.isalpha() or len(token) < 2:
            continue
        if token in SEMANTIC_STOPWORDS:
            continue
        if allowed and token not in allowed:
            continue
        if not keyword_hit_counts_as_evidence(label, token, text):
            continue
        if token not in semantic_clf.embedding_model.wv:
            continue
        vec = semantic_clf.embedding_model.wv[token]
        sim = float(np.dot(vec, prototype) / (np.linalg.norm(vec) * np.linalg.norm(prototype) + 1e-9))
        if sim > 0:
            weights.append((token, sim))

    xai = _pack_xai(weights)
    return xai if xai.get("contributions") else _no_evidence_xai()


def _append_ranked_predictions(
    target: list[dict[str, Any]],
    candidates: list[dict[str, float | str]],
    *,
    text: str,
    seen: set[str],
    k: int,
    evidence_backed: bool,
) -> None:
    for item in candidates:
        if len(target) >= k:
            return
        lab = str(item["label"])
        if label_negated_in_text(text, lab):
            continue
        key = lab.lower()
        if key in seen:
            continue
        entry: dict[str, Any] = {
            "label": lab,
            "confidence": float(item["confidence"]),
        }
        if not evidence_backed:
            entry["evidence_backed"] = False
        target.append(entry)
        seen.add(key)


def _row_top_predictions(
    label: str,
    scores_row: pd.Series,
    text: str,
    *,
    restrict_to: set[str] | None = None,
    k: int = 3,
) -> list[dict[str, Any]]:
    """
    Top-K for review: evidence-backed ranks first, then pad from full ensemble
    scores so engineers still see alternates when only one class has keyword hits.
    """
    restricted = top_k_predictions(
        scores_row, k=k, primary_label=label, restrict_to=restrict_to
    )
    full = (
        top_k_predictions(scores_row, k=k, primary_label=label, restrict_to=None)
        if restrict_to
        else restricted
    )

    filtered: list[dict[str, Any]] = []
    seen: set[str] = set()
    _append_ranked_predictions(
        filtered, restricted, text=text, seen=seen, k=k, evidence_backed=True
    )
    if len(filtered) < k:
        _append_ranked_predictions(
            filtered, full, text=text, seen=seen, k=k, evidence_backed=False
        )

    if not filtered and str(label).lower() != UNCLASSIFIED_LABEL:
        filtered = [{"label": str(label), "confidence": 0.0, "evidence_backed": True}]
    return filtered[:k]


def _apply_token_priority_when_ambiguous(
    label: str,
    ensemble_row: pd.Series,
    token_row: pd.Series | None,
    *,
    supported: set[str] | None,
    require_keyword_evidence: bool,
    token_clf: TokenMatchingClassifier | None = None,
    token_tfidf=None,
    row_pos: int | None = None,
    text: str = "",
) -> str:
    """
    Prefer token-matching when ECOC scores are flat but keywords are clear.

    Fixes cases like «broken» in the text where token matching is right but the
    ensemble spreads mass across unrelated classes (odor/worn/unsecure/…).
    """
    token_label = token_clear_winner(token_row, min_absolute=0.05, min_margin_ratio=0.15)
    if not token_label and token_row is not None and not token_row.empty:
        if float(token_row.max()) > MIN_CLASS_SCORE:
            token_label = str(token_row.idxmax())

    if not token_label:
        return label

    allowed = {str(s).lower() for s in (supported or [])}
    if require_keyword_evidence and token_label.lower() not in allowed:
        return label

    has_strict = (
        token_clf is not None
        and token_tfidf is not None
        and row_pos is not None
        and _token_strict_support(token_clf, token_tfidf, row_pos, token_label, text=text)
    )

    ambiguous = ensemble_is_ambiguous(ensemble_row)
    disagrees = str(label).lower() != token_label.lower()

    if ambiguous and (has_strict or not require_keyword_evidence):
        return token_label

    if disagrees and has_strict and token_row is not None:
        token_score = float(token_row.get(token_label, 0.0))
        ens_score = float(ensemble_row.get(token_label, 0.0))
        if ambiguous or token_score >= max(ens_score, MIN_CLASS_SCORE):
            return token_label

    return label


def _resolve_label_after_negation(
    label: str,
    text: str,
    scores_row: pd.Series,
    supported: set[str] | None = None,
) -> str:
    """Drop labels contradicted by negation phrases (e.g. NO LEAKS → not leaking)."""
    if not label_negated_in_text(text, label):
        return label
    pool = list(supported) if supported else [str(c) for c in scores_row.index]
    pool = [
        str(c)
        for c in pool
        if str(c).lower() not in (str(label).lower(), UNCLASSIFIED_LABEL)
        and not label_negated_in_text(text, c)
    ]
    if not pool:
        return UNCLASSIFIED_LABEL
    return max(pool, key=lambda c: float(scores_row.get(c, 0)))


def _collect_supported_labels(
    candidate_labels: list[str],
    *,
    token_clf: TokenMatchingClassifier | None,
    keyword_index: dict | None,
    token_tfidf,
    row_pos: int,
    equipment_clf: EquipmentBasedClassifier | None,
    equip_tfidf,
    semantic_clf: SemanticSimilarityClassifier | None,
    text: str,
) -> set[str]:
    """Labels with at least one strict keyword/part hit for this row."""
    supported: set[str] = set()
    for label in candidate_labels:
        clean = str(label).strip().lower()
        if not clean or clean == UNCLASSIFIED_LABEL:
            continue
        if _token_classifier_fitted(token_clf) and token_tfidf is not None:
            if _token_strict_support(token_clf, token_tfidf, row_pos, label, text=text):
                supported.add(str(label))
        if equipment_clf is not None and equip_tfidf is not None:
            if _equipment_strict_support(equipment_clf, equip_tfidf, row_pos, label, text):
                supported.add(str(label))
        if semantic_clf is not None and _semantic_strict_support(semantic_clf, text, label):
            supported.add(str(label))
    return supported


def _collect_xai_terms(bases: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for base in bases:
        for item in base.get("contributions") or []:
            term = str(item.get("term", "")).strip()
            if not term or term.lower() in seen:
                continue
            seen.add(term.lower())
            terms.append(term)
    return terms


def _attach_simple_xai(
    xai: dict[str, Any],
    *,
    label: str,
    confidence: float,
    tier: str,
    runner_up: str | None,
    bases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    bases = bases if bases is not None else xai.get("bases") or []
    n_total = len(bases)
    n_agree = sum(1 for b in bases if b.get("agrees_with_ensemble"))
    keywords = _collect_xai_terms(bases)
    if not keywords and xai.get("contributions"):
        keywords = [str(c.get("term", "")) for c in xai["contributions"] if c.get("term")]

    parts: list[str] = []
    if n_total:
        parts.append(f"{n_agree} of {n_total} models agree on «{label}»")
    if keywords:
        parts.append(f"Matched: {', '.join(keywords[:4])}")
    if runner_up:
        parts.append(f"Next choice: {runner_up}")

    xai["simple"] = {
        "tier": tier,
        "tier_label": tier_display_label(tier),
        "confidence": confidence,
        "models_agree": n_agree,
        "models_total": n_total,
        "keywords": keywords[:6],
        "runner_up": runner_up,
        "one_liner": (
            " · ".join(parts)
            if parts
            else "Weak or conflicting signals — please review manually."
        ),
    }
    xai["explanation"] = xai["simple"]["one_liner"]
    return xai


def _apply_text_explainability(
    xai: dict[str, Any],
    *,
    text: str,
    label: str,
    confidence: float,
    runner_up: str | None,
    top_predictions: list[dict[str, Any]],
    failure_keywords: dict[str, list[str]] | None,
    xai_top_k: int = 3,
) -> dict[str, Any]:
    """Attach inline text spans and a concise review-friendly summary."""
    bases = xai.get("bases") or []
    raw_terms = _collect_xai_terms(bases)
    if xai.get("contributions"):
        raw_terms.extend(str(c.get("term", "")) for c in xai["contributions"] if c.get("term"))
    label_key = str(label).strip().lower()
    allowed_support: set[str] = set()
    for cat_label, kws in (failure_keywords or {}).items():
        if str(cat_label).strip().lower() != label_key:
            continue
        for kw in kws or []:
            allowed_support.update(surface_terms_in_text(text, [kw]))
    for base in bases:
        for item in base.get("contributions") or []:
            allowed_support.update(surface_terms_in_text(text, [str(item.get("term", ""))]))

    support_pool = filter_highlight_terms(sorted(allowed_support), label=label)

    highlights = build_text_highlights(
        text,
        predicted_label=label,
        top_predictions=top_predictions[:xai_top_k],
        failure_keywords=failure_keywords,
        support_terms=support_pool,
    )

    simple = xai.setdefault("simple", {})
    simple["keywords"] = highlights["support_terms"][:6]
    simple["alternate_keywords"] = highlights["alternate_terms"][:4]
    simple["other_terms"] = highlights["other_terms"][:4]
    simple["text_spans"] = highlights["spans"]
    simple["top_ranked"] = [
        {"label": str(item["label"]), "confidence": float(item.get("confidence") or 0.0)}
        for item in (top_predictions or [])[:xai_top_k]
    ]

    parts: list[str] = []
    label_key = str(label).strip().lower()
    if label_key == UNCLASSIFIED_LABEL:
        parts.append("No class had clear keyword support — assign manually.")
    else:
        parts.append(f"Predicted «{label}» ({round(confidence * 100)}% confidence)")
        if highlights["support_terms"]:
            parts.append(f"Evidence in text: {', '.join(highlights['support_terms'][:4])}")
        alts = [
            item
            for item in simple["top_ranked"]
            if str(item["label"]).strip().lower() not in ("", label_key, UNCLASSIFIED_LABEL)
        ]
        if alts:
            alt = alts[0]
            parts.append(
                f"Next: «{alt['label']}» ({round(float(alt['confidence']) * 100)}%)"
            )
        elif runner_up and str(runner_up).lower() != label_key:
            parts.append(f"Next: «{runner_up}»")
        if highlights["other_terms"]:
            parts.append(f"Other terms: {', '.join(highlights['other_terms'][:2])}")

    if parts:
        simple["one_liner"] = " · ".join(parts)
        xai["explanation"] = simple["one_liner"]

    simple["top_k_details"] = _build_top_k_details(
        text,
        top_predictions=top_predictions,
        failure_keywords=failure_keywords,
        xai_top_k=xai_top_k,
    )
    return xai


def _keywords_for_label_in_text(
    text: str,
    label: str,
    failure_keywords: dict[str, list[str]] | None,
) -> list[str]:
    label_key = str(label).strip().lower()
    pool: set[str] = set()
    for cat_label, kws in (failure_keywords or {}).items():
        if str(cat_label).strip().lower() != label_key:
            continue
        for kw in kws or []:
            pool.update(surface_terms_in_text(text, [kw]))
    if label_key == "leaking":
        return [t for t in sorted(pool) if is_strong_keyword_hit(label, t)]
    return sorted(pool)


def _build_top_k_details(
    text: str,
    *,
    top_predictions: list[dict[str, Any]],
    failure_keywords: dict[str, list[str]] | None,
    xai_top_k: int,
) -> list[dict[str, Any]]:
    """Per-rank label evidence for the review table."""
    details: list[dict[str, Any]] = []
    for rank, item in enumerate((top_predictions or [])[:xai_top_k], start=1):
        lab = str(item.get("label", "")).strip()
        if not lab or lab.lower() == UNCLASSIFIED_LABEL:
            continue
        support = _keywords_for_label_in_text(text, lab, failure_keywords)
        highlights = build_text_highlights(
            text,
            predicted_label=lab,
            top_predictions=top_predictions,
            failure_keywords=failure_keywords,
            support_terms=support,
        )
        details.append(
            {
                "rank": rank,
                "label": lab,
                "confidence": float(item.get("confidence") or 0.0),
                "keywords": highlights["support_terms"][:8],
                "text_spans": highlights["spans"],
                "evidence_backed": item.get("evidence_backed", True),
            }
        )
    return details


def _row_component(row: pd.Series, part_column: str | None) -> str:
    if not part_column or part_column not in row.index:
        return ""
    return str(row.get(part_column) or "").strip()


def _score_row_argmax(row: pd.Series | None) -> str | None:
    if row is None or row.empty or float(row.max()) <= MIN_CLASS_SCORE:
        return None
    return str(row.idxmax())


def _base_votes_from_scores(
    token_row: pd.Series | None,
    equip_row: pd.Series | None,
    semantic_row: pd.Series | None,
) -> list[tuple[str, str]]:
    votes: list[tuple[str, str]] = []
    token_vote = _score_row_argmax(token_row)
    if token_vote:
        votes.append(("token", token_vote))
    equip_vote = _score_row_argmax(equip_row)
    if equip_vote:
        votes.append(("equipment", equip_vote))
    sem_vote = _score_row_argmax(semantic_row)
    if sem_vote:
        votes.append(("semantic", sem_vote))
    return votes


def _finalize_row_confidence(
    label: str,
    scores_row: pd.Series,
    xai: dict[str, Any],
    *,
    restrict_to: set[str] | None = None,
    base_votes: list[tuple[str, str]] | None = None,
    extra_score_rows: tuple[pd.Series | None, ...] = (),
    boost_equipment: bool = False,
) -> tuple[float, str, str | None, dict[str, Any]]:
    bases = xai.get("bases") or []
    n_agree = sum(1 for b in bases if b.get("agrees_with_ensemble"))
    n_total = len(bases)
    has_hits = bool(_collect_xai_terms(bases)) or bool(xai.get("contributions"))
    confidence, tier, runner_up = review_confidence(
        scores_row,
        label,
        restrict_to=restrict_to,
        base_votes=base_votes,
        has_keyword_hits=has_hits,
        models_agree=n_agree,
        models_total=n_total,
        extra_score_rows=extra_score_rows,
    )
    if boost_equipment and base_votes:
        label_key = str(label).lower()
        for model_key, vote in base_votes:
            if model_key == "equipment" and str(vote).lower() == label_key:
                confidence = min(1.0, float(confidence) + 0.08)
                break
    xai = _attach_simple_xai(
        xai,
        label=label,
        confidence=confidence,
        tier=tier,
        runner_up=runner_up,
        bases=xai.get("bases"),
    )
    if xai.get("ensemble"):
        xai["ensemble"]["confidence"] = confidence
    return confidence, tier, runner_up, xai


def _base_xai_entry(
    model_key: str,
    model_label: str,
    ensemble_label: str,
    support_xai: dict[str, Any],
    scores_row: pd.Series | None,
) -> dict[str, Any]:
    base_probs = scores_row_to_probs(scores_row) if scores_row is not None and not scores_row.empty else {}
    if scores_row is not None and not scores_row.empty:
        base_label, base_conf, _ = decode_from_scores(scores_row, require_positive_score=False)
    else:
        base_label, base_conf = UNCLASSIFIED_LABEL, 0.0

    support_text = str(support_xai.get("explanation", "n/a"))
    if not support_xai.get("contributions"):
        support_text = f"No keyword evidence for «{ensemble_label}»"

    agrees = str(base_label).lower() == str(ensemble_label).lower()

    return {
        "model": model_key,
        "model_label": model_label,
        "base_vote": base_label,
        "base_confidence": base_conf,
        "ensemble_label": ensemble_label,
        "agrees_with_ensemble": agrees,
        "vote_note": "",
        "keyword": support_xai.get("keyword", "n/a"),
        "explanation": support_text,
        "contributions": support_xai.get("contributions", []),
        "class_scores": base_probs,
    }


def explain_umec_row(
    ensemble_label: str,
    ensemble_confidence: float,
    class_scores_row: pd.Series,
    token_clf: TokenMatchingClassifier | None,
    keyword_index: dict[str, list[tuple[str, int]]] | None,
    token_tfidf,
    row_pos: int,
    equipment_clf: EquipmentBasedClassifier | None,
    equip_tfidf,
    semantic_clf: SemanticSimilarityClassifier | None,
    text: str,
    token_scores_row: pd.Series | None,
    equip_scores_row: pd.Series | None,
    semantic_scores_row: pd.Series | None,
) -> dict[str, Any]:
    ensemble_label = str(ensemble_label)
    explain_label = ensemble_label
    if explain_label == UNCLASSIFIED_LABEL and not class_scores_row.empty:
        explain_label = str(class_scores_row.idxmax())

    bases: list[dict[str, Any]] = []

    if _token_classifier_fitted(token_clf) and token_tfidf is not None:
        token_xai = explain_token_matching_row(
            keyword_index or {},
            token_tfidf,
            row_pos,
            explain_label,
            token_clf=token_clf,
            text=text,
        )
        bases.append(
            _base_xai_entry("token", MODEL_TOKEN, explain_label, token_xai, token_scores_row)
        )

    if equipment_clf is not None and equip_tfidf is not None:
        equip_xai = explain_equipment_based_row(
            equipment_clf,
            equip_tfidf,
            row_pos,
            explain_label,
            scores_row=equip_scores_row,
            text=text,
        )
        bases.append(
            _base_xai_entry("equipment", MODEL_EQUIPMENT, explain_label, equip_xai, equip_scores_row)
        )

    if semantic_clf is not None:
        sem_xai = explain_semantic_similarity(
            semantic_clf, text, explain_label, scores_row=semantic_scores_row
        )
        bases.append(
            _base_xai_entry("semantic", MODEL_SEMANTIC, explain_label, sem_xai, semantic_scores_row)
        )

    ensemble_probs = scores_row_to_probs(class_scores_row)
    top_scores = class_scores_row.nlargest(min(4, len(class_scores_row)))
    score_text = _format_class_scores(top_scores, highlight=ensemble_label)
    n_agree = sum(1 for b in bases if b.get("agrees_with_ensemble"))
    n_total = len(bases)
    terms = _collect_xai_terms(bases)

    if ensemble_label == UNCLASSIFIED_LABEL:
        summary = "No category keywords found in this text — assign a label manually."
    elif n_total:
        summary = f"{n_agree}/{n_total} models support «{ensemble_label}»."
        if terms:
            summary += f" Key terms: {', '.join(terms[:4])}."
    else:
        summary = f"Predicted «{ensemble_label}»."

    return {
        "keyword": ensemble_label if ensemble_label != UNCLASSIFIED_LABEL else "n/a",
        "explanation": summary,
        "contributions": [],
        "bases": bases,
        "ensemble": {
            "predicted_label": ensemble_label,
            "confidence": ensemble_confidence,
            "explanation": score_text,
            "class_scores": ensemble_probs,
        },
    }


def _token_texts(df: pd.DataFrame, text_column: str, token_clf: TokenMatchingClassifier) -> pd.Series:
    texts = df[text_column].fillna("").astype(str)
    if token_clf.config.normalize_tokens and token_clf.token_map:
        texts = texts.apply(lambda x: normalize_tokens(x, token_clf.token_map))
    return texts


def _equipment_texts(df: pd.DataFrame, text_column: str, equipment_clf: EquipmentBasedClassifier) -> pd.Series:
    texts = df[text_column].fillna("").astype(str)
    if equipment_clf.config.normalize_tokens and equipment_clf.token_map:
        texts = texts.apply(lambda x: normalize_tokens(x, equipment_clf.token_map))
    return texts


INFERENCE_CHUNK_THRESHOLD = 3_000
INFERENCE_CHUNK_SIZE = 4_000


def _inference_chunk_size(assets: dict[str, object]) -> int:
    umec = assets.get("umec")
    if umec is not None and hasattr(umec, "config"):
        size = int(getattr(umec.config, "predict_chunk_size", 0) or 0)
        if size > 0:
            return size
    return INFERENCE_CHUNK_SIZE


def _predict_with_model_frame(
    model_name: str,
    df: pd.DataFrame,
    text_column: str,
    source_text_column: str,
    assets: dict[str, object],
) -> list[dict[str, Any]]:
    require_kw = bool(assets.get("require_keyword_evidence", False))
    failure_keywords = assets.get("failure_keywords") or {}
    xai_top_k = int(assets.get("xai_top_k") or 3)
    slim_rows = bool(assets.get("slim_rows", False))
    part_column = assets.get("part_column")
    boost_equipment = bool(assets.get("boost_equipment"))
    token_clf = assets.get("token_clf")
    equipment_clf = assets.get("equipment_clf")
    semantic_clf = assets.get("semantic_clf")
    umec = assets.get("umec")
    keyword_index = assets.get("keyword_index")

    if model_name == MODEL_TOKEN:
        if token_clf is None:
            raise ValueError("Token matching model is not available.")
        scores = token_clf.transform(df, column_name=text_column)
        tfidf_matrix = token_clf.vectorizer.transform(_token_texts(df, text_column, token_clf))
        rows = _rows_from_scores(
            df,
            scores,
            source_text_column,
            MODEL_TOKEN,
            token_clf,
            keyword_index,
            text_column,
            token_tfidf=tfidf_matrix,
            require_keyword_evidence=require_kw,
            failure_keywords=failure_keywords,
            xai_top_k=xai_top_k,
            part_column=part_column,
            boost_equipment=boost_equipment,
        )
        return [_slim_row_payload(row) for row in rows] if slim_rows else rows

    if model_name == MODEL_EQUIPMENT:
        if equipment_clf is None:
            raise ValueError("Equipment-based model is not available.")
        scores = equipment_clf.transform(df, column_name=text_column)
        rows = _rows_from_scores(
            df,
            scores,
            source_text_column,
            MODEL_EQUIPMENT,
            equipment_clf,
            None,
            text_column,
            require_keyword_evidence=require_kw,
            failure_keywords=failure_keywords,
            xai_top_k=xai_top_k,
            part_column=part_column,
            boost_equipment=boost_equipment,
        )
        return [_slim_row_payload(row) for row in rows] if slim_rows else rows

    if model_name == MODEL_SEMANTIC:
        if semantic_clf is None:
            raise ValueError("Semantic similarity model is not available.")
        scores = semantic_clf.transform(df, column_name=text_column)
        rows = _rows_from_scores(
            df,
            scores,
            source_text_column,
            MODEL_SEMANTIC,
            semantic_clf,
            None,
            text_column,
            require_keyword_evidence=require_kw,
            failure_keywords=failure_keywords,
            xai_top_k=xai_top_k,
            part_column=part_column,
            boost_equipment=boost_equipment,
        )
        return [_slim_row_payload(row) for row in rows] if slim_rows else rows

    if model_name == MODEL_UMEC:
        if umec is None:
            raise ValueError("UMEC model is not available.")
        preds, reduction = umec.predict(df, column_name=text_column)
        class_scores = umec.class_score_df(reduction)
        rows = _rows_from_umec(
            df,
            preds,
            class_scores,
            source_text_column,
            umec,
            token_clf,
            equipment_clf,
            semantic_clf,
            keyword_index,
            text_column,
            require_keyword_evidence=require_kw,
            failure_keywords=failure_keywords,
            xai_top_k=xai_top_k,
            part_column=part_column,
            boost_equipment=boost_equipment,
        )
        return [_slim_row_payload(row) for row in rows] if slim_rows else rows

    raise ValueError(f"Unknown model: {model_name}")


def predict_with_model(
    model_name: str,
    df: pd.DataFrame,
    text_column: str,
    source_text_column: str,
    assets: dict[str, object],
    *,
    on_progress: object | None = None,
    on_chunk_rows: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    import logging
    import time

    from app.services.progress import emit_progress

    log = logging.getLogger(__name__)
    n = len(df)
    if n <= INFERENCE_CHUNK_THRESHOLD:
        rows = _predict_with_model_frame(model_name, df, text_column, source_text_column, assets)
        if on_chunk_rows is not None:
            on_chunk_rows(rows)
            return []
        return rows

    chunk_size = _inference_chunk_size(assets)
    total_chunks = max(1, (n + chunk_size - 1) // chunk_size)
    model_label = {
        MODEL_TOKEN: "Token matching",
        MODEL_EQUIPMENT: "Equipment based",
        MODEL_SEMANTIC: "Semantic similarity",
        MODEL_UMEC: "UMEC ensemble",
    }.get(model_name, model_name)

    log.info(
        "[%s] Chunked scoring started: %s rows in %s chunks of up to %s",
        model_name,
        f"{n:,}",
        total_chunks,
        f"{chunk_size:,}",
    )

    combined: list[dict[str, Any]] = []
    for chunk_index, start in enumerate(range(0, n, chunk_size), start=1):
        end = min(start + chunk_size, n)
        t_chunk = time.perf_counter()
        emit_progress(
            on_progress,
            {
                "phase": "chunk",
                "model": model_name,
                "model_label": model_label,
                "chunk": chunk_index,
                "chunks": total_chunks,
                "row_start": start + 1,
                "row_end": end,
                "row_count": n,
                "chunk_size": chunk_size,
            },
        )
        chunk = df.iloc[start:end]
        chunk_rows = _predict_with_model_frame(
            model_name,
            chunk,
            text_column,
            source_text_column,
            assets,
        )
        if on_chunk_rows is not None:
            on_chunk_rows(chunk_rows)
        else:
            combined.extend(chunk_rows)
        elapsed = time.perf_counter() - t_chunk
        log.info(
            "[%s] Chunk %s/%s done in %.1fs (rows %s–%s, %s predictions so far)",
            model_name,
            chunk_index,
            total_chunks,
            elapsed,
            f"{start + 1:,}",
            f"{end:,}",
            f"{(len(combined) if on_chunk_rows is None else end):,}",
        )
        emit_progress(
            on_progress,
            {
                "phase": "chunk",
                "model": model_name,
                "model_label": model_label,
                "chunk": chunk_index,
                "chunks": total_chunks,
                "row_start": start + 1,
                "row_end": end,
                "row_count": n,
                "chunk_size": chunk_size,
                "elapsed_seconds": elapsed,
            },
        )

    if on_chunk_rows is not None:
        log.info("[%s] Chunked scoring finished (streamed): %s rows", model_name, f"{n:,}")
        return []
    log.info("[%s] Chunked scoring finished: %s rows", model_name, f"{len(combined):,}")
    return combined


def _row_id(row: pd.Series, idx: int) -> int:
    row_id = row.get("id", idx)
    try:
        return int(row_id)
    except (TypeError, ValueError):
        return int(idx)


def _index_positions(df: pd.DataFrame) -> dict:
    return {idx: pos for pos, idx in enumerate(df.index)}


def _rows_from_scores(
    df: pd.DataFrame,
    scores: pd.DataFrame,
    source_text_column: str,
    model_name: str,
    clf: object,
    keyword_index: dict | None,
    text_column: str,
    token_tfidf=None,
    require_keyword_evidence: bool = False,
    failure_keywords: dict | None = None,
    xai_top_k: int = 3,
    part_column: str | None = None,
    boost_equipment: bool = False,
) -> list[dict[str, Any]]:
    positions = _index_positions(df)
    equip_tfidf = None

    if model_name == MODEL_EQUIPMENT and clf is not None:
        equip_tfidf = clf.vectorizer.transform(_equipment_texts(df, text_column, clf))

    rows: list[dict[str, Any]] = []
    for idx in scores.index:
        row = df.loc[idx]
        row_pos = positions[idx]
        scores_row = scores.loc[idx]
        supported: set[str] = set()
        if require_keyword_evidence:
            candidates = [str(c) for c in scores_row.index]
            supported = _collect_supported_labels(
                candidates,
                token_clf=clf if model_name == MODEL_TOKEN else None,
                keyword_index=keyword_index if model_name == MODEL_TOKEN else None,
                token_tfidf=token_tfidf if model_name == MODEL_TOKEN else None,
                row_pos=row_pos,
                equipment_clf=clf if model_name == MODEL_EQUIPMENT else None,
                equip_tfidf=equip_tfidf if model_name == MODEL_EQUIPMENT else None,
                semantic_clf=clf if model_name == MODEL_SEMANTIC else None,
                text=str(row.get(text_column, row.get(source_text_column, ""))),
            )
            token_row = scores_row if model_name == MODEL_TOKEN else None
            label, confidence, class_scores = pick_label_with_evidence(
                scores_row, supported, token_scores_row=token_row
            )
        else:
            label, confidence, class_scores = decode_from_scores(
                scores_row, require_positive_score=True
            )

        row_text = str(row.get(text_column, row.get(source_text_column, "")))
        label = _resolve_label_after_negation(
            label, row_text, scores_row, supported if require_keyword_evidence else None
        )

        if model_name == MODEL_TOKEN and token_tfidf is not None:
            xai = explain_token_matching_row(
                keyword_index or {},
                token_tfidf,
                row_pos,
                label,
                token_clf=clf if isinstance(clf, TokenMatchingClassifier) else None,
                text=row_text,
            )
        elif model_name == MODEL_EQUIPMENT and equip_tfidf is not None:
            xai = explain_equipment_based_row(
                clf, equip_tfidf, row_pos, label, scores_row=scores_row, text=row_text
            )
        elif model_name == MODEL_SEMANTIC:
            xai = explain_semantic_similarity(
                clf,
                row_text,
                label,
                scores_row=scores_row,
            )
        else:
            xai = _no_evidence_xai()

        restrict = supported if require_keyword_evidence else None
        confidence, tier, runner_up, xai = _finalize_row_confidence(
            label,
            scores_row,
            xai,
            restrict_to=restrict,
            base_votes=None,
            extra_score_rows=(scores_row,),
            boost_equipment=boost_equipment,
        )
        top_predictions = _row_top_predictions(
            label, scores_row, row_text, restrict_to=restrict, k=xai_top_k
        )
        discrepancy = str(row.get(source_text_column, ""))
        xai = _apply_text_explainability(
            xai,
            text=discrepancy,
            label=label,
            confidence=confidence,
            runner_up=runner_up,
            top_predictions=top_predictions,
            failure_keywords=failure_keywords,
            xai_top_k=xai_top_k,
        )
        simple = xai.get("simple") or {}

        rows.append(
            {
                "row_id": _row_id(row, idx),
                "discrepancy": discrepancy,
                "component": _row_component(row, part_column),
                "predicted_condition": label,
                "top_predictions": top_predictions,
                "confidence": float(confidence),
                "confidence_tier": tier,
                "runner_up": runner_up,
                "models_agree": simple.get("models_agree", 0),
                "models_total": simple.get("models_total", 0),
                "class_scores": class_scores,
                "xai": xai,
                "model": model_name,
            }
        )
    return rows


def _rows_from_umec(
    df: pd.DataFrame,
    preds: pd.Series,
    class_scores: pd.DataFrame,
    source_text_column: str,
    umec: UMECClassifier,
    token_clf: TokenMatchingClassifier | None,
    equipment_clf: EquipmentBasedClassifier | None,
    semantic_clf: SemanticSimilarityClassifier | None,
    keyword_index: dict | None,
    text_column: str,
    require_keyword_evidence: bool = False,
    failure_keywords: dict | None = None,
    xai_top_k: int = 3,
    part_column: str | None = None,
    boost_equipment: bool = False,
) -> list[dict[str, Any]]:
    positions = _index_positions(df)
    token_tfidf = None
    equip_tfidf = None
    token_scores = equip_scores = semantic_scores = None

    if _token_classifier_fitted(token_clf):
        token_tfidf = token_clf.vectorizer.transform(_token_texts(df, text_column, token_clf))
        token_scores = token_clf.transform(df, column_name=text_column)
    if equipment_clf is not None:
        equip_tfidf = equipment_clf.vectorizer.transform(_equipment_texts(df, text_column, equipment_clf))
        equip_scores = equipment_clf.transform(df, column_name=text_column)
    if semantic_clf is not None:
        semantic_scores = semantic_clf.transform(df, column_name=text_column)

    rows: list[dict[str, Any]] = []
    for idx in class_scores.index:
        row = df.loc[idx]
        row_pos = positions[idx]
        ensemble_row = class_scores.loc[idx]
        umec_label = str(preds.loc[idx])
        text = str(row.get(text_column, row.get(source_text_column, "")))
        candidates = [str(c) for c in ensemble_row.index]
        supported = _collect_supported_labels(
            candidates,
            token_clf=token_clf,
            keyword_index=keyword_index,
            token_tfidf=token_tfidf,
            row_pos=row_pos,
            equipment_clf=equipment_clf,
            equip_tfidf=equip_tfidf,
            semantic_clf=semantic_clf,
            text=text,
        )

        class_probs = scores_row_to_probs(ensemble_row)
        tok_row = token_scores.loc[idx] if token_scores is not None else None
        if umec_label == UNCLASSIFIED_LABEL:
            label = UNCLASSIFIED_LABEL
        elif require_keyword_evidence:
            label, _, class_probs = pick_label_with_evidence(
                ensemble_row, supported, token_scores_row=tok_row
            )
        else:
            label, _, class_probs = decode_from_scores(
                ensemble_row, require_positive_score=False
            )
            if umec_label != UNCLASSIFIED_LABEL:
                label = umec_label

        label = _apply_token_priority_when_ambiguous(
            label,
            ensemble_row,
            tok_row,
            supported=supported,
            require_keyword_evidence=require_keyword_evidence,
            token_clf=token_clf,
            token_tfidf=token_tfidf,
            row_pos=row_pos,
            text=text,
        )

        label = _resolve_label_after_negation(
            label, text, ensemble_row, supported if require_keyword_evidence else None
        )
        equip_row = equip_scores.loc[idx] if equip_scores is not None else None
        sem_row = semantic_scores.loc[idx] if semantic_scores is not None else None
        base_votes = _base_votes_from_scores(tok_row, equip_row, sem_row)

        xai = explain_umec_row(
            label,
            0.0,
            ensemble_row,
            token_clf,
            keyword_index,
            token_tfidf,
            row_pos,
            equipment_clf,
            equip_tfidf,
            semantic_clf,
            text,
            tok_row,
            equip_row,
            sem_row,
        )

        restrict = supported if require_keyword_evidence else None
        confidence, tier, runner_up, xai = _finalize_row_confidence(
            label,
            ensemble_row,
            xai,
            restrict_to=restrict,
            base_votes=base_votes,
            extra_score_rows=(tok_row, equip_row, sem_row),
            boost_equipment=boost_equipment,
        )
        top_predictions = _row_top_predictions(
            label, ensemble_row, text, restrict_to=restrict, k=xai_top_k
        )
        discrepancy = str(row.get(source_text_column, ""))
        xai = _apply_text_explainability(
            xai,
            text=discrepancy,
            label=label,
            confidence=confidence,
            runner_up=runner_up,
            top_predictions=top_predictions,
            failure_keywords=failure_keywords,
            xai_top_k=xai_top_k,
        )
        simple = xai.get("simple") or {}

        rows.append(
            {
                "row_id": _row_id(row, idx),
                "discrepancy": discrepancy,
                "component": _row_component(row, part_column),
                "predicted_condition": label,
                "top_predictions": top_predictions,
                "confidence": float(confidence),
                "confidence_tier": tier,
                "runner_up": runner_up,
                "models_agree": simple.get("models_agree", 0),
                "models_total": simple.get("models_total", 0),
                "class_scores": class_probs,
                "xai": xai,
                "model": MODEL_UMEC,
            }
        )
    return rows


def preprocess_rows(
    rows: list[dict],
    text_column: str,
    cfg,
    token_map,
) -> tuple[pd.DataFrame, str]:
    df = pd.DataFrame(rows)
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in uploaded data.")

    processed_column = cfg.data.text_column
    if cfg.data.preprocess.get("enabled", True):
        df = preprocess_dataframe(
            df,
            text_column=text_column,
            output_column=processed_column,
            preprocess_cfg=cfg.data.preprocess,
            token_map=token_map,
        )
    else:
        df[processed_column] = df[text_column].fillna("").astype(str)

    return df, processed_column
