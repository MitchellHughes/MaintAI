"""Decode classifier scores into labels and calibrated confidence (likelihood)."""

from __future__ import annotations

import numpy as np
import pandas as pd

UNCLASSIFIED_LABEL = "unclassified"
MIN_CLASS_SCORE = 1e-9

# Review workflow tiers (auto-accept vs manual review).
TIER_HIGH = "high"
TIER_MEDIUM = "medium"
TIER_LOW = "low"
TIER_REVIEW = "review"
HIGH_CONFIDENCE_CUTOFF = 0.82
MEDIUM_CONFIDENCE_CUTOFF = 0.45


def softmax_probs(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    shifted = scores - float(np.max(scores))
    exp = np.exp(shifted)
    return exp / max(float(exp.sum()), 1e-9)


def scores_row_to_probs(
    scores_row: pd.Series,
    *,
    restrict_to: set[str] | list[str] | None = None,
) -> dict[str, float]:
    """Softmax probabilities; optionally only over a subset of labels."""
    row = scores_row
    if restrict_to:
        allowed = {str(x).lower() for x in restrict_to}
        keep = [c for c in scores_row.index if str(c).lower() in allowed]
        if not keep:
            return {}
        row = scores_row[keep]
    values = row.values.astype(float)
    probs = softmax_probs(values)
    return {str(label): float(p) for label, p in zip(row.index, probs)}


def label_confidence(
    scores_row: pd.Series,
    label: str,
    *,
    restrict_to: set[str] | list[str] | None = None,
) -> float:
    """
    Interpretable confidence for the assigned label.

    - With restrict_to (e.g. keyword-supported labels): share among those labels only.
    - Otherwise: top-two comparison — “how much stronger is the winner vs runner-up?”
    """
    if scores_row.empty or str(label).lower() == UNCLASSIFIED_LABEL:
        return 0.0

    label_key = str(label)
    if restrict_to:
        probs = scores_row_to_probs(scores_row, restrict_to=restrict_to)
        if label_key in probs:
            return probs[label_key]
        for k, v in probs.items():
            if str(k).lower() == label_key.lower():
                return v
        return 0.0

    top2 = scores_row.nlargest(2)
    if top2.empty:
        return 0.0
    if len(top2) == 1:
        return 1.0 if str(top2.index[0]).lower() == label_key.lower() else 0.0

    first_label, second_label = str(top2.index[0]), str(top2.index[1])
    pair_probs = scores_row_to_probs(top2)
    winner_prob = pair_probs.get(first_label, 0.0)

    if label_key.lower() == first_label.lower():
        return winner_prob
    if label_key.lower() == second_label.lower():
        return pair_probs.get(second_label, 1.0 - winner_prob)

    # Picked label not in top two (e.g. token override): use its share among top-3 scores.
    top3 = scores_row.nlargest(min(3, len(scores_row)))
    if any(str(c).lower() == label_key.lower() for c in top3.index):
        probs3 = scores_row_to_probs(top3)
        for k, v in probs3.items():
            if str(k).lower() == label_key.lower():
                return v * 0.85
    return float(scores_row_to_probs(scores_row).get(label_key, 0.0))


def runner_up_for_label(scores_row: pd.Series, label: str) -> str | None:
    """Second-place class by score (for display)."""
    if scores_row.empty:
        return None
    label_key = str(label).lower()
    others = scores_row[[c for c in scores_row.index if str(c).lower() != label_key]]
    if others.empty:
        return None
    return str(others.idxmax())


def _agreement_ratio(
    label: str,
    base_votes: list[str] | list[tuple[str, str]] | None,
) -> float:
    """Agreement on the final label; token+equipment unanimous counts as strong agreement."""
    if not base_votes:
        return 0.0

    keyed: list[tuple[str, str]] = []
    for item in base_votes:
        if isinstance(item, tuple) and len(item) == 2:
            keyed.append((str(item[0]), str(item[1])))
        else:
            keyed.append(("model", str(item)))

    label_key = str(label).lower()
    all_votes = [v for _, v in keyed]
    n_all = len(all_votes)
    agree_all = sum(1 for v in all_votes if str(v).lower() == label_key) / max(n_all, 1)

    core = [(k, v) for k, v in keyed if k in ("token", "equipment")]
    if len(core) >= 2:
        agree_core = sum(1 for _, v in core if str(v).lower() == label_key) / len(core)
        if agree_core >= 1.0:
            return max(agree_all, 0.9)
    return agree_all


def _best_label_margin(
    label: str,
    *score_rows: pd.Series | None,
    restrict_to: set[str] | list[str] | None = None,
) -> float:
    """Highest top-2 margin for this label across base score rows (not raw ensemble share)."""
    best = 0.0
    for row in score_rows:
        if row is None or row.empty:
            continue
        best = max(best, label_confidence(row, label, restrict_to=restrict_to))
    return best


def _token_equipment_match(label: str, base_votes: list[tuple[str, str]] | None) -> tuple[bool, bool]:
    if not base_votes:
        return False, False
    label_key = str(label).lower()
    token_ok = False
    equip_ok = False
    for key, vote in base_votes:
        if str(key) == "token" and str(vote).lower() == label_key:
            token_ok = True
        if str(key) == "equipment" and str(vote).lower() == label_key:
            equip_ok = True
    return token_ok, equip_ok


def review_confidence(
    scores_row: pd.Series,
    label: str,
    *,
    restrict_to: set[str] | list[str] | None = None,
    base_votes: list[str] | list[tuple[str, str]] | None = None,
    has_keyword_hits: bool = False,
    models_agree: int | None = None,
    models_total: int | None = None,
    extra_score_rows: tuple[pd.Series | None, ...] = (),
) -> tuple[float, str, str | None]:
    """
    Review triage confidence — driven by model agreement and token/equipment scores.

    Ensemble ECOC margins are not used as a probability share (they are often ~10–25%
    even when the correct label is clear). Keyword + token agreement is the main signal.
    """
    if str(label).lower() == UNCLASSIFIED_LABEL:
        return 0.0, TIER_REVIEW, runner_up_for_label(scores_row, label)

    keyed: list[tuple[str, str]] | None = None
    if base_votes:
        keyed = []
        for item in base_votes:
            if isinstance(item, tuple) and len(item) == 2:
                keyed.append((str(item[0]), str(item[1])))
            else:
                keyed.append(("model", str(item)))

    margin = _best_label_margin(label, scores_row, *extra_score_rows, restrict_to=restrict_to)
    agree_ratio = _agreement_ratio(label, keyed) if keyed else 0.5
    combined = 0.15 * margin + 0.85 * agree_ratio

    token_ok, equip_ok = _token_equipment_match(label, keyed)
    if has_keyword_hits and token_ok:
        combined = max(combined, 0.78)
    if token_ok and equip_ok:
        combined = max(combined, 0.9)
    elif token_ok and models_agree is not None and models_agree >= 2:
        combined = max(combined, 0.85)
    elif models_total and models_agree == models_total and models_total >= 2:
        combined = max(combined, 0.88)

    calibrated = min(0.99, combined)

    if calibrated >= HIGH_CONFIDENCE_CUTOFF:
        tier = TIER_HIGH
    elif calibrated >= MEDIUM_CONFIDENCE_CUTOFF:
        tier = TIER_MEDIUM
    else:
        tier = TIER_LOW

    runner_up = runner_up_for_label(scores_row, label)
    if combined >= 0.75:
        runner_up = None
    return calibrated, tier, runner_up


def top_k_predictions(
    scores_row: pd.Series,
    k: int = 3,
    *,
    primary_label: str | None = None,
    restrict_to: set[str] | list[str] | None = None,
) -> list[dict[str, float | str]]:
    """Top-k labels by raw score with per-label confidence (excludes zero-evidence rows)."""
    if scores_row.empty or float(scores_row.max()) <= MIN_CLASS_SCORE:
        return []

    row = scores_row
    if restrict_to:
        allowed = {str(x).lower() for x in restrict_to}
        keep = [c for c in scores_row.index if str(c).lower() in allowed]
        if not keep:
            return []
        row = scores_row[keep]
        if float(row.max()) <= MIN_CLASS_SCORE:
            return []

    ordered = row.sort_values(ascending=False)
    results: list[dict[str, float | str]] = []
    seen: set[str] = set()

    if primary_label and str(primary_label).lower() != UNCLASSIFIED_LABEL:
        pk = str(primary_label)
        if pk in ordered.index and float(ordered[pk]) > MIN_CLASS_SCORE:
            conf = label_confidence(scores_row, pk, restrict_to=None)
            results.append({"label": pk, "confidence": float(conf)})
            seen.add(pk.lower())

    for label in ordered.index:
        if len(results) >= k:
            break
        lab = str(label)
        if lab.lower() in seen:
            continue
        if float(ordered[label]) <= MIN_CLASS_SCORE:
            break
        conf = label_confidence(scores_row, lab, restrict_to=None)
        results.append({"label": lab, "confidence": float(conf)})
        seen.add(lab.lower())

    return results[:k]


def tier_display_label(tier: str) -> str:
    return {
        TIER_HIGH: "Auto-accept",
        TIER_MEDIUM: "Spot-check",
        TIER_LOW: "Manual review",
        TIER_REVIEW: "Needs review",
    }.get(tier, tier)


def decode_from_scores(
    scores_row: pd.Series,
    *,
    require_positive_score: bool = True,
) -> tuple[str, float, dict[str, float]]:
    """
    Pick argmax class with softmax likelihood as confidence.

    Returns (label, confidence in [0,1], all_class_probs).
    """
    if scores_row.empty:
        return UNCLASSIFIED_LABEL, 0.0, {}

    probs = scores_row_to_probs(scores_row)
    raw = scores_row.values.astype(float)
    max_raw = float(np.max(raw)) if raw.size else 0.0

    if require_positive_score and max_raw <= MIN_CLASS_SCORE:
        return UNCLASSIFIED_LABEL, 0.0, probs

    label = str(scores_row.idxmax())
    confidence, tier, _ = review_confidence(scores_row, label)
    return label, confidence, probs


def has_label_evidence(xai: dict) -> bool:
    """True when explainability found supporting terms for the label."""
    if not xai:
        return False
    if xai.get("contributions"):
        return True
    explanation = str(xai.get("explanation", "")).strip()
    lower = explanation.lower()
    if not explanation or lower == "n/a":
        return False
    if "no keyword evidence" in lower:
        return False
    # Formatted contribution strings (e.g. dirty=100.0%)
    if "=" in explanation and "%" in explanation:
        return True
    return bool(xai.get("keyword")) and str(xai.get("keyword")).lower() != "n/a"


def evidence_for_label(xai: dict, label: str) -> bool:
    """Check UMEC base entries for evidence supporting a specific predicted label."""
    bases = xai.get("bases") or []
    if not bases:
        return has_label_evidence(xai)
    for base in bases:
        if str(base.get("ensemble_label", base.get("predicted_condition", ""))).lower() != str(
            label
        ).lower():
            continue
        if base.get("contributions"):
            return True
        if has_label_evidence(base):
            return True
    return False


def ensemble_is_ambiguous(scores_row: pd.Series, *, margin: float = 0.06) -> bool:
    """True when ensemble scores are nearly tied (uses softmax probs when spread is flat)."""
    if scores_row.empty:
        return True
    probs = sorted(scores_row_to_probs(scores_row).values(), reverse=True)
    if not probs or probs[0] <= MIN_CLASS_SCORE:
        return True
    if len(probs) < 2:
        return False
    if (probs[0] - probs[1]) < margin:
        return True
    # Flat ECOC margins with many classes often peak below 35% probability share.
    return probs[0] < 0.38


def token_clear_winner(
    token_row: pd.Series | None,
    *,
    min_absolute: float = 0.12,
    min_margin_ratio: float = 0.2,
) -> str | None:
    """Return token-matching argmax when it leads the runner-up by a clear margin."""
    if token_row is None or token_row.empty:
        return None
    ordered = token_row.sort_values(ascending=False)
    top_label = str(ordered.index[0])
    top = float(ordered.iloc[0])
    if top <= MIN_CLASS_SCORE:
        return None
    second = float(ordered.iloc[1]) if len(ordered) > 1 else 0.0
    if top >= min_absolute and (top - second) >= min_margin_ratio * top:
        return top_label
    return None


def pick_label_with_evidence(
    scores_row: pd.Series,
    supported_labels: set[str] | list[str],
    token_scores_row: pd.Series | None = None,
) -> tuple[str, float, dict[str, float]]:
    """
    Choose a label that has strict keyword evidence.

    Prefer the token-matching argmax among supported labels, then highest ensemble margin.
    """
    probs = scores_row_to_probs(scores_row)
    if not supported_labels:
        return UNCLASSIFIED_LABEL, 0.0, probs

    allowed = {str(l).lower() for l in supported_labels}
    eligible = [str(c) for c in scores_row.index if str(c).lower() in allowed]
    if not eligible:
        return UNCLASSIFIED_LABEL, 0.0, probs

    if token_scores_row is not None and not token_scores_row.empty:
        token_eligible = [
            str(c)
            for c in token_scores_row.index
            if str(c).lower() in allowed and float(token_scores_row[c]) > 0
        ]
        if token_eligible:
            best = max(token_eligible, key=lambda c: float(token_scores_row[c]))
            conf, _, _ = review_confidence(
                token_scores_row,
                best,
                restrict_to=eligible,
                has_keyword_hits=True,
            )
            return best, conf, probs

    best = max(eligible, key=lambda c: float(scores_row[c]))
    conf, _, _ = review_confidence(
        scores_row, best, restrict_to=eligible, has_keyword_hits=True
    )
    return best, conf, probs
