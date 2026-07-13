"""Map free-text CMMS labels to engineer-defined categories (matches web UI logic)."""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable


def normalize_label(value: Any) -> str:
    return str(value or "").strip().lower()


def parse_keywords(category: dict[str, Any]) -> list[str]:
    raw = category.get("keywords")
    if isinstance(raw, list):
        return [normalize_label(k) for k in raw if normalize_label(k)]
    return [normalize_label(k) for k in re.split(r"[,;]", str(raw or "")) if normalize_label(k)]


def parse_reference_aliases(category: dict[str, Any]) -> list[str]:
    """Extra CMMS reference strings that should map to this category."""
    raw = category.get("reference_aliases")
    if isinstance(raw, list):
        return [normalize_label(a) for a in raw if normalize_label(a)]
    return [normalize_label(a) for a in re.split(r"[,;]", str(raw or "")) if normalize_label(a)]


def build_category_matcher(
    categories: Iterable[dict[str, Any]],
    *,
    label_groups: Iterable[dict[str, Any]] | None = None,
) -> Callable[[Any], str | None]:
    entries: list[tuple[str, list[str]]] = []
    registered: set[str] = set()

    for cat in categories or []:
        label = normalize_label(cat.get("label"))
        if not label:
            continue
        aliases = parse_keywords(cat) + parse_reference_aliases(cat)
        if label not in aliases:
            aliases = [label, *aliases]
        entries.append((label, aliases))
        registered.add(label)

    if label_groups:
        from umec.evaluation.label_groups import normalize_label_groups

        for group in normalize_label_groups(label_groups):
            head = group["group_label"]
            if head in registered:
                continue
            aliases = [head, *(group.get("members") or [])]
            entries.append((head, aliases))
            registered.add(head)

    def match(raw: Any) -> str | None:
        text = normalize_label(raw)
        if not text or text == "unclassified":
            return None

        # Pass 1: exact category / group-head labels (CMMS codes like DETERIORATED).
        for label, _keywords in entries:
            if text == label:
                return label

        # Pass 2: exact keyword / alias hits.
        for label, keywords in entries:
            for kw in keywords:
                if kw and text == kw:
                    return label

        # Pass 3: substring fallback for narrative text (e.g. mined phrases).
        for label, keywords in entries:
            for kw in keywords:
                if not kw:
                    continue
                if kw in text or text in kw:
                    return label
        return None

    return match


def category_labels(categories: Iterable[dict[str, Any]]) -> list[str]:
    return [normalize_label(c.get("label")) for c in categories or [] if normalize_label(c.get("label"))]
