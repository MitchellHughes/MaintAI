"""Collapse fine-grained categories into evaluation groups (and optional category merge)."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from umec.evaluation.category_matching import (
    category_labels,
    normalize_label,
    parse_keywords,
    parse_reference_aliases,
)


def normalize_label_groups(groups: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Validate and dedupe label group definitions."""
    normalized: list[dict[str, Any]] = []
    assigned_members: set[str] = set()

    for group in groups or []:
        group_label = normalize_label(group.get("group_label"))
        if not group_label:
            continue

        members: list[str] = []
        for raw in group.get("members") or []:
            member = normalize_label(raw)
            if not member or member == group_label or member in assigned_members:
                continue
            members.append(member)
            assigned_members.add(member)

        reason = str(group.get("reason") or "").strip()
        entry: dict[str, Any] = {
            "group_label": group_label,
            "members": sorted(set(members)),
        }
        if reason:
            entry["reason"] = reason
        normalized.append(entry)

    return normalized


def build_group_applier(
    groups: Iterable[dict[str, Any]] | None,
) -> Callable[[str], str]:
    """Map a matched category label to its group label (identity when ungrouped)."""
    member_to_group: dict[str, str] = {}
    for group in normalize_label_groups(groups):
        group_label = group["group_label"]
        member_to_group[group_label] = group_label
        for member in group["members"]:
            member_to_group[member] = group_label

    def apply(label: str) -> str:
        norm = normalize_label(label)
        if not norm:
            return norm
        return member_to_group.get(norm, norm)

    return apply


def evaluation_labels(
    categories: Iterable[dict[str, Any]],
    groups: Iterable[dict[str, Any]] | None,
) -> list[str]:
    """Target label list after removing grouped members and adding group heads."""
    normalized_groups = normalize_label_groups(groups)
    grouped_members = {member for group in normalized_groups for member in group["members"]}
    group_heads = {group["group_label"] for group in normalized_groups}

    labels: list[str] = []
    for label in category_labels(categories):
        if label in grouped_members:
            continue
        labels.append(label)

    for head in sorted(group_heads):
        if head not in labels:
            labels.append(head)

    return sorted(set(labels))


def apply_groups_to_pairs(
    y_true: list[str],
    y_pred: list[str],
    groups: Iterable[dict[str, Any]] | None,
) -> tuple[list[str], list[str]]:
    applier = build_group_applier(groups)
    return [applier(y) for y in y_true], [applier(y) for y in y_pred]


def merge_categories_by_groups(
    categories: list[dict[str, Any]],
    groups: Iterable[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Collapse member categories into group categories with merged keywords."""
    normalized_groups = normalize_label_groups(groups)
    if not normalized_groups:
        return [dict(cat) for cat in categories]

    member_to_group = {
        member: group["group_label"]
        for group in normalized_groups
        for member in group["members"]
    }
    group_heads = {group["group_label"] for group in normalized_groups}

    merged: dict[str, dict[str, Any]] = {}
    standalone: list[dict[str, Any]] = []

    for cat in categories:
        label = normalize_label(cat.get("label"))
        if not label:
            continue

        if label in member_to_group:
            target = member_to_group[label]
            bucket = merged.setdefault(
                target,
                {
                    "label": target,
                    "keywords": set(),
                    "reference_aliases": set(),
                },
            )
            bucket["keywords"].update(parse_keywords(cat))
            bucket["keywords"].add(label)
            bucket["reference_aliases"].update(parse_reference_aliases(cat))
            bucket["reference_aliases"].add(label)
            continue

        if label in group_heads:
            bucket = merged.setdefault(
                label,
                {
                    "label": str(cat.get("label") or label),
                    "keywords": set(),
                    "reference_aliases": set(),
                },
            )
            bucket["keywords"].update(parse_keywords(cat))
            bucket["reference_aliases"].update(parse_reference_aliases(cat))
            continue

        standalone.append(dict(cat))

    for group in normalized_groups:
        head = group["group_label"]
        if head in merged:
            merged[head]["label"] = head

    grouped_categories: list[dict[str, Any]] = []
    for head in sorted(merged):
        bucket = merged[head]
        entry: dict[str, Any] = {
            "label": bucket["label"],
            "keywords": ", ".join(sorted(bucket["keywords"])),
        }
        aliases = sorted(bucket["reference_aliases"])
        if aliases:
            entry["reference_aliases"] = aliases
        grouped_categories.append(entry)

    return standalone + grouped_categories


def apply_label_groups_config(
    categories: list[dict[str, Any]],
    groups: Iterable[dict[str, Any]] | None,
    *,
    merge_categories: bool = False,
) -> dict[str, Any]:
    normalized = normalize_label_groups(groups)
    return {
        "label_groups": normalized,
        "custom_categories": (
            merge_categories_by_groups(categories, normalized)
            if merge_categories
            else [dict(cat) for cat in categories]
        ),
    }
