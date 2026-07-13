"""Shared inference asset cache."""

from __future__ import annotations

_CACHE: dict[str, dict[str, object]] = {}


def get_cache() -> dict[str, dict[str, object]]:
    return _CACHE


def clear_cache() -> None:
    _CACHE.clear()
