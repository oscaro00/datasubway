"""Column context resolution: allow() and exclude() functions.

Delegates to Rust implementation (src/model/column_context.rs).
"""

from __future__ import annotations

from datasubway._engine import allow as _rust_allow
from datasubway._engine import exclude as _rust_exclude


def allow(
    pattern: str | list[str],
    context: str | list[str] | dict,
    include: str | list[str] | None = None,
) -> list[str]:
    """Return columns from context that match the pattern."""
    context_cols = _normalize_context(context)
    include_list = _normalize_include(include) or None
    return _rust_allow(pattern, context_cols, include_list)


def exclude(
    pattern: str | list[str],
    context: str | list[str] | dict,
    include: str | list[str] | None = None,
) -> list[str]:
    """Return columns from context that do NOT match the pattern."""
    context_cols = _normalize_context(context)
    include_list = _normalize_include(include) or None
    return _rust_exclude(pattern, context_cols, include_list)


def _normalize_context(context: str | list[str] | dict) -> list[str]:
    if isinstance(context, str):
        return [context]
    if isinstance(context, dict):
        return _extract_columns_from_filter(context)
    return list(context)


def _normalize_include(include: str | list[str] | None) -> list[str]:
    if include is None:
        return []
    if isinstance(include, str):
        return [include]
    return list(include)


def _extract_columns_from_filter(filter_dict: dict) -> list[str]:
    columns = []
    for key, value in filter_dict.items():
        if key in ("AND", "OR") and isinstance(value, list):
            for item in value:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    columns.append(item[0])
                elif isinstance(item, dict):
                    columns.extend(_extract_columns_from_filter(item))
    return columns
