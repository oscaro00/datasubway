"""Column context resolution: allow() and exclude() functions.

These filter columns based on QueryContext filters/groups, determining
which columns a measure should operate on.
"""

from __future__ import annotations

import re

_TABLE_COL_RE = re.compile(r"^([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)$")
_PATTERN_RE = re.compile(r"^(\*|[a-zA-Z0-9_]+)\.(\*|[a-zA-Z0-9_]+)$")


def parse_table_column(table_column_str: str) -> tuple[str, str]:
    """Parse 'table.column' into (table, column). Raises on invalid format."""
    m = _TABLE_COL_RE.match(table_column_str)
    if not m:
        raise ValueError(
            f"Invalid table.column format: '{table_column_str}'. "
            f"Expected 'table_name.column_name'."
        )
    return m.group(1), m.group(2)


def parse_table_columns(
    table_column_list: list[str] | dict,
) -> list[tuple[str, str]]:
    """Parse a list of 'table.column' strings or extract columns from a filter dict."""
    if isinstance(table_column_list, dict):
        return [
            parse_table_column(c)
            for c in _extract_columns_from_filter(table_column_list)
        ]
    return [parse_table_column(tc) for tc in table_column_list]


def parse_pattern(pattern_str: str) -> tuple[str, str]:
    """Parse a pattern like '*', 'orders.*', '*.amount', 'orders.amount'.
    Returns (table_pattern, column_pattern) where '*' means match any.
    """
    if pattern_str == "*":
        return ("*", "*")
    m = _PATTERN_RE.match(pattern_str)
    if not m:
        raise ValueError(
            f"Invalid pattern: '{pattern_str}'. "
            f"Expected '*', 'table.*', '*.col', or 'table.col'."
        )
    return m.group(1), m.group(2)


def parse_patterns(pattern_list: list[str]) -> list[tuple[str, str]]:
    """Parse a list of pattern strings."""
    return [parse_pattern(p) for p in pattern_list]


class _AllowExcludeResult(list):
    """List subclass that carries metadata about the allow/exclude operation."""

    def __init__(
        self,
        items: list[str],
        ae_type: str,
        pattern: str | list[str],
        include: str | list[str] | None,
    ):
        super().__init__(items)
        self.ae_type = ae_type
        self.pattern = pattern
        self.include = include


def allow(
    pattern: str | list[str],
    context: str | list[str] | dict,
    include: str | list[str] | None = None,
    *,
    include_tables: bool = True,
) -> _AllowExcludeResult:
    """Return columns from context that match the pattern.

    Args:
        pattern: Pattern(s) to match against. '*' matches all.
        context: Column references (list of 'table.col') or filter dict.
        include: Additional columns to always include.
        include_tables: If True, return 'table.col'; if False, return 'col' only.
    """
    context_cols = _normalize_context(context)
    patterns = _normalize_patterns(pattern)
    include_cols = _normalize_include(include)

    matched = _filter_columns(context_cols, patterns, mode="allow")
    all_cols = list(
        dict.fromkeys(matched + include_cols)
    )  # deduplicate, preserve order

    if not include_tables:
        all_cols = [c.split(".", 1)[1] if "." in c else c for c in all_cols]

    return _AllowExcludeResult(all_cols, "allow", pattern, include or [])


def exclude(
    pattern: str | list[str],
    context: str | list[str] | dict,
    include: str | list[str] | None = None,
    *,
    include_tables: bool = True,
) -> _AllowExcludeResult:
    """Return columns from context that do NOT match the pattern.

    Args:
        pattern: Pattern(s) to exclude. '*' excludes all.
        context: Column references (list of 'table.col') or filter dict.
        include: Additional columns to always include regardless.
        include_tables: If True, return 'table.col'; if False, return 'col' only.
    """
    context_cols = _normalize_context(context)
    patterns = _normalize_patterns(pattern)
    include_cols = _normalize_include(include)

    matched = _filter_columns(context_cols, patterns, mode="exclude")
    all_cols = list(dict.fromkeys(matched + include_cols))

    if not include_tables:
        all_cols = [c.split(".", 1)[1] if "." in c else c for c in all_cols]

    return _AllowExcludeResult(all_cols, "exclude", pattern, include or [])


# ── Internal helpers ──


def _normalize_context(context: str | list[str] | dict) -> list[str]:
    """Normalize context to a flat list of 'table.col' strings."""
    if isinstance(context, str):
        return [context]
    if isinstance(context, dict):
        return _extract_columns_from_filter(context)
    return list(context)


def _normalize_patterns(pattern: str | list[str]) -> list[tuple[str, str]]:
    """Normalize pattern(s) to parsed (table_pat, col_pat) tuples."""
    if isinstance(pattern, str):
        return [parse_pattern(pattern)]
    return parse_patterns(pattern)


def _normalize_include(include: str | list[str] | None) -> list[str]:
    """Normalize include to a list of column strings."""
    if include is None:
        return []
    if isinstance(include, str):
        return [include]
    return list(include)


def _matches(col_str: str, patterns: list[tuple[str, str]]) -> bool:
    """Check if a 'table.col' string matches any of the parsed patterns."""
    table, col = parse_table_column(col_str)
    for tpat, cpat in patterns:
        t_match = tpat == "*" or tpat == table
        c_match = cpat == "*" or cpat == col
        if t_match and c_match:
            return True
    return False


def _filter_columns(
    columns: list[str], patterns: list[tuple[str, str]], mode: str
) -> list[str]:
    """Filter columns by pattern. mode='allow' keeps matches; mode='exclude' keeps non-matches."""
    if mode == "allow":
        return [c for c in columns if _matches(c, patterns)]
    else:
        return [c for c in columns if not _matches(c, patterns)]


def _extract_columns_from_filter(filter_dict: dict) -> list[str]:
    """Recursively extract column names from a filter tree.
    Format: {"AND": [("table.col", "op", value), ...]} or nested AND/OR.
    """
    columns = []
    for key, value in filter_dict.items():
        if key in ("AND", "OR"):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        columns.append(item[0])
                    elif isinstance(item, dict):
                        columns.extend(_extract_columns_from_filter(item))
    return columns
