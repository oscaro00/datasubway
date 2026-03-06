from __future__ import annotations

from functools import reduce

import polars as pl


def _strip_table_prefix(col: str) -> str:
    """'geography.country' -> 'country', 'country' -> 'country'"""
    return col.rsplit(".", 1)[-1]


def _build_leaf_expr(condition: tuple, strip_prefixes: bool = True) -> pl.Expr:
    col, op, value = condition
    col_name = _strip_table_prefix(col) if strip_prefixes else col
    col_expr = pl.col(col_name)
    match op:
        case "=":
            return col_expr == value
        case "!=":
            return col_expr != value
        case ">":
            return col_expr > value
        case ">=":
            return col_expr >= value
        case "<":
            return col_expr < value
        case "<=":
            return col_expr <= value
        case "in":
            return col_expr.is_in(value)
        case "not in":
            return ~col_expr.is_in(value)
        case "is null":
            return col_expr.is_null()
        case "is not null":
            return col_expr.is_not_null()
        case _:
            raise ValueError(f"Unsupported filter operator: {op!r}")


def build_filter_expr(spec: dict | tuple, strip_prefixes: bool = True) -> pl.Expr:
    """Convert a filter/having specification to a polars expression."""
    if isinstance(spec, tuple):
        return _build_leaf_expr(spec, strip_prefixes=strip_prefixes)
    if "AND" in spec:
        return reduce(
            lambda a, b: a & b,
            [build_filter_expr(s, strip_prefixes=strip_prefixes) for s in spec["AND"]],
        )
    if "OR" in spec:
        return reduce(
            lambda a, b: a | b,
            [build_filter_expr(s, strip_prefixes=strip_prefixes) for s in spec["OR"]],
        )
    raise ValueError(
        f"Invalid filter spec; expected tuple or dict with 'AND'/'OR', got: {spec!r}"
    )


def extract_table_columns_from_filter_dict(spec: dict | tuple) -> list[str]:
    """Recursively extract all 'table.column' references from a filter spec."""
    if isinstance(spec, tuple):
        return [spec[0]]
    if "AND" in spec:
        return [
            col
            for s in spec["AND"]
            for col in extract_table_columns_from_filter_dict(s)
        ]
    if "OR" in spec:
        return [
            col for s in spec["OR"] for col in extract_table_columns_from_filter_dict(s)
        ]
    raise ValueError(
        f"Invalid filter spec; expected tuple or dict with 'AND'/'OR', got: {spec!r}"
    )
