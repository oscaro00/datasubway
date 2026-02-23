from __future__ import annotations

from functools import reduce

import polars as pl


def _strip_table_prefix(col: str) -> str:
    """'geography.country' -> 'country', 'country' -> 'country'"""
    return col.rsplit(".", 1)[-1]


def _build_leaf_expr(condition: tuple) -> pl.Expr:
    col, op, value = condition
    col_expr = pl.col(_strip_table_prefix(col))
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


def build_filter_expr(spec: dict | tuple) -> pl.Expr:
    """Convert a filter/having specification to a polars expression."""
    if isinstance(spec, tuple):
        return _build_leaf_expr(spec)
    if "AND" in spec:
        return reduce(lambda a, b: a & b, [build_filter_expr(s) for s in spec["AND"]])
    if "OR" in spec:
        return reduce(lambda a, b: a | b, [build_filter_expr(s) for s in spec["OR"]])
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


if __name__ == "__main__":
    df = pl.DataFrame(
        {
            "country": ["US", "CA", "UK", "CA"],
            "revenue": [500, 1500, 200, 800],
        }
    )

    print("DataFrame:")
    print(df)
    print()

    # Simple equality
    expr = build_filter_expr(("country", "=", "US"))
    print("country = 'US':")
    print(df.filter(expr))
    print()

    # in operator
    expr = build_filter_expr(("country", "in", ["US", "CA"]))
    print("country in ['US', 'CA']:")
    print(df.filter(expr))
    print()

    # AND
    expr = build_filter_expr({"AND": [("country", "=", "CA"), ("revenue", ">", 1000)]})
    print("country = 'CA' AND revenue > 1000:")
    print(df.filter(expr))
    print()

    # OR with nested AND
    expr = build_filter_expr(
        {
            "OR": [
                ("country", "=", "US"),
                {"AND": [("country", "=", "CA"), ("revenue", ">", 1000)]},
            ]
        }
    )
    print("country = 'US' OR (country = 'CA' AND revenue > 1000):")
    print(df.filter(expr))
