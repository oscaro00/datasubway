"""Functions for applying post-aggregation query modifiers."""

from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from datasubway.query_context import QueryContext


def having_to_polars(having_expr: Any) -> pl.Expr:
    """
    Convert having expression to Polars boolean expression.
    Uses same syntax as filter clause.

    Args:
        having_expr: Having expression (same format as filter)

    Returns:
        Polars boolean expression for filtering

    Examples:
        ('total_revenue', '>', 1000) -> pl.col('total_revenue') > 1000
        {'AND': [('total_revenue', '>', 1000), ('count', '>=', 10)]}
            -> (pl.col('total_revenue') > 1000) & (pl.col('count') >= 10)
    """
    from datasubway.column_context import OPERATOR_MAP

    # Simple condition (tuple)
    if isinstance(having_expr, tuple):
        column, operator, value = having_expr
        # Strip table prefix if present (post-agg columns might not have prefixes)
        column_name = column.split('.')[-1] if '.' in column else column
        col_expr = pl.col(column_name)

        if operator not in OPERATOR_MAP:
            raise ValueError(f"Unsupported operator in having clause: {operator}")

        return OPERATOR_MAP[operator](col_expr, value)

    # AND/OR dict
    if isinstance(having_expr, dict):
        key = next(iter(having_expr.keys()))
        conditions = having_expr[key]

        if key == 'AND':
            # Combine with &
            result = having_to_polars(conditions[0])
            for cond in conditions[1:]:
                result = result & having_to_polars(cond)
            return result

        elif key == 'OR':
            # Combine with |
            result = having_to_polars(conditions[0])
            for cond in conditions[1:]:
                result = result | having_to_polars(cond)
            return result

    raise ValueError(f"Invalid having expression: {having_expr}")


def apply_query_modifiers(
    lazy_frame: pl.LazyFrame,
    qc: 'QueryContext'
) -> pl.LazyFrame:
    """
    Apply post-aggregation query modifiers to combined result.

    Applies modifiers in SQL-compliant order:
    1. HAVING - filter aggregated results
    2. ORDER BY - sort results
    3. LIMIT/OFFSET - slice results

    Args:
        lazy_frame: Combined LazyFrame from combine_measure_results
        qc: QueryContext with having, sort, limit, offset parameters

    Returns:
        Modified LazyFrame with all post-aggregation operations applied

    Raises:
        ValueError: If having/sort columns don't exist in result
    """
    result = lazy_frame

    # Step 1: Apply HAVING clause (post-aggregation filtering)
    having = qc.context.get('having')
    if having is not None:
        try:
            having_expr = having_to_polars(having)
            result = result.filter(having_expr)
        except Exception as e:
            raise ValueError(f"Failed to apply having clause: {e}") from e

    # Step 2: Apply ORDER BY (sorting)
    sort = qc.context.get('sort')
    if sort is not None and len(sort) > 0:
        try:
            # Extract column names and directions
            columns = [col_name for col_name, _ in sort]
            # Build descending list: True for 'desc', False for 'asc'
            descending = [direction == 'desc' for _, direction in sort]

            # Strip table prefixes from column names
            # (post-aggregation, columns may not have table prefixes)
            clean_columns = [
                col.split('.')[-1] if '.' in col else col
                for col in columns
            ]

            result = result.sort(by=clean_columns, descending=descending)
        except Exception as e:
            raise ValueError(f"Failed to apply sort: {e}") from e

    # Step 3: Apply LIMIT and OFFSET (slicing)
    limit = qc.context.get('limit', 10000)
    offset = qc.context.get('offset', 0)

    # Only apply slice if limit is positive and meaningful
    if limit is not None and limit > 0:
        result = result.slice(offset=offset, length=limit)
    elif offset > 0:
        # Edge case: offset without limit means "skip first N rows, return all remaining"
        result = result.slice(offset=offset, length=None)

    return result
