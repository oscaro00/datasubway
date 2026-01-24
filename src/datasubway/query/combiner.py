"""Functions for combining multiple measure results."""

from typing import List, Optional

import polars as pl


def combine_measure_results(
    measure_results: List[pl.LazyFrame],
    group_by_cols: Optional[List[str]]
) -> pl.LazyFrame:
    """
    Combine multiple measure results via outer join or cross join.

    Args:
        measure_results: List of LazyFrames from different measures
        group_by_cols: Columns to join on (from query_context['group']), or None

    Returns:
        Combined LazyFrame
    """
    if len(measure_results) == 1:
        return measure_results[0]

    result = measure_results[0]
    for i, subsequent in enumerate(measure_results[1:], 1):
        if group_by_cols:
            # Outer join on group by columns with coalesce to avoid duplicate join columns
            result = result.join(subsequent, on=group_by_cols, how='full', coalesce=True, suffix=f'_{i}')
        else:
            # Cross join (cartesian product) when no group by
            result = result.join(subsequent, how='cross', suffix=f'_{i}')

    return result
