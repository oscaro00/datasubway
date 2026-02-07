from __future__ import annotations

from contextvars import ContextVar
from typing import Any

import polars as pl

# Context variable to hold pre-agg metadata (None = normal mode)
_pre_agg_ctx: ContextVar[dict[str, Any] | None] = ContextVar('pre_agg_ctx', default=None)


class PreAggContext:
    """Context manager to activate pre-agg mode for expressions."""

    def __init__(self, metadata: dict[str, Any]):
        self.metadata = metadata
        self._token = None

    def __enter__(self):
        self._token = _pre_agg_ctx.set(self.metadata)
        return self

    def __exit__(self, *args):
        _pre_agg_ctx.reset(self._token)


@pl.api.register_expr_namespace("ds")
class DSExpr:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def _col_name(self) -> str:
        return self._expr.meta.output_name()

    def _is_pre_agg(self) -> bool:
        meta = _pre_agg_ctx.get()
        if meta is None:
            return False
        col = self._col_name()
        aggs = meta.get('aggregations', {})
        return col in aggs

    def sum(self) -> pl.Expr:
        if self._is_pre_agg():
            col = self._col_name()
            return pl.col(f'{col}-sum').sum()
        return self._expr.sum()

    def mean(self) -> pl.Expr:
        if self._is_pre_agg():
            col = self._col_name()
            return (
                pl.col(f'{col}-mean-sum').sum() /
                pl.col(f'{col}-mean-count').sum()
            ).alias(col)
        return self._expr.mean()

    def min(self) -> pl.Expr:
        if self._is_pre_agg():
            col = self._col_name()
            return pl.col(f'{col}-min').min()
        return self._expr.min()

    def max(self) -> pl.Expr:
        if self._is_pre_agg():
            col = self._col_name()
            return pl.col(f'{col}-max').max()
        return self._expr.max()

    def count(self) -> pl.Expr:
        if self._is_pre_agg():
            col = self._col_name()
            return pl.col(f'{col}-count').sum()
        return self._expr.count()

    def std(self) -> pl.Expr:
        if self._is_pre_agg():
            col = self._col_name()
            n = pl.col(f'{col}-std-count').sum()
            s = pl.col(f'{col}-std-sum').sum()
            sq = pl.col(f'{col}-std-sumsq').sum()
            return ((sq - s.pow(2) / n) / (n - 1)).sqrt().alias(col)
        return self._expr.std()

    def var(self) -> pl.Expr:
        if self._is_pre_agg():
            col = self._col_name()
            n = pl.col(f'{col}-var-count').sum()
            s = pl.col(f'{col}-var-sum').sum()
            sq = pl.col(f'{col}-var-sumsq').sum()
            return ((sq - s.pow(2) / n) / (n - 1)).alias(col)
        return self._expr.var()


if __name__ == "__main__":
    # Simulate raw data
    raw = pl.DataFrame({
        "region": ["A", "A", "B", "B", "B"],
        "revenue": [10, 20, 30, 40, 50],
    })

    # Simulate a pre-agg table (what PreAggManager would produce)
    pre_agg = pl.DataFrame({
        "region": ["A", "B"],
        "revenue-sum": [30, 120],
        "revenue-mean-sum": [30.0, 120.0],
        "revenue-mean-count": [2, 3],
    })

    metadata = {
        "aggregations": {"revenue": ["sum", "mean"]},
        "group_by": ["region"],
    }

    # --- Normal mode: single agg per column ---
    print("=== Normal mode: sum ===")
    result = raw.lazy().group_by("region").agg(
        pl.col("revenue").ds.sum(),
    ).sort("region").collect()
    print(result)

    print("\n=== Normal mode: mean ===")
    result = raw.lazy().group_by("region").agg(
        pl.col("revenue").ds.mean(),
    ).sort("region").collect()
    print(result)

    # --- Normal mode: multiple aggs with explicit aliases ---
    print("\n=== Normal mode: multiple aggs with aliases ===")
    result = raw.lazy().group_by("region").agg(
        pl.col("revenue").ds.sum().alias("total_revenue"),
        pl.col("revenue").ds.mean().alias("avg_revenue"),
    ).sort("region").collect()
    print(result)

    # --- Pre-agg mode: single agg ---
    print("\n=== Pre-agg mode: sum ===")
    with PreAggContext(metadata):
        result = pre_agg.lazy().group_by("region").agg(
            pl.col("revenue").ds.sum(),
        ).sort("region").collect()
    print(result)

    print("\n=== Pre-agg mode: mean ===")
    with PreAggContext(metadata):
        result = pre_agg.lazy().group_by("region").agg(
            pl.col("revenue").ds.mean(),
        ).sort("region").collect()
    print(result)

    # --- Pre-agg mode: multiple aggs with explicit aliases ---
    print("\n=== Pre-agg mode: multiple aggs with aliases ===")
    with PreAggContext(metadata):
        result = pre_agg.lazy().group_by("region").agg(
            pl.col("revenue").ds.sum().alias("total_revenue"),
            pl.col("revenue").ds.mean().alias("avg_revenue"),
        ).sort("region").collect()
    print(result)
