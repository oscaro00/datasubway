from __future__ import annotations

from typing import Any

import polars as pl


class PreAggCol:
    """Column reference that knows how to read from pre-aggregated data."""

    def __init__(self, name: str, metadata: dict[str, Any] | None = None):
        self._name = name
        self._meta = metadata
        self._is_pre_agg = (
            metadata is not None and
            name in metadata.get('aggregations', {})
        )

    def sum(self) -> pl.Expr:
        if self._is_pre_agg:
            return pl.col(f'{self._name}-sum').sum()
        return pl.col(self._name).sum()

    def mean(self) -> pl.Expr:
        if self._is_pre_agg:
            return (
                pl.col(f'{self._name}-mean-sum').sum() /
                pl.col(f'{self._name}-mean-count').sum()
            ).alias(self._name)
        return pl.col(self._name).mean()

    def min(self) -> pl.Expr:
        if self._is_pre_agg:
            return pl.col(f'{self._name}-min').min()
        return pl.col(self._name).min()

    def max(self) -> pl.Expr:
        if self._is_pre_agg:
            return pl.col(f'{self._name}-max').max()
        return pl.col(self._name).max()

    def count(self) -> pl.Expr:
        if self._is_pre_agg:
            return pl.col(f'{self._name}-count').sum()
        return pl.col(self._name).count()

    def std(self) -> pl.Expr:
        if self._is_pre_agg:
            n = pl.col(f'{self._name}-std-count').sum()
            s = pl.col(f'{self._name}-std-sum').sum()
            sq = pl.col(f'{self._name}-std-sumsq').sum()
            return ((sq - s.pow(2) / n) / (n - 1)).sqrt().alias(self._name)
        return pl.col(self._name).std()

    def var(self) -> pl.Expr:
        if self._is_pre_agg:
            n = pl.col(f'{self._name}-var-count').sum()
            s = pl.col(f'{self._name}-var-sum').sum()
            sq = pl.col(f'{self._name}-var-sumsq').sum()
            return ((sq - s.pow(2) / n) / (n - 1)).alias(self._name)
        return pl.col(self._name).var()

    def rank(self, method: str = 'average', *, descending: bool = False) -> pl.Expr:
        if self._is_pre_agg:
            return pl.col(f'{self._name}-sum').rank(method, descending=descending).alias(self._name)
        return pl.col(self._name).rank(method, descending=descending)

    def alias(self, name: str) -> AliasedPreAggCol:
        """Allow chaining: col('x').alias('y') for use in select/with_columns."""
        return AliasedPreAggCol(self, name)


class AliasedPreAggCol:
    """Wraps PreAggCol to override the alias on the final expression."""

    def __init__(self, inner: PreAggCol, alias_name: str):
        self._inner = inner
        self._alias = alias_name

    def __getattr__(self, name: str):
        method = getattr(self._inner, name)
        def wrapper(*args, **kwargs):
            expr = method(*args, **kwargs)
            return expr.alias(self._alias)
        return wrapper


def col(name: str, metadata: dict[str, Any] | None = None) -> PreAggCol:
    """Drop-in replacement for pl.col() that supports pre-agg mode."""
    return PreAggCol(name, metadata)


if __name__ == "__main__":
    raw = pl.DataFrame({
        "region": ["A", "A", "B", "B", "B"],
        "revenue": [10, 20, 30, 40, 50],
    })

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

    # --- Normal mode: single agg ---
    print("=== Normal mode: sum ===")
    c = col("revenue")
    result = raw.lazy().group_by("region").agg(
        c.sum(),
    ).sort("region").collect()
    print(result)

    print("\n=== Normal mode: mean ===")
    c = col("revenue")
    result = raw.lazy().group_by("region").agg(
        c.mean(),
    ).sort("region").collect()
    print(result)

    # --- Normal mode: multiple aggs with explicit aliases ---
    print("\n=== Normal mode: multiple aggs with aliases ===")
    c = col("revenue")
    result = raw.lazy().group_by("region").agg(
        c.sum().alias("total_revenue"),
        c.mean().alias("avg_revenue"),
    ).sort("region").collect()
    print(result)

    # --- Pre-agg mode: single agg ---
    print("\n=== Pre-agg mode: sum ===")
    c = col("revenue", metadata)
    result = pre_agg.lazy().group_by("region").agg(
        c.sum(),
    ).sort("region").collect()
    print(result)

    print("\n=== Pre-agg mode: mean ===")
    c = col("revenue", metadata)
    result = pre_agg.lazy().group_by("region").agg(
        c.mean(),
    ).sort("region").collect()
    print(result)

    # --- Pre-agg mode: multiple aggs with explicit aliases ---
    print("\n=== Pre-agg mode: multiple aggs with aliases ===")
    c = col("revenue", metadata)
    result = pre_agg.lazy().group_by("region").agg(
        c.sum().alias("total_revenue"),
        c.mean().alias("avg_revenue"),
    ).sort("region").collect()
    print(result)
