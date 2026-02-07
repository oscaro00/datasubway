from __future__ import annotations

from typing import Any, Iterable, Self

import numpy as np
import polars as pl
from polars._typing import EngineType, ExplainFormat, IntoExpr, IntoExprColumn
from polars.lazyframe.group_by import LazyGroupBy
from polars.lazyframe.in_process import InProcessQuery
from polars.lazyframe.opt_flags import QueryOptFlags


class LazyFrameWrapper:
    def __init__(self, lf: pl.LazyFrame) -> None:
        self.lf = lf
        self.schema = lf.collect_schema()
        self.columns = lf.collect_schema().names()
        self.types = lf.collect_schema().dtypes()

    def filter(
        self,
        *predicates: IntoExprColumn
        | Iterable[IntoExprColumn]
        | bool
        | list[bool]
        | np.ndarray[Any, Any],
        **constraints: Any,
    ) -> Self:

        if len(predicates) == 0 and len(constraints) == 0:
            return self
        else:
            return self.__class__(self.lf.filter(*predicates, **constraints))

    def group_by(
        self,
        *by: IntoExpr | Iterable[IntoExpr],
        maintain_order: bool = False,
        **named_by: IntoExpr,
    ) -> Self | LazyGroupByWrapper:

        if len(by) == 0 and len(named_by) == 0:
            return self.__class__(self.lf)
        else:
            return LazyGroupByWrapper(
                self.lf.group_by(*by, maintain_order=maintain_order, **named_by),
                self.lf,  # need this for access to the schema
            )

    def agg(
        self,
        *aggs: IntoExpr | Iterable[IntoExpr],
        **named_aggs: IntoExpr,
    ) -> LazyFrameWrapper:
        return self.__class__(self.lf.select(*aggs, **named_aggs))

    def collect(
        self,
        *,
        engine: EngineType = "auto",
        background: bool = False,
        optimizations: QueryOptFlags = QueryOptFlags(),
        **_kwargs: Any,
    ) -> pl.DataFrame | InProcessQuery:
        return self.lf.collect(
            engine=engine, background=background, optimizations=optimizations, **_kwargs
        )

    def explain(
        self,
        *,
        format: ExplainFormat = "plain",
        optimized: bool = True,
        engine: EngineType = "auto",
        tree_format: bool | None = None,
        optimizations: QueryOptFlags = QueryOptFlags(),
    ) -> str:
        return self.lf.explain(
            format=format,
            optimized=optimized,
            engine=engine,
            tree_format=tree_format,
            optimizations=optimizations,
        )


class LazyGroupByWrapper:
    def __init__(self, lgb: LazyGroupBy, original_lazyframe: pl.LazyFrame) -> None:
        self.lgb = lgb
        self.schema = original_lazyframe.collect_schema()
        self.columns = original_lazyframe.collect_schema().names()
        self.types = original_lazyframe.collect_schema().dtypes()

    def agg(
        self,
        *aggs: IntoExpr | Iterable[IntoExpr],
        **named_aggs: IntoExpr,
    ) -> LazyFrameWrapper:
        return LazyFrameWrapper(self.lgb.agg(*aggs, **named_aggs))


@pl.api.register_expr_namespace("ds")
class DataSubwayExpr:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def main(self) -> pl.Expr:
        return (p ** (self._expr.log(p).ceil()).cast(pl.Int64)).cast(pl.Int64)


if __name__ == "__main__":
    lf = pl.LazyFrame(
        {
            "store_id": [1, 2, 3, 4, 5],
            "product_id": [9, 8, 7, 6, 5],
            "revenue": [23, 67, 34, 78, 34],
        }
    )

    lfw = LazyFrameWrapper(lf)

    print(lfw.group_by())
    print(lfw.filter().collect())

    print(
        lfw.filter()
        .group_by()
        .agg(pl.col("revenue").sum().alias("total_revenue"))
        .collect()
    )

    print(
        lfw.filter(pl.col("store_id") <= 3)
        .group_by(pl.col("product_id"))
        .agg(pl.col("revenue").sum().alias("total_revenue"))
        .collect()
    )
