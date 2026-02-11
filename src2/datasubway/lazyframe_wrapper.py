from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Self

if TYPE_CHECKING:
    from datasubway.lazygroupby_wrapper import LazyGroupByWrapper

import numpy as np
import polars as pl
from polars._typing import EngineType, ExplainFormat, IntoExpr, IntoExprColumn
from polars.lazyframe.group_by import LazyGroupBy
from polars.lazyframe.in_process import InProcessQuery
from polars.lazyframe.opt_flags import QueryOptFlags
from pre_agg_expr import rewrite_agg_expr


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
        rewritten = [
            rewrite_agg_expr(a, self.schema) if isinstance(a, pl.Expr) else a
            for a in aggs
        ]
        named_rewritten = {
            k: rewrite_agg_expr(v, self.schema) if isinstance(v, pl.Expr) else v
            for k, v in named_aggs.items()
        }
        return LazyFrameWrapper(self.lf.select(*rewritten, **named_rewritten))

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
