from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from datasubway.lazyframe_wrapper import LazyFrameWrapper

import polars as pl
from polars._typing import IntoExpr
from polars.lazyframe.group_by import LazyGroupBy

from datasubway.pre_agg_expr import rewrite_agg_expr


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
        rewritten = [
            rewrite_agg_expr(a, self.schema) if isinstance(a, pl.Expr) else a
            for a in aggs
        ]
        named_rewritten = {
            k: rewrite_agg_expr(v, self.schema) if isinstance(v, pl.Expr) else v
            for k, v in named_aggs.items()
        }
        from datasubway.lazyframe_wrapper import LazyFrameWrapper

        return LazyFrameWrapper(self.lgb.agg(*rewritten, **named_rewritten))
