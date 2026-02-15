from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from datasubway.lazyframe_wrapper import LazyFrameWrapper

import polars as pl
from polars._typing import IntoExpr
from polars.lazyframe.group_by import LazyGroupBy

from datasubway.pre_agg_expr import rewrite_agg_expr


class LazyGroupByWrapper:
    def __init__(self, lgb: LazyGroupBy, original_lazyframe: pl.LazyFrame) -> None:
        self.lgb = lgb
        self.original_lazyframe = original_lazyframe

    # This allows polars LazyGroupBy methods that don't need custom functionality to work as expect
    # without having to explicitly write them as methods for LazyGroupByWrapper
    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.lgb, name)
        if not callable(attr):
            return attr

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            from datasubway.lazyframe_wrapper import LazyFrameWrapper

            result = attr(*args, **kwargs)
            if isinstance(result, pl.LazyFrame):
                return LazyFrameWrapper(result)
            return result

        return wrapper

    def agg(
        self,
        *aggs: IntoExpr | Iterable[IntoExpr],
        **named_aggs: IntoExpr,
    ) -> LazyFrameWrapper:
        # TODO: the table() method of DataModel could emit a LazyFrameWrapper object with a parameter
        # is_pre_agg. This would avoid a potenial expensive collect_schema() call
        schema = self.original_lazyframe.collect_schema()
        rewritten = [
            rewrite_agg_expr(a, schema) if isinstance(a, pl.Expr) else a for a in aggs
        ]
        named_rewritten = {
            k: rewrite_agg_expr(v, schema) if isinstance(v, pl.Expr) else v
            for k, v in named_aggs.items()
        }
        from datasubway.lazyframe_wrapper import LazyFrameWrapper

        return LazyFrameWrapper(self.lgb.agg(*rewritten, **named_rewritten))
