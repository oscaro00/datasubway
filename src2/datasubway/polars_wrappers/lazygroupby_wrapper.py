from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from datasubway.polars_wrappers.lazyframe_wrapper import LazyFrameWrapper

import polars as pl
from datasubway.polars_wrappers.pre_agg_expr import rewrite_agg_expr
from polars._typing import IntoExpr
from polars.lazyframe.group_by import LazyGroupBy


class LazyGroupByWrapper:
    def __init__(self, lgb: LazyGroupBy, from_pre_agg: bool = False) -> None:
        self.lgb = lgb
        self.from_pre_agg

    # This allows polars LazyGroupBy methods that don't need custom functionality to work as expect
    # without having to explicitly write them as methods for LazyGroupByWrapper
    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.lgb, name)
        if not callable(attr):
            return attr

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            from datasubway.polars_wrappers.lazyframe_wrapper import LazyFrameWrapper

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

        from datasubway.polars_wrappers.lazyframe_wrapper import LazyFrameWrapper

        if not self.from_pre_agg:
            return LazyFrameWrapper(
                self.lgb.agg(*aggs, **named_aggs), self.from_pre_agg
            )

        # if self.from_pre_agg == true, then all aggregations need to be rewritten to match
        # either new column names and potentially corrected calculations
        else:
            rewritten = [
                rewrite_agg_expr(a) if isinstance(a, pl.Expr) else a for a in aggs
            ]
            named_rewritten = {
                k: rewrite_agg_expr(v) if isinstance(v, pl.Expr) else v
                for k, v in named_aggs.items()
            }

            return LazyFrameWrapper(
                self.lgb.agg(*rewritten, **named_rewritten), self.from_pre_agg
            )
